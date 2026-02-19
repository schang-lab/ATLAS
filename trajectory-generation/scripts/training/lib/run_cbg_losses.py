from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from src.losses import aggregate_poi_distribution, poi_marginal_kl_loss


def poi_mask_index_tensor(trainer, *, device: torch.device) -> Optional[torch.Tensor]:
    if not getattr(trainer, "poi_token_mask_enabled", False):
        return None
    idx = getattr(trainer, "poi_token_mask_indices", None) or []
    if not idx:
        return None
    return torch.tensor(idx, device=device, dtype=torch.long)


def apply_poi_token_mask_to_probs(
    trainer,
    poi_probs: torch.Tensor,
    *,
    stats: Optional[Dict[str, float]] = None,
    epsilon: Optional[float] = None,
) -> torch.Tensor:
    idx = poi_mask_index_tensor(trainer, device=poi_probs.device)
    if idx is None:
        return poi_probs

    eps = float(trainer.poi_token_mask_eps if epsilon is None else epsilon)
    V = int(poi_probs.size(-1))
    if idx.numel() >= V:
        raise ValueError("POI token mask would remove all tokens (idx >= V).")

    if stats is not None:
        masked_mass = poi_probs.index_select(-1, idx).sum(dim=-1).mean().detach().item()
        stats["poi_masked_mass"] = float(masked_mass)
        stats["poi_mask_num_tokens"] = float(idx.numel())

    poi_probs = poi_probs.clone()
    poi_probs.index_fill_(-1, idx, 0.0)

    if not bool(getattr(trainer, "poi_token_mask_renormalize", True)):
        return poi_probs

    denom = poi_probs.sum(dim=-1, keepdim=True)
    bad = denom.squeeze(-1) <= eps
    denom = denom.clamp_min(eps)
    poi_probs = poi_probs / denom
    if bad.any():
        base = torch.ones((V,), device=poi_probs.device, dtype=poi_probs.dtype)
        base.index_fill_(0, idx, 0.0)
        support = base.sum().clamp_min(1.0)
        base = base / support
        poi_probs[bad] = base
    return poi_probs


def apply_poi_token_mask_to_dist(
    trainer,
    dist: torch.Tensor,
    *,
    epsilon: Optional[float] = None,
) -> torch.Tensor:
    idx = poi_mask_index_tensor(trainer, device=dist.device)
    if idx is None:
        return dist
    eps = float(trainer.poi_token_mask_eps if epsilon is None else epsilon)
    out = dist.clone()
    out.index_fill_(-1, idx, 0.0)
    denom = out.sum(dim=-1, keepdim=True).clamp_min(eps)
    out = out / denom
    return out


def transition_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be [B, T]")
    if attention_mask.size(1) <= 1:
        return attention_mask.new_zeros((attention_mask.size(0), 0))
    return attention_mask[:, :-1].long() * attention_mask[:, 1:].long()


def category_transition_probs(cat_probs: torch.Tensor) -> torch.Tensor:
    if cat_probs.dim() != 3:
        raise ValueError("cat_probs must be [B, T, C]")
    if cat_probs.size(1) <= 1:
        B = int(cat_probs.size(0))
        C = int(cat_probs.size(2))
        return cat_probs.new_zeros((B, 0, C * C))
    p0 = cat_probs[:, :-1, :]
    p1 = cat_probs[:, 1:, :]
    outer = torch.einsum("btc,btd->btcd", p0, p1)
    return outer.reshape(cat_probs.size(0), cat_probs.size(1) - 1, -1)


def normalize_dist(x: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    x = torch.clamp(x, min=0.0)
    return (x + epsilon) / (x.sum() + epsilon * x.numel())


def distribution_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    loss_type: str,
    epsilon: float,
) -> torch.Tensor:
    p = normalize_dist(target, epsilon=epsilon)
    q = normalize_dist(pred, epsilon=epsilon)

    if loss_type == "kl":
        return torch.sum(p * (torch.log(p + epsilon) - torch.log(q + epsilon)))
    if loss_type == "js":
        m = normalize_dist(0.5 * (p + q), epsilon=epsilon)
        kl_pm = torch.sum(p * (torch.log(p + epsilon) - torch.log(m + epsilon)))
        kl_qm = torch.sum(q * (torch.log(q + epsilon) - torch.log(m + epsilon)))
        return 0.5 * (kl_pm + kl_qm)
    if loss_type == "tv":
        return 0.5 * torch.sum(torch.abs(p - q))
    if loss_type == "hellinger":
        sp = torch.sqrt(p + epsilon)
        sq = torch.sqrt(q + epsilon)
        return 0.5 * torch.sum((sp - sq) ** 2)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def aggregate_feature_losses(
    trainer,
    *,
    cbg: str,
    poi_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    age_raw: torch.Tensor,
    gender_raw: torch.Tensor,
    poi_store=None,
    cache=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    stats: Dict[str, float] = {}
    src_store = poi_store or trainer.poi_store
    src_cache = cache or trainer.cache

    def _loss_for(
        probs: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        *,
        prefix: str,
        apply_poi_token_mask: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if trainer.llp_enabled and trainer.num_age_bins > 0 and trainer.num_genders > 0:
            loss, lstat = trainer._llp_mixture_kl(
                probs,
                mask,
                cbg=cbg,
                batch_age=age_raw,
                batch_gender=gender_raw,
                cache=src_cache,
                poi_store=src_store,
                target_dist=target,
                apply_poi_token_mask=apply_poi_token_mask,
            )
        else:
            if trainer.aggregate_loss_type == "kl":
                loss, lstat = poi_marginal_kl_loss(probs, target, attention_mask=mask)
            else:
                pred = aggregate_poi_distribution(probs, attention_mask=mask, epsilon=trainer.aggregate_loss_eps)
                loss = distribution_loss(
                    target.to(device=pred.device, dtype=pred.dtype),
                    pred,
                    loss_type=trainer.aggregate_loss_type,
                    epsilon=trainer.aggregate_loss_eps,
                )
                lstat = {f"agg_{trainer.aggregate_loss_type}": float(loss.item())}
        out_stat: Dict[str, float] = {}
        for k, v in lstat.items():
            out_stat[f"{prefix}/{k}"] = float(v)
        return loss, out_stat

    total = poi_probs.new_zeros(())

    if trainer.aggregate_feature == "poi":
        target = src_store.get_distribution(cbg, device=poi_probs.device)
        if trainer.poi_token_mask_enabled:
            target = apply_poi_token_mask_to_dist(trainer, target, epsilon=trainer.aggregate_loss_eps)
        loss, lstat = _loss_for(
            poi_probs,
            attention_mask,
            target,
            prefix="poi",
            apply_poi_token_mask=True,
        )
        total = total + loss
        stats.update(lstat)
        stats["agg_loss_poi"] = float(loss.detach().item())
        return total, stats

    if trainer.category_map is None:
        raise RuntimeError("Category map not initialized but aggregate_feature requires it.")

    if trainer.aggregate_feature in {"category", "category+transition"}:
        cat_probs = trainer.category_map.map_probs(poi_probs, epsilon=trainer.aggregate_loss_eps)
        poi_target = src_store.get_distribution(cbg, device=poi_probs.device)
        cat_target = trainer.category_map.map_dist(poi_target, epsilon=trainer.aggregate_loss_eps)
        loss_cat, lstat = _loss_for(
            cat_probs,
            attention_mask,
            cat_target,
            prefix="cat",
            apply_poi_token_mask=False,
        )
        w = float(trainer.category_weight)
        total = total + w * loss_cat
        stats.update(lstat)
        stats["agg_loss_cat"] = float(loss_cat.detach().item())
        stats["agg_weight_cat"] = float(w)
    else:
        cat_probs = trainer.category_map.map_probs(poi_probs, epsilon=trainer.aggregate_loss_eps)

    if trainer.aggregate_feature in {"category_transition", "category+transition"}:
        if trainer.category_transition_store is None:
            raise RuntimeError("Transition store not initialized but aggregate_feature requests transitions.")
        trans_probs = category_transition_probs(cat_probs)
        trans_m = transition_mask(attention_mask)
        trans_target = trainer.category_transition_store.get_distribution(cbg, device=poi_probs.device)
        loss_tr, lstat = _loss_for(
            trans_probs,
            trans_m,
            trans_target,
            prefix="cat_trans",
            apply_poi_token_mask=False,
        )
        w = float(trainer.transition_weight)
        total = total + w * loss_tr
        stats.update(lstat)
        stats["agg_loss_cat_trans"] = float(loss_tr.detach().item())
        stats["agg_weight_cat_trans"] = float(w)

    return total, stats
