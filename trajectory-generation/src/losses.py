import torch
import torch.nn.functional as F
from typing import Dict, Optional, Set, Tuple
from transformers.modeling_outputs import BaseModelOutput


def _collect_special_token_ids(autoencoder) -> Set[int]:
    """Collect known special token IDs from model/tokenizer config if available."""

    specials: Set[int] = set()
    cfg = getattr(autoencoder, 'config', None)

    if cfg is None:
        return specials

    # Core single-value ids that should always be ignored during accuracy calc.
    single_attrs = [
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "mask_token_id",
        "cls_token_id",
        "sep_token_id",
        "decoder_start_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
    ]

    for attr in single_attrs:
        tid = getattr(cfg, attr, None)
        if isinstance(tid, int) and tid >= 0:
            specials.add(int(tid))

    def _should_skip(token_str: Optional[str]) -> bool:
        if token_str is None:
            return False
        lowered = token_str.lower()
        return lowered in {"[unk]"}

    # Some configs expose lists of additional special tokens (e.g. tokenizer adds).
    addl_ids = getattr(cfg, "additional_special_tokens_ids", None)
    addl_tokens = getattr(cfg, "additional_special_tokens", None)
    if addl_ids:
        unk_id = getattr(cfg, "unk_token_id", None)
        for idx, tid in enumerate(addl_ids):
            if tid is None:
                continue
            token_str = None
            if addl_tokens and idx < len(addl_tokens):
                token_str = addl_tokens[idx]
            if unk_id is not None and tid == unk_id:
                continue
            if _should_skip(token_str):
                continue
            specials.add(int(tid))

    # Hugging Face configs sometimes keep richer metadata in dict form.
    cfg_dict = {}
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        cfg_dict = {}

    for key in ("special_tokens_map", "special_tokens_map_extended"):
        value = cfg_dict.get(key)
        if not value:
            continue

        if isinstance(value, dict):
            iterable = value.values()
        else:
            iterable = value

        for item in iterable:
            # map entries may be dicts with "id"/"token" pairs.
            token_str = None
            if isinstance(item, dict):
                tid = item.get("id")
                token_type = item.get("type") or item.get("special_token")
                token_str = item.get("token") or item.get("content")
                if token_type and "unk" in str(token_type).lower():
                    continue
            else:
                tid = None

            if tid is None:
                continue
            if isinstance(tid, (list, tuple)):
                token_list = None
                if isinstance(item, dict):
                    token_list = item.get("tokens") or item.get("token_ids")
                for idx, t in enumerate(tid):
                    if t is None:
                        continue
                    token_for_entry = token_str
                    if token_list and idx < len(token_list):
                        token_for_entry = token_list[idx]
                    if _should_skip(token_for_entry):
                        continue
                    specials.add(int(t))
            elif isinstance(tid, int):
                if _should_skip(token_str):
                    continue
                specials.add(int(tid))

    return specials

def _sequence_lengths_excluding_specials(
    sequence: torch.Tensor,
    *,
    special_ids: Set[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-sample token counts after filtering specials (keeps [UNK])."""

    if sequence.dim() == 1:
        seq = sequence.unsqueeze(0)
    else:
        seq = sequence

    valid = torch.ones(seq.shape, device=seq.device, dtype=torch.bool)

    if attention_mask is not None:
        if attention_mask.dim() == 1:
            mask = attention_mask.unsqueeze(0).bool()
        else:
            mask = attention_mask.bool()
        valid &= mask

    if special_ids:
        specials_tensor = torch.tensor(sorted(special_ids), device=seq.device, dtype=seq.dtype)
        valid &= ~torch.isin(seq, specials_tensor)

    return valid.sum(dim=-1)


def _masked_latent_similarity(x: torch.Tensor, y: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """
    Compute masked cosine similarity and L2 distance between two latent tensors of shape (B, L, D).
    Only positions with attention_mask==1 are considered.
    Returns dict with mean cosine and L2 over batch.
    """
    with torch.no_grad():
        # Align shapes
        B, L, D = x.shape
        mask = attention_mask.bool().view(B, L, 1)
        # Avoid division by zero by adding small epsilon
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        cos = (x_norm * y_norm).sum(dim=-1)  # (B, L)
        l2 = torch.norm(x - y, dim=-1)      # (B, L)
        valid = mask.squeeze(-1)
        cos_masked = cos.masked_select(valid)
        l2_masked = l2.masked_select(valid)
        mean_cos = cos_masked.mean().item() if cos_masked.numel() > 0 else 0.0
        mean_l2 = l2_masked.mean().item() if l2_masked.numel() > 0 else 0.0
        count = int(valid.sum().item())
        return {"cosine": mean_cos, "l2": mean_l2, "count": count}

def _token_accuracies_from_generate(generated_ids: torch.Tensor,
                                    labels: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    autoencoder=None) -> dict:
    """
    Token accuracy that mirrors evaluation filtering: remove special tokens
    (pad/bos/eos/mask) from both reference and generated sequences, then compare.
    """
    with torch.no_grad():
        specials = _collect_special_token_ids(autoencoder) if autoencoder is not None else set()
        B = labels.size(0)
        total_matches_lenient = 0
        total_tokens_lenient = 0
        total_matches_strict = 0
        total_tokens_strict = 0
        for i in range(B):
            valid_len = int(attention_mask[i].sum().item())
            if valid_len == 0:
                continue
            # Reference: drop specials
            ref_seq = labels[i, :valid_len]
            if specials:
                ref_seq = ref_seq[~torch.isin(ref_seq, torch.tensor(list(specials), device=ref_seq.device))]
            # Generated: drop specials (optionally stop at eos not necessary once removed)
            gen_seq = generated_ids[i]
            if specials:
                gen_seq = gen_seq[~torch.isin(gen_seq, torch.tensor(list(specials), device=gen_seq.device))]
            # Align lengths
            min_len = min(ref_seq.size(0), gen_seq.size(0))
            if min_len == 0:
                continue
            ref_seq = ref_seq[:min_len]
            gen_seq = gen_seq[:min_len]
            matches = (ref_seq == gen_seq).sum().item()
            total_matches_lenient += matches
            total_tokens_lenient += min_len
            # Strict: denominator = original ref length after filtering (before truncation)
            strict_den = int(attention_mask[i].sum().item())
            # Recompute original ref filtered length
            ref_full = labels[i, :valid_len]
            if specials:
                ref_full = ref_full[~torch.isin(ref_full, torch.tensor(list(specials), device=ref_full.device))]
            strict_den = ref_full.size(0)
            total_matches_strict += matches
            total_tokens_strict += strict_den
        return {
            "lenient": (total_matches_lenient / total_tokens_lenient) if total_tokens_lenient > 0 else 0.0,
            "strict": (total_matches_strict / total_tokens_strict) if total_tokens_strict > 0 else 0.0
        }


def compute_anchor_loss(
    predicted_x0: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    autoencoder,
    *,
    latent_scale: float = 1.0,
    latent_pca=None,
    training_phase: str = "phase2",
    return_per_sample: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
    """Compute anchor loss as masked negative log-likelihood."""

    device = predicted_x0.device
    if attention_mask is None:
        mask = torch.ones_like(labels, dtype=torch.float32, device=device)
    else:
        mask = attention_mask.to(device=device, dtype=torch.float32)

    scaled_latents = predicted_x0 * latent_scale if latent_scale != 1.0 else predicted_x0
    no_compression = getattr(autoencoder, 'no_compression', False)

    if training_phase == "phase1" or no_compression:
        decoder_latents = latent_pca.unproject(scaled_latents) if latent_pca is not None else scaled_latents
    else:
        if latent_pca is not None:
            decoder_latents = latent_pca.unproject(scaled_latents)
        else:
            decoder_latents = autoencoder.get_decoder_input(scaled_latents.clone())

    encoder_outputs = BaseModelOutput(last_hidden_state=decoder_latents)
    decoder_out = autoencoder(encoder_outputs=encoder_outputs, labels=labels, return_dict=True)
    logits = decoder_out.logits

    vocab_dim = logits.size(-1)
    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_dim),
        labels.view(-1),
        reduction='none'
    ).view_as(labels)

    masked_loss = per_token_loss * mask
    denom = mask.sum().clamp_min(1.0)
    anchor_loss = masked_loss.sum() / denom

    per_sample = None
    if return_per_sample:
        token_counts = mask.sum(dim=-1).clamp_min(1.0)
        per_sample = masked_loss.sum(dim=-1) / token_counts

    stats = {
        "token_count": denom.item(),
        "mean_neg_logp": anchor_loss.item(),
    }

    return anchor_loss, stats, per_sample


def aggregate_poi_distribution(
    poi_probs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Convert per-token POI probabilities into a single marginal distribution."""

    if poi_probs.dim() != 3:
        raise ValueError("poi_probs must have shape [batch, time, vocab]")
    probs = poi_probs
    if attention_mask is not None:
        mask = attention_mask.to(dtype=probs.dtype).unsqueeze(-1)
        probs = probs * mask
    summed = probs.sum(dim=(0, 1))
    total = summed.sum()
    if total <= 0:
        vocab = summed.shape[0]
        return torch.full_like(summed, 1.0 / max(vocab, 1))
    return (summed + epsilon) / (total + epsilon * summed.numel())


def poi_marginal_kl_loss(
    poi_probs: torch.Tensor,
    target_dist: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """KL divergence between model-implied marginals and target CBG marginals."""

    model_dist = aggregate_poi_distribution(poi_probs, attention_mask=attention_mask, epsilon=epsilon)
    if target_dist.dim() != 1:
        raise ValueError("target_dist must be a 1D tensor [vocab]")
    target_dist = target_dist.to(device=model_dist.device, dtype=model_dist.dtype)
    target_dist = (target_dist + epsilon) / (target_dist.sum() + epsilon * target_dist.numel())
    # Use forward KL: KL(target || model) to encourage coverage and reduce mode collapse
    kl = torch.sum(target_dist * (torch.log(target_dist + epsilon) - torch.log(model_dist + epsilon)))
    stats = {
        "agg_kl": float(kl.item()),
    }
    return kl, stats
