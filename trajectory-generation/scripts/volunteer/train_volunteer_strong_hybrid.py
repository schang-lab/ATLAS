#!/usr/bin/env python3
"""Train a hybrid VOLUNTEER strong model with a prior-sampled aggregate branch.

This leaves `train_volunteer_setups.py` untouched.

Hybrid objective:
  1. Standard supervised VOLUNTEER ELBO on labeled trajectories.
  2. An auxiliary prior-sampled decoder loss that matches batch-level POI
     distributions from `z ~ N(0, I)` to the real labeled batch.

The default aggregate target is grouped by `(age_bin, gender_id)` so the model
is pushed to preserve demographic differences under prior sampling, not only
under teacher forcing.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.losses import aggregate_poi_distribution
from train_volunteer_setups import (
    VolunteerSetupTrainer,
    _build_supervised_optimizer,
    _set_backbone_lr,
    _set_scheduler_backbone_base_lr,
    load_yaml,
    set_seed,
    wandb,
)


class VolunteerStrongHybridTrainer(VolunteerSetupTrainer):
    def __init__(self, cfg: Dict[str, object]):
        super().__init__(cfg, setup_override="strong")
        train_cfg = self.cfg["training"]
        self.lambda_supervised = float(train_cfg.get("lambda_supervised", 1.0))
        self.lambda_prior = float(train_cfg.get("lambda_prior", 1.0))
        self.prior_loss_type = str(train_cfg.get("prior_loss_type", "js")).lower().strip()
        self.prior_group_by = str(train_cfg.get("prior_group_by", "age_gender")).lower().strip()
        self.prior_pos_mode = str(train_cfg.get("prior_pos_mode", "ramp")).lower().strip()
        self.prior_pos_step_minutes = float(train_cfg.get("prior_pos_step_minutes", 30.0))
        self.prior_val_batches = max(1, int(train_cfg.get("prior_val_batches", 20)))
        self.best_metric = str(train_cfg.get("best_metric", "prior")).lower().strip()

        if self.prior_loss_type not in {"kl", "js", "tv", "hellinger"}:
            raise ValueError("training.prior_loss_type must be one of: kl, js, tv, hellinger")
        if self.prior_group_by not in {"age_gender", "batch"}:
            raise ValueError("training.prior_group_by must be one of: age_gender, batch")
        if self.prior_pos_mode not in {"zero", "ramp"}:
            raise ValueError("training.prior_pos_mode must be one of: zero, ramp")
        if self.best_metric not in {"prior", "supervised", "combined"}:
            raise ValueError("training.best_metric must be one of: prior, supervised, combined")

        self.target_vocab = self.model.vocab_size - self.num_special_tokens
        if self.target_vocab <= 0:
            raise ValueError("model.vocab_size must exceed num_special_tokens")

    def _build_prior_pos(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.prior_pos_mode == "zero":
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        base = torch.arange(seq_len, device=device, dtype=torch.float32)
        return base.unsqueeze(0).expand(batch_size, -1) * self.prior_pos_step_minutes

    def _prior_poi_probs_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = batch["loc"].shape
        z = torch.randn(batch_size, seq_len, self.model.latent_size, device=self.device)
        pos = self._build_prior_pos(batch_size, seq_len, self.device)
        demo_emb = self.model.demo_block(
            batch["age_bin"],
            batch["gender_id"],
            batch["home"],
            batch["work"],
            seq_len,
        )
        dec_input = self.model._build_decoder_input(z, demo_emb, pos)
        _, _, logits = self.model.decoder(dec_input)
        if self.num_special_tokens > 0:
            logits = logits.clone()
            logits[..., : self.num_special_tokens] = logits[..., : self.num_special_tokens] - 1e4
        probs = F.softmax(logits, dim=-1)
        if self.num_special_tokens > 0:
            probs = probs[..., self.num_special_tokens :]
        if probs.size(-1) != self.target_vocab:
            raise ValueError(
                f"Prior probs dim ({probs.size(-1)}) != target vocab ({self.target_vocab})."
            )
        return probs

    def _real_batch_distribution(
        self,
        loc: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        valid_mask = mask.bool() & (loc >= self.num_special_tokens)
        if not bool(valid_mask.any()):
            return torch.full(
                (self.target_vocab,),
                1.0 / self.target_vocab,
                device=loc.device,
                dtype=torch.float32,
            ), 0

        tokens = loc.masked_select(valid_mask) - self.num_special_tokens
        tokens = tokens.long().clamp_(0, self.target_vocab - 1)
        counts = torch.bincount(tokens, minlength=self.target_vocab).to(dtype=torch.float32)
        total = int(tokens.numel())
        dist = counts / counts.sum().clamp_min(1.0)
        return dist, total

    def _distribution_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        target = (target + epsilon) / (target.sum() + epsilon * target.numel())
        pred = (pred + epsilon) / (pred.sum() + epsilon * pred.numel())

        if self.prior_loss_type == "kl":
            return torch.sum(target * (torch.log(target) - torch.log(pred)))
        if self.prior_loss_type == "js":
            mix = 0.5 * (target + pred)
            return 0.5 * torch.sum(target * (torch.log(target) - torch.log(mix))) + 0.5 * torch.sum(
                pred * (torch.log(pred) - torch.log(mix))
            )
        if self.prior_loss_type == "tv":
            return 0.5 * torch.sum(torch.abs(target - pred))
        if self.prior_loss_type == "hellinger":
            return (1.0 / math.sqrt(2.0)) * torch.sqrt(torch.sum((torch.sqrt(target) - torch.sqrt(pred)) ** 2))
        raise ValueError(f"Unknown prior loss type: {self.prior_loss_type}")

    def _age_gender_prior_loss(self, batch: Dict[str, torch.Tensor], poi_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        age = batch["age_bin"].long()
        gender = batch["gender_id"].long()
        mask = batch["mask"]
        loc = batch["loc"]

        group_losses: List[torch.Tensor] = []
        token_weights: List[int] = []
        groups_used = 0

        for age_bin in range(self.num_age_bins):
            for gender_id in range(self.num_genders):
                group_sel = (age == age_bin) & (gender == gender_id)
                if not bool(group_sel.any()):
                    continue

                target_dist, num_tokens = self._real_batch_distribution(loc[group_sel], mask[group_sel])
                if num_tokens <= 0:
                    continue

                pred_dist = aggregate_poi_distribution(
                    poi_probs[group_sel],
                    attention_mask=mask[group_sel],
                    epsilon=1e-8,
                )
                group_losses.append(self._distribution_loss(target_dist, pred_dist))
                token_weights.append(num_tokens)
                groups_used += 1

        if not group_losses:
            zero = poi_probs.sum() * 0.0
            return zero, {"prior_loss": 0.0, "prior_groups_used": 0.0}

        weights = torch.tensor(token_weights, device=poi_probs.device, dtype=poi_probs.dtype)
        weights = weights / weights.sum().clamp_min(1.0)
        loss = torch.stack(group_losses)
        weighted = torch.sum(loss * weights)
        return weighted, {
            "prior_loss": float(weighted.item()),
            "prior_groups_used": float(groups_used),
        }

    def _batch_prior_loss(self, batch: Dict[str, torch.Tensor], poi_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        target_dist, num_tokens = self._real_batch_distribution(batch["loc"], batch["mask"])
        if num_tokens <= 0:
            zero = poi_probs.sum() * 0.0
            return zero, {"prior_loss": 0.0, "prior_groups_used": 0.0}

        pred_dist = aggregate_poi_distribution(
            poi_probs,
            attention_mask=batch["mask"],
            epsilon=1e-8,
        )
        loss = self._distribution_loss(target_dist, pred_dist)
        return loss, {"prior_loss": float(loss.item()), "prior_groups_used": 1.0}

    def _prior_loss_on_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        poi_probs = self._prior_poi_probs_from_batch(batch)
        if self.prior_group_by == "age_gender":
            return self._age_gender_prior_loss(batch, poi_probs)
        return self._batch_prior_loss(batch, poi_probs)

    def _validate_prior(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": float("nan"), "groups_used": float("nan")}

        self.model.eval()
        total_loss = 0.0
        total_groups = 0.0
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.prior_val_batches:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                prior_loss, prior_stats = self._prior_loss_on_batch(batch)
                total_loss += float(prior_loss.item())
                total_groups += float(prior_stats["prior_groups_used"])
                count += 1

        self.model.train()
        if count == 0:
            return {"loss": float("nan"), "groups_used": float("nan")}
        return {
            "loss": total_loss / count,
            "groups_used": total_groups / count,
        }

    def _selection_value(self, supervised_val: Dict[str, float], prior_val: Dict[str, float]) -> float:
        if self.best_metric == "supervised":
            return float(supervised_val["loss"])
        if self.best_metric == "combined":
            return self.lambda_supervised * float(supervised_val["loss"]) + self.lambda_prior * float(prior_val["loss"])
        return float(prior_val["loss"])

    def train_hybrid(self) -> None:
        train_cfg = self.cfg["training"]
        epochs = int(train_cfg.get("epochs", train_cfg.get("phase1_epochs", 30)))
        lr = float(train_cfg.get("lr", train_cfg.get("phase1_lr", 1e-4)))
        save_every_epochs = int(train_cfg.get("save_every_epochs", 0) or 0)
        optimizer = _build_supervised_optimizer(self.model, train_cfg)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        demo_only_steps = int(train_cfg.get("demo_only_steps", 0) or 0)
        freeze_backbone_for_demo = bool(train_cfg.get("freeze_backbone_for_demo", False))
        demo_warmup_active = (not freeze_backbone_for_demo) and demo_only_steps > 0
        backbone_unfrozen = not demo_warmup_active
        backbone_lr_scale = float(train_cfg.get("backbone_lr_scale", 0.1))
        best_val = float("inf")
        global_step = 0

        print("=" * 60)
        print("STRONG HYBRID: supervised ELBO + prior aggregate loss")
        print("=" * 60)
        print(
            f"lambda_supervised={self.lambda_supervised} "
            f"lambda_prior={self.lambda_prior} "
            f"prior_group_by={self.prior_group_by} "
            f"prior_loss_type={self.prior_loss_type} "
            f"prior_pos_mode={self.prior_pos_mode}"
        )

        for epoch in range(epochs):
            self.model.train()
            stats: Dict[str, List[float]] = {
                "total_loss": [],
                "supervised_loss": [],
                "prior_loss": [],
                "kl": [],
                "nll_loc": [],
                "nll_tim": [],
                "prior_groups_used": [],
            }

            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch)
                supervised = self.model.vae_loss(output, batch)
                prior_loss, prior_stats = self._prior_loss_on_batch(batch)

                total_loss = self.lambda_supervised * supervised["loss"] + self.lambda_prior * prior_loss

                optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                global_step += 1

                if demo_warmup_active and not backbone_unfrozen and global_step >= demo_only_steps:
                    backbone_lr = lr * backbone_lr_scale
                    _set_backbone_lr(optimizer, backbone_lr)
                    _set_scheduler_backbone_base_lr(scheduler, optimizer, backbone_lr)
                    backbone_unfrozen = True
                    print(
                        f"[strong_hybrid] demo-only warmup complete at step={global_step}; "
                        f"backbone_lr={backbone_lr:.6g}"
                    )

                stats["total_loss"].append(float(total_loss.item()))
                stats["supervised_loss"].append(float(supervised["loss"].item()))
                stats["prior_loss"].append(float(prior_loss.item()))
                stats["kl"].append(float(supervised["kl"].item()))
                stats["nll_loc"].append(float(supervised["nll_loc"].item()))
                stats["nll_tim"].append(float(supervised["nll_tim"].item()))
                stats["prior_groups_used"].append(float(prior_stats["prior_groups_used"]))

                if batch_idx % self.log_every == 0:
                    backbone_group_lr = next(
                        (group["lr"] for group in optimizer.param_groups if group.get("name") == "backbone"),
                        optimizer.param_groups[0]["lr"],
                    )
                    demo_group_lr = next(
                        (group["lr"] for group in optimizer.param_groups if group.get("name") == "demo"),
                        optimizer.param_groups[0]["lr"],
                    )
                    print(
                        f"[strong_hybrid] epoch={epoch} batch={batch_idx} "
                        f"total={total_loss.item():.4f} supervised={supervised['loss'].item():.4f} "
                        f"prior={prior_loss.item():.4f} kl={supervised['kl'].item():.4f} "
                        f"nll_loc={supervised['nll_loc'].item():.4f} nll_tim={supervised['nll_tim'].item():.4f} "
                        f"groups={prior_stats['prior_groups_used']:.1f} "
                        f"backbone_lr={backbone_group_lr:.6g} demo_lr={demo_group_lr:.6g}"
                    )

            scheduler.step()

            avg = {key: float(np.mean(values)) for key, values in stats.items() if values}
            val_supervised = self._validate_vae(demo_source="data")
            val_prior = self._validate_prior()
            selection = self._selection_value(val_supervised, val_prior)

            print(
                f"[strong_hybrid] epoch={epoch} train={avg} "
                f"val_supervised={val_supervised} val_prior={val_prior} "
                f"selection={selection:.6f}"
            )

            if self.use_wandb:
                log = {f"strong_hybrid/train_{key}": value for key, value in avg.items()}
                log.update({f"strong_hybrid/val_supervised_{key}": value for key, value in val_supervised.items()})
                log.update({f"strong_hybrid/val_prior_{key}": value for key, value in val_prior.items()})
                log["strong_hybrid/selection_metric"] = selection
                wandb.log(log, step=epoch)

            if not np.isnan(selection) and selection < best_val:
                best_val = selection
                self._save_checkpoint("strong_hybrid_best.pt", step=epoch)

            if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
                self._save_checkpoint(f"strong_hybrid_epoch{epoch + 1}.pt", step=epoch + 1)

        self._save_checkpoint("strong_hybrid_final.pt", step=epochs)

    def run(self) -> None:
        with open(self.output_dir / "config.yaml", "w") as f:
            cfg_to_save = dict(self.cfg)
            cfg_to_save.setdefault("model", {})
            cfg_to_save["model"]["use_demo_condition"] = self.model.use_demo_condition
            cfg_to_save["model"]["demo_conditioning_type"] = self.model.demo_conditioning_type
            cfg_to_save.setdefault("experiment", {})
            cfg_to_save["experiment"]["setup"] = "strong_hybrid"
            yaml.safe_dump(cfg_to_save, f)

        self.train_hybrid()
        if self.use_wandb:
            wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid VOLUNTEER strong model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.device:
        cfg["device"] = args.device
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    cfg["seed"] = seed
    trainer = VolunteerStrongHybridTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
