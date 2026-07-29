#!/usr/bin/env python3
"""Train VOLUNTEER for the three comparison setups.

Setups:
  baseline: home/work-conditioned VAE pretraining, no demographic signal.
  strong:   start from baseline, enable demographic conditioning, train with
            per-trajectory demographic labels.
  atlas:    start from baseline, enable demographic conditioning, train from
            aggregate CBG POI marginals and CBG demographic proportions.

Example:
    python trajectory-generation/scripts/volunteer/train_volunteer_setups.py \
        --config trajectory-generation/scripts/volunteer/config_volunteer_baseline.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from volunteer_dataset import build_dataloaders
from volunteer_model import VolunteerVAE
from src.data import CBGConditionCache, POIMarginalStore
from src.losses import aggregate_poi_distribution, poi_marginal_kl_loss

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

DEMO_PARAM_PREFIXES = (
    "demo_block.age_emb",
    "demo_block.gender_emb",
    "demo_block.demo_scale",
    "demo_block.demo_shift",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def distribution_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    loss_type: str = "kl",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    target = (target + epsilon) / (target.sum() + epsilon * target.numel())
    pred = (pred + epsilon) / (pred.sum() + epsilon * pred.numel())
    if loss_type == "kl":
        return torch.sum(target * (torch.log(target) - torch.log(pred)))
    if loss_type == "js":
        mix = 0.5 * (target + pred)
        return 0.5 * torch.sum(target * (torch.log(target) - torch.log(mix))) + 0.5 * torch.sum(
            pred * (torch.log(pred) - torch.log(mix))
        )
    if loss_type == "tv":
        return 0.5 * torch.sum(torch.abs(target - pred))
    if loss_type == "hellinger":
        return (1.0 / np.sqrt(2.0)) * torch.sqrt(torch.sum((torch.sqrt(target) - torch.sqrt(pred)) ** 2))
    raise ValueError(f"Unknown aggregate loss type: {loss_type}")


def load_compatible_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and torch.is_tensor(value) and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(f"No compatible tensors found in checkpoint: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  compatible tensors: {len(compatible)} / {len(current)}")
    if skipped:
        print(f"  skipped tensors: {skipped}")
    if missing:
        print(f"  missing tensors initialized from current model: {missing}")
    if unexpected:
        print(f"  unexpected tensors ignored: {unexpected}")


def _is_demo_param(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in DEMO_PARAM_PREFIXES)


def _build_supervised_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> optim.Optimizer:
    lr = float(train_cfg.get("lr", train_cfg.get("phase1_lr", 1e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    demo_only_steps = int(train_cfg.get("demo_only_steps", 0) or 0)
    freeze_backbone = bool(train_cfg.get("freeze_backbone_for_demo", False))
    requested_demo_tuning = freeze_backbone or demo_only_steps > 0
    use_demo_groups = bool(getattr(model, "use_demo_condition", False)) and requested_demo_tuning

    if not use_demo_groups:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    demo_params = [param for name, param in named_params if _is_demo_param(name)]
    backbone_params = [param for name, param in named_params if not _is_demo_param(name)]
    if not demo_params:
        raise ValueError("Demo-only warm start requested, but no demo branch parameters were found.")

    demo_count = sum(param.numel() for param in demo_params)
    backbone_count = sum(param.numel() for param in backbone_params)
    print(
        "Demo warm start enabled: "
        f"freeze_backbone_for_demo={freeze_backbone}, demo_only_steps={demo_only_steps}, "
        f"demo_params={demo_count:,}, backbone_params={backbone_count:,}"
    )

    return optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": 0.0,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": demo_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "demo",
            },
        ]
    )


def _set_backbone_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            group["lr"] = lr


def _set_scheduler_backbone_base_lr(scheduler: CosineAnnealingLR, optimizer: optim.Optimizer, lr: float) -> None:
    if not hasattr(scheduler, "base_lrs"):
        return
    for idx, group in enumerate(optimizer.param_groups):
        if group.get("name") == "backbone":
            scheduler.base_lrs[idx] = lr


class VolunteerSetupTrainer:
    def __init__(self, cfg: Dict[str, Any], *, setup_override: Optional[str] = None):
        self.cfg = cfg
        exp_cfg = cfg.get("experiment", {}) or {}
        self.setup = str(setup_override or exp_cfg.get("setup", "baseline")).lower().strip()
        if self.setup not in {"baseline", "strong", "atlas"}:
            raise ValueError("experiment.setup must be one of: baseline, strong, atlas")

        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_cfg = dict(cfg["model"])
        model_cfg["use_demo_condition"] = self.setup in {"strong", "atlas"}
        self.model = VolunteerVAE(model_cfg).to(self.device)
        self.model_cfg = model_cfg
        print(f"Setup: {self.setup}")
        print(f"Model use_demo_condition={self.model.use_demo_condition}")
        print(f"Model demo_conditioning_type={self.model.demo_conditioning_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        data_cfg = cfg["data"]
        train_cfg = cfg["training"]
        self.loaders = build_dataloaders(
            data_root=data_cfg["split_data_dir"],
            batch_size=int(train_cfg.get("batch_size", 64)),
            max_seq_len=int(model_cfg.get("max_seq_len", 64)),
            num_workers=int(data_cfg.get("num_workers", 4)),
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders.get("val")
        train_has_demo = bool(getattr(self.train_loader.dataset, "has_demo_attrs", False))
        if self.setup == "strong" and not train_has_demo:
            raise ValueError(
                "Strong setup requires per-trajectory demo labels. Use a split with "
                "all_attr_results_with_demo.npy, not a nodemo split."
            )

        self.output_dir = Path(train_cfg.get("output_dir", f"runs/volunteer_{self.setup}"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = int(train_cfg.get("log_every", 50))
        self.save_every = int(train_cfg.get("save_every", 1000))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.num_special_tokens = int(model_cfg.get("num_special_tokens", 5))
        self.num_age_bins = int(model_cfg.get("num_age_bins", 4))
        self.num_genders = int(model_cfg.get("num_genders", 2))

        self.use_wandb = bool(train_cfg.get("wandb", False))
        if self.use_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is not installed, but training.wandb is true.")
            wandb.init(
                project=str(train_cfg.get("wandb_project", "volunteer-atlas")),
                name=train_cfg.get("wandb_run_name") or f"volunteer-{self.setup}",
                config=cfg,
            )

        init_checkpoint = exp_cfg.get("init_checkpoint") or train_cfg.get("init_checkpoint")
        if self.setup in {"strong", "atlas"} and not init_checkpoint:
            raise ValueError(f"{self.setup} setup must set experiment.init_checkpoint to the baseline checkpoint.")
        if init_checkpoint:
            load_compatible_checkpoint(self.model, str(init_checkpoint), self.device)

        atlas_cfg = cfg.get("atlas", {}) or {}
        self.cache: Optional[CBGConditionCache] = None
        self.poi_store: Optional[POIMarginalStore] = None
        self.cbgs: List[str] = []
        self.val_cache: Optional[CBGConditionCache] = None
        self.val_poi_store: Optional[POIMarginalStore] = None
        self.val_cbgs: List[str] = []
        self.val_agg_enabled = False
        self.val_agg_num_batches = 0
        self.val_agg_batch_size = 0
        self.val_agg_demo_source = "pi"
        self.val_agg_log_by_cbg = False
        self.val_agg_max_cbgs_to_log = 0
        if self.setup == "atlas":
            self.cache = CBGConditionCache(str(atlas_cfg["cbg_cache_dir"]))
            self.poi_store = POIMarginalStore(str(atlas_cfg["poi_marginal_csv"]))
            allowed = set(str(c) for c in atlas_cfg.get("allowed_cbgs", []) or [])
            cbgs = set(self.cache.available_cbgs()) & set(self.poi_store.available_cbgs())
            if allowed:
                cbgs &= allowed
            if not cbgs:
                raise ValueError("No overlapping CBGs between atlas cache and POI marginals.")
            self.cbgs = sorted(cbgs)
            print(f"ATLAS CBGs: {len(self.cbgs)}")

            val_cfg = atlas_cfg.get("validation", {}) or {}
            self.val_agg_enabled = bool(val_cfg.get("enabled", False))
            if self.val_agg_enabled:
                val_cache_dir = val_cfg.get("cbg_cache_dir")
                val_poi_csv = val_cfg.get("poi_marginal_csv")
                if not val_cache_dir or not val_poi_csv:
                    raise ValueError("atlas.validation.enabled requires cbg_cache_dir and poi_marginal_csv.")
                self.val_cache = CBGConditionCache(str(val_cache_dir))
                self.val_poi_store = POIMarginalStore(str(val_poi_csv))
                val_allowed = set(str(c) for c in val_cfg.get("allowed_cbgs", []) or [])
                val_cbgs = set(self.val_cache.available_cbgs()) & set(self.val_poi_store.available_cbgs())
                if val_allowed:
                    val_cbgs &= val_allowed
                if not val_cbgs:
                    raise ValueError("No overlapping CBGs between atlas validation cache and POI marginals.")
                self.val_cbgs = sorted(val_cbgs)
                self.val_agg_num_batches = max(1, int(val_cfg.get("num_batches", 10) or 10))
                self.val_agg_batch_size = int(
                    val_cfg.get(
                        "batch_size",
                        self.cfg["training"].get("aggregate_batch_size", self.cfg["training"].get("batch_size", 256)),
                    )
                )
                demo_source = str(val_cfg.get("demo_source", "inherit")).lower().strip()
                if demo_source == "inherit":
                    demo_source = str(atlas_cfg.get("demo_source", "pi")).lower().strip()
                if demo_source not in {"pi", "cache"}:
                    raise ValueError("atlas.validation.demo_source must be one of: inherit, pi, cache")
                self.val_agg_demo_source = demo_source
                self.val_agg_log_by_cbg = bool(val_cfg.get("log_by_cbg", False))
                self.val_agg_max_cbgs_to_log = max(0, int(val_cfg.get("max_cbgs_to_log", 0) or 0))
                print(f"ATLAS validation CBGs: {len(self.val_cbgs)}")

    def _save_checkpoint(self, name: str, step: Optional[int] = None) -> None:
        path = self.output_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.cfg,
                "setup": self.setup,
                "step": step,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def _validate_vae(self, *, demo_source: str = "data") -> Dict[str, float]:
        """Validate with the per-trajectory VAE objective.

        demo_source="null" masks age/gender during validation, matching the
        default ATLAS reconstruction regularizer.
        """
        if demo_source not in {"data", "null"}:
            raise ValueError("demo_source must be 'data' or 'null'")
        if self.val_loader is None:
            return {key: float("nan") for key in ("loss", "kl", "nll_loc", "nll_tim")}
        self.model.eval()
        totals = {"loss": 0.0, "kl": 0.0, "nll_loc": 0.0, "nll_tim": 0.0}
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if demo_source == "null":
                    batch["age_bin"] = torch.full_like(batch["age_bin"], -1)
                    batch["gender_id"] = torch.full_like(batch["gender_id"], -1)
                output = self.model(batch)
                losses = VolunteerVAE.vae_loss(output, batch)
                bsz = int(batch["loc"].size(0))
                for key in totals:
                    totals[key] += float(losses[key].item()) * bsz
                count += bsz
        self.model.train()
        return {key: value / max(count, 1) for key, value in totals.items()}

    def _validate_supervised(self) -> float:
        return self._validate_vae(demo_source="data")["loss"]

    def train_supervised(self) -> None:
        """Train baseline or strong setup with per-trajectory ELBO."""
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
        steps_per_epoch = len(self.train_loader)
        print(
            f"[{self.setup}] epochs={epochs} steps_per_epoch={steps_per_epoch} "
            f"total_steps={epochs * steps_per_epoch}"
        )

        print("=" * 60)
        print(f"{self.setup.upper()}: supervised VAE training")
        print("=" * 60)
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            stats = {"loss": [], "kl": [], "nll_loc": [], "nll_tim": []}
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch)
                losses = VolunteerVAE.vae_loss(output, batch)

                optimizer.zero_grad()
                losses["loss"].backward()
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
                        f"[{self.setup}] demo-only warmup complete at step={global_step}; "
                        f"backbone_lr={backbone_lr:.6g}"
                    )

                for key in stats:
                    stats[key].append(float(losses[key].item()))

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
                        f"[{self.setup}] epoch={epoch} batch={batch_idx} "
                        f"loss={losses['loss'].item():.4f} kl={losses['kl'].item():.4f} "
                        f"nll_loc={losses['nll_loc'].item():.4f} nll_tim={losses['nll_tim'].item():.4f} "
                        f"backbone_lr={backbone_group_lr:.6g} demo_lr={demo_group_lr:.6g}"
                    )

            scheduler.step()
            avg = {key: float(np.mean(values)) for key, values in stats.items()}
            val = self._validate_vae(demo_source="data")
            val_loss = val["loss"]
            print(f"[{self.setup}] epoch={epoch} train={avg} val={val}")

            if self.use_wandb:
                log = {f"{self.setup}/train_{key}": value for key, value in avg.items()}
                log[f"{self.setup}/train_loss"] = avg["loss"]
                if not np.isnan(val_loss):
                    log.update({f"{self.setup}/val_{key}": value for key, value in val.items()})
                wandb.log(log, step=epoch)

            if not np.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(f"{self.setup}_best.pt", step=epoch)

            if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
                self._save_checkpoint(f"{self.setup}_epoch{epoch + 1}.pt", step=epoch + 1)

        self._save_checkpoint(f"{self.setup}_final.pt", step=epochs)

    def _sample_cbg(self) -> str:
        return self.cbgs[random.randrange(len(self.cbgs))]

    def _sample_demo_ids_from_pi(
        self,
        cbg: str,
        batch_size: int,
        *,
        min_per_group: int = 0,
        cache: Optional[CBGConditionCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = cache or self.cache
        assert cache is not None
        D_age = max(self.num_age_bins, 1)
        D_gen = max(self.num_genders, 1)
        D = D_age * D_gen
        pi = cache.as_torch_distribution(
            cbg, num_age_bins=D_age, num_genders=D_gen, device=self.device
        ).to(dtype=torch.float32)
        pi = (pi + 1e-8) / (pi.sum() + 1e-8 * pi.numel())

        if min_per_group > 0:
            required = min_per_group * D
            if batch_size < required:
                raise ValueError(f"aggregate_batch_size={batch_size} is too small for min_per_group={min_per_group} and {D} groups.")
            base = torch.arange(D, device=self.device).repeat_interleave(min_per_group)
            remaining = batch_size - int(base.numel())
            extra = torch.multinomial(pi, num_samples=remaining, replacement=True) if remaining > 0 else torch.empty(0, device=self.device, dtype=torch.long)
            group_idx = torch.cat([base, extra.to(dtype=torch.long)], dim=0)
            group_idx = group_idx[torch.randperm(group_idx.numel(), device=self.device)]
        else:
            group_idx = torch.multinomial(pi, num_samples=batch_size, replacement=True).to(dtype=torch.long)

        age = torch.div(group_idx, D_gen, rounding_mode="floor")
        gender = group_idx % D_gen
        return age, gender

    def _sample_atlas_conditions(
        self,
        cbg: str,
        *,
        cache: Optional[CBGConditionCache] = None,
        batch_size: Optional[int] = None,
        demo_source: Optional[str] = None,
        min_per_group: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = cache or self.cache
        assert cache is not None
        atlas_cfg = self.cfg.get("atlas", {}) or {}
        train_cfg = self.cfg["training"]
        if batch_size is None:
            batch_size = int(train_cfg.get("aggregate_batch_size", train_cfg.get("batch_size", 256)))
        demo_source = str(demo_source or atlas_cfg.get("demo_source", "pi")).lower().strip()
        if demo_source not in {"pi", "cache"}:
            raise ValueError("atlas.demo_source must be 'pi' or 'cache'")

        batch = cache.sample(cbg, batch_size, device=self.device)
        if demo_source == "pi":
            if min_per_group is None:
                min_per_group = int((atlas_cfg.get("llp", {}) or {}).get("min_per_group", 0) or 0)
            age, gender = self._sample_demo_ids_from_pi(
                cbg, batch_size, min_per_group=int(min_per_group or 0), cache=cache
            )
        else:
            age, gender = batch.age_bin, batch.gender_id
        return age, gender, batch.home, batch.work

    def _prior_poi_probs_with_grad(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
        home: torch.Tensor,
        work: torch.Tensor,
        *,
        target_vocab: Optional[int] = None,
    ) -> torch.Tensor:
        B = int(home.size(0))
        T = int(self.model.max_seq_len)
        device = home.device
        z = torch.randn(B, T, self.model.latent_size, device=device)
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30.0
        demo_emb = self.model.demo_block(age, gender, home, work, T)
        dec_input = self.model._build_decoder_input(z, demo_emb, pos)
        _, _, logits = self.model.decoder(dec_input)
        if self.num_special_tokens > 0:
            logits = logits.clone()
            logits[..., : self.num_special_tokens] = logits[..., : self.num_special_tokens] - 1e4
        probs = F.softmax(logits, dim=-1)
        if self.num_special_tokens > 0:
            probs = probs[..., self.num_special_tokens :]

        if target_vocab is None:
            assert self.poi_store is not None
            target_vocab = int(self.poi_store.vocab_size())
        if probs.size(-1) > target_vocab:
            probs = probs[..., :target_vocab]
        elif probs.size(-1) < target_vocab:
            raise ValueError(
                f"Model POI vocab ({probs.size(-1)}) is smaller than target POI vocab ({target_vocab})."
            )
        return probs

    def _llp_aggregate_loss(
        self,
        poi_probs: torch.Tensor,
        cbg: str,
        age: torch.Tensor,
        gender: torch.Tensor,
        *,
        cache: Optional[CBGConditionCache] = None,
        poi_store: Optional[POIMarginalStore] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cache = cache or self.cache
        poi_store = poi_store or self.poi_store
        assert cache is not None
        assert poi_store is not None
        atlas_cfg = self.cfg.get("atlas", {}) or {}
        loss_cfg = atlas_cfg.get("aggregate_loss", {}) or {}
        loss_type = str(loss_cfg.get("type", "kl")).lower().strip()
        epsilon = float(loss_cfg.get("epsilon", 1e-8))

        B, T, V = poi_probs.shape
        D_age = max(self.num_age_bins, 1)
        D_gen = max(self.num_genders, 1)
        D = D_age * D_gen
        group_idx = (age.long().clamp_min(0) * D_gen + gender.long().clamp_min(0)).clamp(0, D - 1)
        one_hot = F.one_hot(group_idx, num_classes=D).to(dtype=poi_probs.dtype, device=poi_probs.device)
        weights = one_hot.unsqueeze(1).expand(B, T, D)

        raw_counts = weights.sum(dim=(0, 1))
        group_mass = (poi_probs.unsqueeze(2) * weights.unsqueeze(-1)).sum(dim=(0, 1))
        group_counts = raw_counts.clamp_min(1.0)
        per_group = group_mass / group_counts.unsqueeze(-1)

        empty = raw_counts <= 0
        if empty.any():
            overall = aggregate_poi_distribution(poi_probs, attention_mask=None, epsilon=epsilon)
            per_group[empty] = overall.unsqueeze(0).expand(int(empty.sum().item()), V)

        pi = cache.as_torch_distribution(
            cbg, num_age_bins=D_age, num_genders=D_gen, device=poi_probs.device
        ).to(dtype=poi_probs.dtype)
        pi = (pi + epsilon) / (pi.sum() + epsilon * pi.numel())
        pred = (pi.unsqueeze(-1) * per_group).sum(dim=0)
        pred = (pred + epsilon) / (pred.sum() + epsilon * pred.numel())

        target = poi_store.get_distribution(cbg, device=poi_probs.device).to(dtype=poi_probs.dtype)
        if target.numel() != pred.numel():
            raise ValueError(f"Target POI dim ({target.numel()}) != prediction dim ({pred.numel()}) for CBG {cbg}.")
        loss = distribution_loss(target, pred, loss_type=loss_type, epsilon=epsilon)
        return loss, {f"agg_{loss_type}": float(loss.item())}

    def _flat_aggregate_loss(
        self,
        poi_probs: torch.Tensor,
        cbg: str,
        *,
        poi_store: Optional[POIMarginalStore] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        poi_store = poi_store or self.poi_store
        assert poi_store is not None
        loss_cfg = (self.cfg.get("atlas", {}) or {}).get("aggregate_loss", {}) or {}
        loss_type = str(loss_cfg.get("type", "kl")).lower().strip()
        epsilon = float(loss_cfg.get("epsilon", 1e-8))
        target = poi_store.get_distribution(cbg, device=poi_probs.device).to(dtype=poi_probs.dtype)
        if loss_type == "kl":
            return poi_marginal_kl_loss(poi_probs, target, attention_mask=None, epsilon=epsilon)
        pred = aggregate_poi_distribution(poi_probs, attention_mask=None, epsilon=epsilon)
        loss = distribution_loss(target, pred, loss_type=loss_type, epsilon=epsilon)
        return loss, {f"agg_{loss_type}": float(loss.item())}

    def _validate_atlas_aggregate(self, *, llp_enabled: bool) -> Dict[str, float]:
        """Validate aggregate ATLAS loss on a held-out CBG cache/marginal store."""
        if not self.val_agg_enabled:
            return {}
        assert self.val_cache is not None
        assert self.val_poi_store is not None
        if not self.val_cbgs:
            return {}

        self.model.eval()
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        by_cbg_sum: Dict[str, float] = {}
        by_cbg_count: Dict[str, int] = {}

        with torch.no_grad():
            for _ in range(self.val_agg_num_batches):
                cbg = self.val_cbgs[random.randrange(len(self.val_cbgs))]
                age, gender, home, work = self._sample_atlas_conditions(
                    cbg,
                    cache=self.val_cache,
                    batch_size=self.val_agg_batch_size,
                    demo_source=self.val_agg_demo_source,
                    min_per_group=0,
                )
                poi_probs = self._prior_poi_probs_with_grad(
                    age,
                    gender,
                    home,
                    work,
                    target_vocab=int(self.val_poi_store.vocab_size()),
                )
                if llp_enabled:
                    loss, stats = self._llp_aggregate_loss(
                        poi_probs,
                        cbg,
                        age,
                        gender,
                        cache=self.val_cache,
                        poi_store=self.val_poi_store,
                    )
                else:
                    loss, stats = self._flat_aggregate_loss(
                        poi_probs,
                        cbg,
                        poi_store=self.val_poi_store,
                    )
                stats = dict(stats)
                stats["agg_loss"] = float(loss.item())
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
                if self.val_agg_log_by_cbg:
                    by_cbg_sum[cbg] = by_cbg_sum.get(cbg, 0.0) + float(loss.item())
                    by_cbg_count[cbg] = by_cbg_count.get(cbg, 0) + 1

        self.model.train()
        out = {key: totals[key] / max(counts[key], 1) for key in totals}

        if self.val_agg_log_by_cbg:
            max_to_log = self.val_agg_max_cbgs_to_log
            if max_to_log <= 0:
                max_to_log = len(by_cbg_sum) if len(by_cbg_sum) <= 8 else 0
            for cbg in sorted(by_cbg_sum)[:max_to_log]:
                out[f"agg_loss_by_cbg/{cbg}"] = by_cbg_sum[cbg] / max(by_cbg_count.get(cbg, 0), 1)
        return out

    def train_atlas(self) -> None:
        train_cfg = self.cfg["training"]
        atlas_cfg = self.cfg.get("atlas", {}) or {}
        steps = int(train_cfg.get("steps", train_cfg.get("phase2_steps", 5000)))
        lr = float(train_cfg.get("lr", train_cfg.get("phase2_lr", 1e-5)))
        lambda_agg = float(train_cfg.get("lambda_agg", 1.0))
        lambda_vae = float(train_cfg.get("lambda_vae", 0.0))
        raw_demo_src = train_cfg.get("vae_regularizer_demo_source", "null")
        vae_regularizer_demo_source = "null" if raw_demo_src is None else str(raw_demo_src).lower().strip()
        if vae_regularizer_demo_source not in {"null", "data"}:
            raise ValueError("training.vae_regularizer_demo_source must be 'null' or 'data'")
        lambda_entropy = float(train_cfg.get("lambda_entropy", 0.0))
        val_every = int(train_cfg.get("val_every", 500))
        llp_enabled = bool((atlas_cfg.get("llp", {}) or {}).get("enabled", True))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
        train_iter = iter(self.train_loader)
        log_window: Dict[str, List[float]] = {"total_loss": [], "agg_loss": []}

        print("=" * 60)
        print("ATLAS: aggregate demographic supervision")
        print("=" * 60)

        for step in range(steps):
            cbg = self._sample_cbg()
            age, gender, home, work = self._sample_atlas_conditions(cbg)
            poi_probs = self._prior_poi_probs_with_grad(age, gender, home, work)

            if llp_enabled:
                agg_loss, stats = self._llp_aggregate_loss(poi_probs, cbg, age, gender)
            else:
                agg_loss, stats = self._flat_aggregate_loss(poi_probs, cbg)

            total_loss = lambda_agg * agg_loss
            stats["total_loss"] = float(total_loss.item())
            stats["agg_loss"] = float(agg_loss.item())
            if lambda_vae > 0:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if vae_regularizer_demo_source == "null":
                    batch["age_bin"] = torch.full_like(batch["age_bin"], -1)
                    batch["gender_id"] = torch.full_like(batch["gender_id"], -1)
                output = self.model(batch)
                vae_losses = VolunteerVAE.vae_loss(output, batch)
                total_loss = total_loss + lambda_vae * vae_losses["loss"]
                stats.update({
                    "total_loss": float(total_loss.item()),
                    "vae_loss": float(vae_losses["loss"].item()),
                    "vae_kl": float(vae_losses["kl"].item()),
                    "vae_nll_loc": float(vae_losses["nll_loc"].item()),
                    "vae_nll_tim": float(vae_losses["nll_tim"].item()),
                })

            if lambda_entropy > 0:
                entropy = -(poi_probs * (poi_probs + 1e-8).log()).sum(dim=-1).mean()
                total_loss = total_loss - lambda_entropy * entropy
                stats["total_loss"] = float(total_loss.item())
                stats["entropy"] = float(entropy.item())

            optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()

            for key, value in stats.items():
                log_window.setdefault(key, []).append(float(value))

            if step % self.log_every == 0:
                window_avg = {key: float(np.mean(values)) for key, values in log_window.items() if values}
                msg = (
                    f"[atlas] step={step} train_total={window_avg.get('total_loss', total_loss.item()):.4f} "
                    f"train_agg={window_avg.get('agg_loss', agg_loss.item()):.4f} cbg={cbg}"
                )
                if "vae_loss" in window_avg:
                    msg += f" train_vae={window_avg['vae_loss']:.4f}"
                print(msg)
                if self.use_wandb:
                    wandb.log({f"atlas/train_{k}": v for k, v in window_avg.items()}, step=step)
                log_window = {"total_loss": [], "agg_loss": []}

            if val_every > 0 and step > 0 and step % val_every == 0:
                val_vae = self._validate_vae(demo_source=vae_regularizer_demo_source)
                val_agg = self._validate_atlas_aggregate(llp_enabled=llp_enabled)
                print(f"[atlas] step={step} val_vae={val_vae} val_agg={val_agg}")
                if self.use_wandb:
                    log: Dict[str, float] = {}
                    if not np.isnan(val_vae["loss"]):
                        log.update({f"atlas/val_vae_{key}": value for key, value in val_vae.items()})
                    log.update({f"atlas/val_agg_{key}": value for key, value in val_agg.items()})
                    if log:
                        wandb.log(log, step=step)

            if step > 0 and step % self.save_every == 0:
                self._save_checkpoint(f"atlas_step{step}.pt", step=step)

        val_vae = self._validate_vae(demo_source=vae_regularizer_demo_source)
        val_agg = self._validate_atlas_aggregate(llp_enabled=llp_enabled)
        print(f"[atlas] final_val_vae={val_vae} final_val_agg={val_agg}")
        if self.use_wandb:
            log: Dict[str, float] = {}
            if not np.isnan(val_vae["loss"]):
                log.update({f"atlas/final_val_vae_{key}": value for key, value in val_vae.items()})
            log.update({f"atlas/final_val_agg_{key}": value for key, value in val_agg.items()})
            if log:
                wandb.log(log, step=steps)
        self._save_checkpoint("atlas_final.pt", step=steps)

    def run(self) -> None:
        with open(self.output_dir / "config.yaml", "w") as f:
            cfg_to_save = dict(self.cfg)
            cfg_to_save.setdefault("model", {})
            cfg_to_save["model"]["use_demo_condition"] = self.model.use_demo_condition
            cfg_to_save["model"]["demo_conditioning_type"] = self.model.demo_conditioning_type
            yaml.safe_dump(cfg_to_save, f)

        if self.setup in {"baseline", "strong"}:
            self.train_supervised()
        else:
            self.train_atlas()

        if self.use_wandb:
            wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VOLUNTEER baseline, strong, or ATLAS setup.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--setup", type=str, default=None, choices=["baseline", "strong", "atlas"])
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
    trainer = VolunteerSetupTrainer(cfg, setup_override=args.setup)
    trainer.run()


if __name__ == "__main__":
    main()
