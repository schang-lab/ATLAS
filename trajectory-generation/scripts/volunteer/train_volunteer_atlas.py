#!/usr/bin/env python3
"""
Train VOLUNTEER VAE with ATLAS aggregate demographic supervision.

Two-phase training:
  Phase 1 (VAE pretrain): Standard ELBO on trajectory reconstruction.
  Phase 2 (ATLAS fine-tune): Add aggregate POI KL loss using CBG-level demographics.

Example:
    python trajectory-generation/scripts/volunteer/train_volunteer_atlas.py \
        --config trajectory-generation/scripts/volunteer/config_volunteer.yaml
"""

from __future__ import annotations

import argparse
import json
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

# Allow running from any directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
# Also add the volunteer scripts directory for local imports
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from volunteer_model import VolunteerVAE
from volunteer_dataset import build_dataloaders
from src.data import CBGConditionCache, POIMarginalStore
from src.losses import aggregate_poi_distribution, poi_marginal_kl_loss

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VolunteerATLASTrainer:
    """Two-phase trainer: VAE pretrain → ATLAS aggregate fine-tune."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Model
        model_cfg = cfg["model"]
        self.model = VolunteerVAE(model_cfg).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Data loaders (for Phase 1 VAE pretrain)
        data_cfg = cfg["data"]
        self.loaders = build_dataloaders(
            data_root=data_cfg["split_data_dir"],
            batch_size=cfg["training"]["batch_size"],
            max_seq_len=model_cfg.get("max_seq_len", 64),
            num_workers=data_cfg.get("num_workers", 4),
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders.get("val")

        # ATLAS aggregate data (for Phase 2)
        atlas_cfg = cfg.get("atlas", {})
        self.cbg_cache_dir = atlas_cfg.get("cbg_cache_dir")
        self.poi_marginal_csv = atlas_cfg.get("poi_marginal_csv")
        self.cache: Optional[CBGConditionCache] = None
        self.poi_store: Optional[POIMarginalStore] = None
        self.cbg_list: List[str] = []

        if self.cbg_cache_dir and self.poi_marginal_csv:
            self.cache = CBGConditionCache(self.cbg_cache_dir)
            self.poi_store = POIMarginalStore(self.poi_marginal_csv)
            # Use CBGs available in both cache and marginals
            cache_cbgs = set(self.cache.available_cbgs())
            store_cbgs = set(self.poi_store.available_cbgs())
            self.cbg_list = sorted(cache_cbgs & store_cbgs)
            print(f"ATLAS: {len(self.cbg_list)} CBGs available for aggregate training")

        # Training config
        train_cfg = cfg["training"]
        self.num_special_tokens = model_cfg.get("num_special_tokens", 5)
        self.num_age_bins = model_cfg.get("num_age_bins", 4)
        self.num_genders = model_cfg.get("num_genders", 2)

        # Phase 1
        self.phase1_epochs = train_cfg.get("phase1_epochs", 30)
        self.phase1_lr = float(train_cfg.get("phase1_lr", 1e-4))

        # Phase 2
        self.phase2_steps = train_cfg.get("phase2_steps", 5000)
        self.phase2_lr = float(train_cfg.get("phase2_lr", 1e-5))
        self.lambda_vae = float(train_cfg.get("lambda_vae", 1.0))
        self.lambda_agg = float(train_cfg.get("lambda_agg", 1.0))
        self.lambda_entropy = float(train_cfg.get("lambda_entropy", 0.0))
        self.aggregate_batch_size = train_cfg.get("aggregate_batch_size", 256)

        # LLP config
        llp_cfg = atlas_cfg.get("llp", {})
        self.llp_enabled = llp_cfg.get("enabled", True)

        # Logging / checkpointing
        self.output_dir = Path(train_cfg.get("output_dir", "./runs/volunteer_atlas"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = train_cfg.get("log_every", 50)
        self.save_every = train_cfg.get("save_every", 500)
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

        # W&B
        self.use_wandb = train_cfg.get("wandb", False) and HAS_WANDB
        if self.use_wandb:
            wandb.init(project=train_cfg.get("wandb_project", "volunteer-atlas"), config=cfg)

    # ------------------------------------------------------------------
    # Phase 1: VAE pretrain
    # ------------------------------------------------------------------

    def train_phase1(self):
        """Standard VAE training on trajectory reconstruction."""
        print("=" * 60)
        print("PHASE 1: VAE Pretrain")
        print("=" * 60)

        optimizer = optim.Adam(self.model.parameters(), lr=self.phase1_lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.phase1_epochs)

        best_val_loss = float("inf")

        for epoch in range(self.phase1_epochs):
            self.model.train()
            epoch_stats = {"loss": [], "kl": [], "nll_loc": [], "nll_tim": []}

            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                output = self.model(batch)
                losses = VolunteerVAE.vae_loss(output, batch)
                losses["loss"].backward()

                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                for k in epoch_stats:
                    epoch_stats[k].append(losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k])

                if batch_idx % self.log_every == 0:
                    print(
                        f"  [P1] Epoch {epoch} Batch {batch_idx}: "
                        f"loss={losses['loss'].item():.4f} kl={losses['kl'].item():.4f} "
                        f"nll_loc={losses['nll_loc'].item():.4f} nll_tim={losses['nll_tim'].item():.4f}"
                    )

            scheduler.step()

            # Epoch summary
            avg = {k: np.mean(v) for k, v in epoch_stats.items()}
            print(f"  [P1] Epoch {epoch} avg: {avg}")

            if self.use_wandb:
                wandb.log({f"phase1/{k}": v for k, v in avg.items()}, step=epoch)

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                print(f"  [P1] Epoch {epoch} val_loss: {val_loss:.4f}")
                if self.use_wandb:
                    wandb.log({"phase1/val_loss": val_loss}, step=epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("phase1_best.pt")

        self._save_checkpoint("phase1_final.pt")
        print("Phase 1 complete.")

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch)
                losses = VolunteerVAE.vae_loss(output, batch)
                total_loss += losses["loss"].item() * batch["loc"].size(0)
                count += batch["loc"].size(0)
        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Phase 2: ATLAS aggregate fine-tuning
    # ------------------------------------------------------------------

    def train_phase2(self):
        """Fine-tune with ATLAS aggregate demographic supervision."""
        if not self.cbg_list:
            print("No CBG data configured — skipping Phase 2.")
            return

        print("=" * 60)
        print("PHASE 2: ATLAS Aggregate Fine-tune")
        print("=" * 60)

        # Use a smaller LR for fine-tuning
        optimizer = optim.Adam(self.model.parameters(), lr=self.phase2_lr, weight_decay=1e-5)

        # Also iterate over real trajectories for VAE loss
        train_iter = iter(self.train_loader)

        for step in range(self.phase2_steps):
            self.model.train()
            stats: Dict[str, float] = {}

            # --- VAE reconstruction loss on real data ---
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            output = self.model(batch)
            vae_losses = VolunteerVAE.vae_loss(output, batch)
            vae_loss = vae_losses["loss"]
            stats["vae_loss"] = vae_loss.item()
            stats["vae_kl"] = vae_losses["kl"].item()
            stats["vae_nll_loc"] = vae_losses["nll_loc"].item()

            # --- ATLAS aggregate loss ---
            cbg = random.choice(self.cbg_list)
            agg_loss, agg_stats = self._compute_aggregate_loss(cbg)
            stats.update(agg_stats)
            stats["cbg"] = cbg

            # --- Combined loss ---
            total_loss = self.lambda_vae * vae_loss + self.lambda_agg * agg_loss

            # Optional entropy regularizer
            if self.lambda_entropy > 0:
                ent_loss = self._entropy_regularizer(output["loc_probs"], output["mask"])
                total_loss = total_loss + self.lambda_entropy * ent_loss
                stats["entropy_loss"] = ent_loss.item()

            stats["total_loss"] = total_loss.item()
            stats["agg_loss"] = agg_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()

            # Logging
            if step % self.log_every == 0:
                print(
                    f"  [P2] Step {step}: total={stats['total_loss']:.4f} "
                    f"vae={stats['vae_loss']:.4f} agg={stats['agg_loss']:.4f} "
                    f"cbg={cbg}"
                )

            if self.use_wandb and step % self.log_every == 0:
                wandb.log({f"phase2/{k}": v for k, v in stats.items() if isinstance(v, (int, float))}, step=step)

            if step > 0 and step % self.save_every == 0:
                self._save_checkpoint(f"phase2_step{step}.pt")

        self._save_checkpoint("phase2_final.pt")
        print("Phase 2 complete.")

    def _compute_aggregate_loss(self, cbg: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sample from a CBG, run VOLUNTEER forward, compute aggregate POI KL."""
        batch = self.cache.sample(cbg, self.aggregate_batch_size, device=self.device)
        stats: Dict[str, float] = {}

        # Sample latent z and generate POI probabilities
        B = batch.home.size(0)
        age_raw = batch.age_bin
        gender_raw = batch.gender_id

        gen_out = self.model.generate(
            age_bin=age_raw,
            gender_id=gender_raw,
            home=batch.home,
            work=batch.work,
            max_len=self.model.max_seq_len,
        )

        # WAIT — generate() is @torch.no_grad()!
        # For training, we need gradients. Use forward pass with sampled z instead.
        # We'll do a "generation-like" forward pass with gradient.
        poi_probs = self._forward_with_grad(
            age_raw=age_raw,
            gender_raw=gender_raw,
            home=batch.home,
            work=batch.work,
        )

        # Strip special tokens from probs to match p_poi.csv vocabulary
        if self.num_special_tokens > 0:
            poi_probs_clean = poi_probs[..., self.num_special_tokens:]
        else:
            poi_probs_clean = poi_probs

        # Align vocab dimensions: p_poi.csv may cover fewer POIs than the full tokenizer.
        # The target is indexed by poi_index (0..V_target-1); model output covers all
        # non-special tokens (0..V_model-1). We slice model probs to match target size.
        target_vocab = self.poi_store.vocab_size()
        model_vocab = poi_probs_clean.size(-1)
        if model_vocab > target_vocab:
            poi_probs_clean = poi_probs_clean[..., :target_vocab]

        # Attention mask: all ones (generated sequences are full-length)
        attn_mask = torch.ones(poi_probs_clean.size(0), poi_probs_clean.size(1), device=self.device)

        if self.llp_enabled and self.num_age_bins > 0 and self.num_genders > 0:
            agg_loss, agg_stats = self._llp_mixture_kl(
                poi_probs_clean, attn_mask,
                cbg=cbg,
                batch_age=age_raw,
                batch_gender=gender_raw,
            )
        else:
            target = self.poi_store.get_distribution(cbg, device=self.device)
            agg_loss, agg_stats = poi_marginal_kl_loss(
                poi_probs_clean, target, attention_mask=attn_mask,
            )

        stats.update(agg_stats)
        return agg_loss, stats

    def _forward_with_grad(
        self,
        age_raw: torch.Tensor,
        gender_raw: torch.Tensor,
        home: torch.Tensor,
        work: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder with sampled latent z (with gradient) to get POI probabilities.

        Unlike generate(), this keeps gradients flowing for training.
        Returns loc_probs: (B, T, vocab).
        """
        B = home.size(0)
        max_len = self.model.max_seq_len
        device = home.device

        # Sample latent
        z = torch.randn(B, max_len, self.model.latent_size, device=device)

        # Dummy timestamps (fixed spacing)
        pos = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30.0

        # Demo conditioning
        demo_emb = self.model.demo_block(age_raw, gender_raw, home, work, max_len)

        # Build decoder input and run
        dec_input = self.model._build_decoder_input(z, demo_emb, pos)
        loc_probs, _, _ = self.model.decoder(dec_input)

        return loc_probs

    def _llp_mixture_kl(
        self,
        poi_probs: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        cbg: str,
        batch_age: torch.Tensor,
        batch_gender: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """LLP mixture KL: same logic as run_cbg_conditioned_training.py."""
        B, T, V = poi_probs.shape
        device = poi_probs.device
        D_age = max(self.num_age_bins, 1)
        D_gen = max(self.num_genders, 1)
        D = D_age * D_gen

        # Group indices: d = age * num_genders + gender
        group_idx = (batch_age.long().clamp_min(0) * D_gen + batch_gender.long().clamp_min(0)).clamp(0, D - 1)
        one_hot = F.one_hot(group_idx, num_classes=D).to(dtype=poi_probs.dtype, device=device)

        # Per-group POI distributions
        bt_weights = attention_mask.to(dtype=poi_probs.dtype, device=device).unsqueeze(-1) * one_hot.unsqueeze(1)
        group_mass = (poi_probs.unsqueeze(2) * bt_weights.unsqueeze(-1)).sum(dim=(0, 1))
        group_counts = bt_weights.sum(dim=(0, 1)).clamp_min(1.0)
        P_hat = group_mass / group_counts.unsqueeze(-1)

        # Fallback for empty groups
        overall = aggregate_poi_distribution(poi_probs, attention_mask=attention_mask, epsilon=epsilon)
        empty_groups = (group_counts <= 1.0)
        if empty_groups.any():
            P_hat[empty_groups] = overall.unsqueeze(0).expand(int(empty_groups.sum().item()), V)

        # π vector for this CBG
        pi_cbg = self.cache.as_torch_distribution(
            cbg, num_age_bins=D_age, num_genders=D_gen, device=device,
        ).to(dtype=poi_probs.dtype)
        pi_cbg = (pi_cbg + epsilon) / (pi_cbg.sum() + epsilon * pi_cbg.numel())

        # Mixture: P_mix = Σ_d π_d * P_hat_d
        mix = (pi_cbg.unsqueeze(-1) * P_hat).sum(dim=0)
        mix = (mix + epsilon) / (mix.sum() + epsilon * mix.numel())

        # Target
        target = self.poi_store.get_distribution(cbg, device=device).to(dtype=mix.dtype)
        target = (target + epsilon) / (target.sum() + epsilon * target.numel())

        # KL(target || mix)
        kl = torch.sum(target * (torch.log(target + epsilon) - torch.log(mix + epsilon)))
        return kl, {"agg_kl": float(kl.item())}

    def _entropy_regularizer(
        self, poi_probs: torch.Tensor, attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage per-trajectory POI diversity."""
        mask = attn_mask.unsqueeze(-1).to(dtype=poi_probs.dtype)
        masked_probs = poi_probs * mask
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        hist = masked_probs.sum(dim=1) / token_counts
        hist_sum = hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hist = hist / hist_sum
        entropy = -(hist * (hist + 1e-8).log()).sum(dim=-1)
        return -entropy.mean()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str):
        path = self.output_dir / name
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {path}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self):
        """Run both training phases."""
        # Phase 1
        if self.phase1_epochs > 0:
            self.train_phase1()

        # Phase 2
        if self.phase2_steps > 0:
            self.train_phase2()

        print("Training complete.")
        # Save final config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train VOLUNTEER-ATLAS.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--phase1-checkpoint", type=str, default=None,
                        help="Skip Phase 1, load this checkpoint and go straight to Phase 2.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device

    trainer = VolunteerATLASTrainer(cfg)

    if args.phase1_checkpoint:
        trainer.load_checkpoint(args.phase1_checkpoint)
        trainer.cfg["training"]["phase1_epochs"] = 0  # skip Phase 1
        print(f"Loaded Phase 1 checkpoint, skipping to Phase 2.")

    trainer.run()


if __name__ == "__main__":
    main()
