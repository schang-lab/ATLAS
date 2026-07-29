#!/usr/bin/env python3
"""
Fine-tune a pretrained VAE backbone with CBG-conditioned aggregate POI supervision (ATLAS loss).

This mirrors run_cbg_conditioned_training.py but replaces the DiT + diffusion
sampling path with VAE prior sampling. The aggregate POI loss — including LLP
demographic stratification with demo_source=pi — is identical.

Example:
    python run_cbg_conditioned_training_vae.py \
        --config /path/to/cbg_conditioned_training_vae.yaml

YAML config schema:
    vae:
        config: /path/to/config_vae_phase1.yml
        checkpoint: /path/to/vae_final.pt  # optional; null/missing means random init
        seq_len: 512
        latent_dim: 512       # in_channels (after PCA if used)
    autoencoder:
        path: /path/to/pretrained_autoencoder
    latent_pca:
        path: null             # optional PCA path
    data:
        cbg_cache_dir: /path/to/cbg_condition_cache
        poi_marginal_csv: /path/to/p_poi.csv
        num_special_tokens: 0     # optional; inferred when p_poi.csv is POI-only
    training:
        steps: 2000
        batch_size: 256
        lr: 1e-5
        lambda_agg: 1.0
        vae_loss:
            enabled: false
            weight: 1.0
            data_dir: /path/to/split_data
            data_type: controlled  # controlled | uncontrolled | unified
            demo_source: data       # data | null; null keeps recon/KL but hides true per-trajectory demo
            training_phase: phase1
            batch_size: 64
            kl_beta_max: 0.001
            kl_anneal_steps: 5000
        aggregate_loss_type: kl   # kl | js | tv | hellinger
        log_every: 25
        save_every: 500
        output_dir: /path/to/output
        wandb: false
        llp:
            enabled: false
            demo_source: pi       # pi | cache
            min_per_group: 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import yaml
from tqdm import tqdm

from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.vae import TrajectoryVAE
from src.data import CBGConditionCache, POIMarginalStore
from src.losses import poi_marginal_kl_loss, aggregate_poi_distribution
from src.latent_pca import LatentPCA


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_vae(
    vae_cfg: Dict[str, Any],
    device: torch.device,
    latent_pca: Optional[LatentPCA] = None,
) -> TrajectoryVAE:
    config_path = vae_cfg["config"]
    checkpoint_path = vae_cfg.get("checkpoint")
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    vae_kwargs = params.get("VAE", params)
    # Override in_channels when PCA reduces the latent dimension.
    if latent_pca is not None:
        vae_kwargs["in_channels"] = latent_pca.component_dim
        print(f"Overriding VAE in_channels to PCA dim: {latent_pca.component_dim}")
    vae = TrajectoryVAE(**vae_kwargs).to(device)
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        print("[INFO] No VAE checkpoint configured; training VAE from random initialization.")
        return vae
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    return vae


def build_autoencoder(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    ae_dir = cfg["path"]
    autoencoder = BartForConditionalGeneration.from_pretrained(ae_dir).to(device)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    return autoencoder


def select_cbgs(cache: CBGConditionCache, poi_store: POIMarginalStore, allowed: Optional[List[str]]) -> List[str]:
    cache_cbgs = set(str(c) for c in cache.available_cbgs())
    poi_cbgs = set(str(c) for c in poi_store.available_cbgs())
    available = cache_cbgs & poi_cbgs
    if allowed:
        available &= set(allowed)
    if not available:
        raise ValueError("No overlapping CBGs between cache and POI marginals.")
    return sorted(available)


def _autoencoder_vocab_size(autoencoder: nn.Module) -> Optional[int]:
    cfg_vocab = getattr(getattr(autoencoder, "config", None), "vocab_size", None)
    if isinstance(cfg_vocab, int) and cfg_vocab > 0:
        return int(cfg_vocab)
    try:
        emb = autoencoder.get_output_embeddings()
    except Exception:
        emb = None
    if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
        return int(emb.weight.shape[0])
    return None


def _resolve_num_special_tokens(
    configured: int,
    poi_store: POIMarginalStore,
    autoencoder: nn.Module,
    *,
    label: str,
) -> int:
    """Infer omitted special-token count when p_poi.csv is POI-only."""
    configured = max(0, int(configured))
    ae_vocab = _autoencoder_vocab_size(autoencoder)
    poi_vocab = int(poi_store.vocab_size())

    if configured == 0 and ae_vocab is not None and ae_vocab > poi_vocab:
        inferred = ae_vocab - poi_vocab
        print(
            f"[WARN] {label} num_special_tokens is 0/missing, but autoencoder vocab "
            f"({ae_vocab}) exceeds POI marginal vocab ({poi_vocab}); inferring "
            f"num_special_tokens={inferred}."
        )
        return inferred

    if (
        configured > 0
        and ae_vocab is not None
        and ae_vocab != poi_vocab
        and ae_vocab - configured != poi_vocab
    ):
        print(
            f"[WARN] {label} num_special_tokens={configured}, autoencoder vocab={ae_vocab}, "
            f"POI marginal vocab={poi_vocab}; expected either {ae_vocab} or "
            f"{max(ae_vocab - configured, 0)} POI marginal entries."
        )
    return configured


def _align_target_to_poi_dim(
    target: torch.Tensor,
    poi_dim: int,
    *,
    num_special_tokens: int,
    context: str,
) -> torch.Tensor:
    """Return target aligned to the POI-probability dimension."""
    if target.dim() != 1:
        raise ValueError(f"{context}: target_dist must be a 1D tensor, got shape {tuple(target.shape)}")

    target_dim = int(target.shape[0])
    poi_dim = int(poi_dim)
    num_special_tokens = max(0, int(num_special_tokens))

    if target_dim == poi_dim:
        return target
    if num_special_tokens > 0 and target_dim == poi_dim + num_special_tokens:
        return target[num_special_tokens:]

    raise ValueError(
        f"{context}: target distribution length ({target_dim}) does not match "
        f"prediction POI dimension ({poi_dim}); num_special_tokens={num_special_tokens}. "
        "Check data.num_special_tokens and make sure p_poi.csv uses the same "
        "POI-only/full-vocab convention as the decoder probabilities."
    )


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _shift_raw_demo_ids_for_vae(
    age_raw: torch.Tensor,
    gender_raw: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    missing = (age_raw < 0) | (gender_raw < 0)
    age_shifted = (age_raw.clamp_min(0) + 1).to(dtype=dtype, device=age_raw.device)
    gender_shifted = (gender_raw.clamp_min(0) + 1).to(dtype=dtype, device=gender_raw.device)
    if missing.any():
        age_shifted = age_shifted.clone()
        gender_shifted = gender_shifted.clone()
        age_shifted[missing] = 0
        gender_shifted[missing] = 0
    return age_shifted, gender_shifted


def _shift_demo_ids_for_vae(attrs: Optional[torch.Tensor], vae_model: nn.Module) -> Optional[torch.Tensor]:
    if attrs is None or attrs.dim() != 2 or attrs.size(1) < 6:
        return attrs
    if not getattr(vae_model, "use_demo_condition", False):
        return attrs

    age_idx = -2
    gender_idx = -1
    age_shifted, gender_shifted = _shift_raw_demo_ids_for_vae(
        attrs[:, age_idx].long(),
        attrs[:, gender_idx].long(),
        dtype=attrs.dtype,
    )

    out = attrs.clone()
    out[:, age_idx] = age_shifted
    out[:, gender_idx] = gender_shifted
    return out


def _prepare_vae_loss_attrs(
    batch: Dict[str, Any],
    args: SimpleNamespace,
    vae_model: nn.Module,
) -> Optional[torch.Tensor]:
    attrs = batch.get("attrs")
    if attrs is None:
        return None

    attrs = attrs.float()
    use_length = bool(getattr(args, "enable_length_condition", False))
    use_demo = bool(getattr(vae_model, "use_demo_condition", False))
    demo_source_raw = getattr(args, "demo_source", "data")
    demo_source = "null" if demo_source_raw is None else str(demo_source_raw).lower().strip()
    if demo_source in {"none", "no", "false", "off", "zero", "zeros"}:
        demo_source = "null"
    if demo_source not in {"data", "null"}:
        raise ValueError(
            "training.vae_loss.demo_source must be 'data' or 'null' "
            f"(got {demo_source_raw!r})."
        )

    parts = [attrs[:, :4]]

    length_id = batch.get("length_id")
    if length_id is not None and use_length:
        length_tensor = length_id.float().unsqueeze(-1).to(attrs.device)
        parts.append(length_tensor)

    if use_demo and attrs.dim() == 2 and attrs.size(1) >= 6:
        if demo_source == "null":
            age_f = torch.zeros((attrs.size(0), 1), dtype=attrs.dtype, device=attrs.device)
            gender_f = torch.zeros((attrs.size(0), 1), dtype=attrs.dtype, device=attrs.device)
        else:
            age_f, gender_f = _shift_raw_demo_ids_for_vae(
                attrs[:, -2].long(),
                attrs[:, -1].long(),
                dtype=attrs.dtype,
            )
            age_f = age_f.unsqueeze(-1)
            gender_f = gender_f.unsqueeze(-1)
        parts.extend([age_f, gender_f])

    return torch.cat(parts, dim=1)


def _extract_target_latents_for_vae_loss(
    batch: Dict[str, Any],
    autoencoder: nn.Module,
    args: SimpleNamespace,
    latent_pca: Optional[LatentPCA],
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.no_grad():
        encoder_outputs = autoencoder.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if getattr(args, "training_phase", "phase1") == "phase1":
            target_latents = encoder_outputs.last_hidden_state
        else:
            segment_coords = None
            sub_categories = None
            ablation_mode = getattr(args, "ablation_mode", "both")
            if ablation_mode in {"coords_only", "both"} and "lat" in batch and "lon" in batch:
                segment_coords = torch.stack([batch["lat"], batch["lon"]], dim=-1)
            if ablation_mode in {"subcat_only", "both"}:
                sub_categories = batch.get("sub_categories")

            if hasattr(autoencoder, "no_compression") and autoencoder.no_compression:
                enhanced_outputs = autoencoder._add_features_no_compression(
                    encoder_outputs,
                    attention_mask,
                    segment_coords,
                    sub_categories,
                )
                target_latents = enhanced_outputs["last_hidden_state"]
            else:
                target_latents = autoencoder.get_diffusion_latent(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    segment_coords=segment_coords,
                    sub_categories=sub_categories,
                )

    if latent_pca is not None:
        target_latents = latent_pca.project(target_latents)

    latent_scale = float(getattr(args, "latent_scale", 1.0) or 1.0)
    if latent_scale != 1.0:
        target_latents = target_latents / latent_scale

    return target_latents


def _distribution_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    loss_type: str = "kl",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute a scalar divergence between two probability vectors."""
    t = (target + epsilon) / (target.sum() + epsilon * target.numel())
    p = (pred + epsilon) / (pred.sum() + epsilon * pred.numel())
    if loss_type == "kl":
        return (t * (t.log() - p.log())).sum()
    elif loss_type == "js":
        m = 0.5 * (t + p)
        return 0.5 * (t * (t.log() - m.log())).sum() + 0.5 * (p * (p.log() - m.log())).sum()
    elif loss_type == "tv":
        return 0.5 * (t - p).abs().sum()
    elif loss_type == "hellinger":
        return (1.0 / (2.0 ** 0.5)) * ((t.sqrt() - p.sqrt()).pow(2).sum()).sqrt()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using random seed: {seed}")


class AggregateTrainerVAE:
    """CBG-conditioned aggregate fine-tuning for the VAE backbone."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        train_cfg = config["training"]
        seed = train_cfg.get("seed", config.get("seed"))
        self.seed = None if seed is None else int(seed)
        set_random_seed(self.seed)

        vae_cfg = config["vae"]
        self.seq_len = int(vae_cfg["seq_len"])
        self.latent_dim = int(vae_cfg["latent_dim"])

        # PCA must be loaded first so build_vae can override in_channels.
        lp_cfg = config.get("latent_pca", {}) or {}
        self.latent_pca = None
        pca_path = lp_cfg.get("path")
        if pca_path:
            self.latent_pca = LatentPCA(pca_path, device=device)

        self.vae = build_vae(vae_cfg, device, latent_pca=self.latent_pca)
        self.autoencoder = build_autoencoder(config["autoencoder"], device)

        # Cache demographic config from the VAE model for LLP grouping.
        self.num_age_bins = int(getattr(self.vae, "num_age_bins", 0))
        self.num_genders = int(getattr(self.vae, "num_genders", 0))

        # Data
        data_cfg = config["data"]
        self.cache = CBGConditionCache(data_cfg["cbg_cache_dir"])
        self.poi_store = POIMarginalStore(data_cfg["poi_marginal_csv"])
        self.cbgs = select_cbgs(self.cache, self.poi_store, data_cfg.get("allowed_cbgs"))
        self.num_special_tokens = _resolve_num_special_tokens(
            int(data_cfg.get("num_special_tokens", 0) or 0),
            self.poi_store,
            self.autoencoder,
            label="training",
        )

        # Training
        lr = float(train_cfg.get("lr", 1e-5))
        self.lambda_agg = float(train_cfg.get("lambda_agg", 1.0))
        self.steps = int(train_cfg.get("steps", 1000))
        self.batch_size = int(train_cfg.get("batch_size", 128))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.save_every = int(train_cfg.get("save_every", 500))
        self.output_dir = Path(train_cfg.get("output_dir", "./cbg_vae_finetune"))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.use_wandb = bool(train_cfg.get("wandb", False))
        self.wandb_name = train_cfg.get("wandb_name") or None
        self.wandb_project = str(train_cfg.get("wandb_project", "atlas-vae-cbg-finetune"))
        self.log_agg_by_cbg = bool(train_cfg.get("log_agg_by_cbg", False))

        # Aggregate loss type: kl | js | tv | hellinger
        agg_loss_cfg = train_cfg.get("aggregate_loss", {}) or {}
        self.aggregate_loss_type = str(
            agg_loss_cfg.get("type", train_cfg.get("aggregate_loss_type", "kl"))
        ).lower()
        if self.aggregate_loss_type not in {"kl", "js", "tv", "hellinger"}:
            raise ValueError(
                f"aggregate_loss type must be one of kl|js|tv|hellinger (got {self.aggregate_loss_type!r})"
            )
        self.aggregate_loss_eps = float(agg_loss_cfg.get("epsilon", 1e-8))

        # LLP (Learning from Label Proportions) configuration — mirrors DiT version
        llp_cfg = train_cfg.get("llp", {}) or {}
        self.llp_enabled = bool(llp_cfg.get("enabled", train_cfg.get("llp_enabled", False)))
        self.llp_demo_source = str(llp_cfg.get("demo_source", "pi")).lower().strip()
        if self.llp_demo_source not in {"cache", "pi"}:
            raise ValueError(f"training.llp.demo_source must be 'cache' or 'pi' (got {self.llp_demo_source!r})")
        self.llp_min_per_group = int(llp_cfg.get("min_per_group", 0))
        if self.llp_min_per_group < 0:
            self.llp_min_per_group = 0

        if self.llp_enabled and (self.num_age_bins <= 0 or self.num_genders <= 0):
            raise ValueError(
                "LLP is enabled but the VAE model has no demographic conditioning configured. "
                "Update your VAE config to set use_demo_condition: true "
                "and set num_age_bins / num_genders to match your cache."
            )

        # Optional original VAE loss on real trajectory batches:
        # recon MSE in BART/PCA latent space + beta * KL(q(z|x) || N(0, I)).
        vae_loss_cfg = train_cfg.get("vae_loss", config.get("vae_loss", {})) or {}
        self.use_vae_loss = bool(vae_loss_cfg.get("enabled", False))
        self.lambda_vae = float(vae_loss_cfg.get("weight", 1.0))
        self.vae_kl_beta_max = float(vae_loss_cfg.get("kl_beta_max", getattr(self.vae, "beta_kl", 0.001)))
        self.vae_kl_anneal_steps = int(vae_loss_cfg.get("kl_anneal_steps", 0) or 0)
        self.vae_loss_args: Optional[SimpleNamespace] = None
        self.vae_loss_loader = None
        self.vae_loss_iter = None
        self.vae_loss_val_loader = None
        self.vae_loss_val_batches = 0
        if self.use_vae_loss:
            if self.lambda_vae < 0:
                raise ValueError("training.vae_loss.weight must be non-negative.")
            vae_data_dir = vae_loss_cfg.get("data_dir") or data_cfg.get("trajectory_data_dir")
            if not vae_data_dir:
                raise ValueError(
                    "training.vae_loss.enabled is true, but training.vae_loss.data_dir "
                    "is not set. Point it at the split-data root used for VAE pretraining."
                )
            vae_loss_bs = int(vae_loss_cfg.get("batch_size", min(self.batch_size, 64)) or 64)
            self.vae_loss_args = SimpleNamespace(
                BATCH_SIZE=vae_loss_bs,
                NUM_WORKERS=int(vae_loss_cfg.get("num_workers", 0) or 0),
                data_type=str(vae_loss_cfg.get("data_type", "controlled")),
                demo_source=vae_loss_cfg.get("demo_source", "data"),
                training_phase=str(vae_loss_cfg.get("training_phase", "phase1")),
                ablation_mode=str(vae_loss_cfg.get("ablation_mode", "both")),
                sequence_length=int(vae_loss_cfg.get("sequence_length", self.seq_len) or self.seq_len),
                rebuild_attention_masks=bool(vae_loss_cfg.get("rebuild_attention_masks", True)),
                force_full_attention_mask=bool(vae_loss_cfg.get("force_full_attention_mask", False)),
                enable_length_condition=bool(
                    vae_loss_cfg.get(
                        "enable_length_condition",
                        getattr(self.vae, "use_length_condition", False),
                    )
                ),
                length_vocab_size=int(
                    vae_loss_cfg.get("length_vocab_size", getattr(self.vae, "length_vocab_size", 513))
                ),
                conditional_dropout=float(vae_loss_cfg.get("conditional_dropout", 0.0) or 0.0),
                latent_scale=float(vae_loss_cfg.get("latent_scale", 1.0) or 1.0),
            )
            self.vae_loss_val_batches = max(
                1,
                int(vae_loss_cfg.get("val_num_batches", vae_loss_cfg.get("num_val_batches", 10)) or 10),
            )
            from src.training import trajectory_dataset as load_trajectory_dataset

            train_loader, val_loader, _, _ = load_trajectory_dataset(
                self.vae_loss_args,
                data_dir=str(vae_data_dir),
                data_type=self.vae_loss_args.data_type,
            )
            self.vae_loss_loader = train_loader
            self.vae_loss_iter = iter(train_loader)
            self.vae_loss_val_loader = val_loader
            print(
                "Original VAE loss enabled: "
                f"weight={self.lambda_vae}, beta_max={self.vae_kl_beta_max}, "
                f"batch_size={vae_loss_bs}, val_batches={self.vae_loss_val_batches}, "
                f"data_dir={vae_data_dir}"
            )

        # Validation configuration (no early stopping — just logs val aggregate loss).
        val_cfg = train_cfg.get("validation", {}) or {}
        self.val_enabled = bool(val_cfg.get("enabled", False))
        self.val_every = int(val_cfg.get("eval_every", 0) or 0)
        self.val_num_batches = max(1, int(val_cfg.get("num_batches", 10) or 10))
        self.val_batch_size = int(val_cfg.get("batch_size", self.batch_size) or self.batch_size)
        self.val_log_by_cbg = bool(val_cfg.get("log_by_cbg", False))
        self.val_max_cbgs_to_log = max(0, int(val_cfg.get("max_cbgs_to_log", 0) or 0))

        self.val_cache: Optional[CBGConditionCache] = None
        self.val_poi_store: Optional[POIMarginalStore] = None
        self.val_cbgs: Optional[List[str]] = None
        self.val_num_special_tokens: int = self.num_special_tokens
        self.val_llp_demo_source: str = self.llp_demo_source
        if self.val_enabled:
            val_cache_dir = val_cfg.get("cbg_cache_dir")
            val_poi_csv = val_cfg.get("poi_marginal_csv")
            if not val_cache_dir or not val_poi_csv:
                raise ValueError(
                    "training.validation.enabled is true but training.validation.cbg_cache_dir "
                    "or training.validation.poi_marginal_csv is not set."
                )
            self.val_cache = CBGConditionCache(str(val_cache_dir))
            self.val_poi_store = POIMarginalStore(str(val_poi_csv))
            self.val_cbgs = select_cbgs(
                self.val_cache,
                self.val_poi_store,
                val_cfg.get("allowed_cbgs"),
            )
            self.val_num_special_tokens = _resolve_num_special_tokens(
                int(val_cfg.get("num_special_tokens", self.num_special_tokens) or 0),
                self.val_poi_store,
                self.autoencoder,
                label="validation",
            )
            v_demo = str(val_cfg.get("llp_demo_source", "") or "").lower().strip()
            if v_demo and v_demo != "inherit":
                if v_demo not in {"cache", "pi"}:
                    raise ValueError(
                        f"training.validation.llp_demo_source must be one of inherit|cache|pi "
                        f"(got {v_demo!r})."
                    )
                self.val_llp_demo_source = v_demo

        # Optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=lr)

        # Resume
        resume_path = train_cfg.get("resume_from")
        self.start_step = 0
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            self.vae.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_step = ckpt.get("step", 0)
            print(f"Resumed from {resume_path} at step {self.start_step}")

    # ------------------------------------------------------------------ #
    #  Sampling helpers
    # ------------------------------------------------------------------ #
    def _sample_cbg(self) -> str:
        return self.cbgs[torch.randint(len(self.cbgs), (1,)).item()]

    def _sample_demo_ids_from_pi(
        self,
        cbg: str,
        batch_size: int,
        *,
        cache: Optional[CBGConditionCache] = None,
        epsilon: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Sample raw (0-based) demo ids (age_bin, gender_id) from pi_cbg.

        If llp.min_per_group > 0, enforces >= K samples per group.
        """
        D_age = max(int(self.num_age_bins), 1)
        D_gen = max(int(self.num_genders), 1)
        D = D_age * D_gen

        src_cache = cache or self.cache
        pi = src_cache.as_torch_distribution(
            cbg,
            num_age_bins=D_age,
            num_genders=D_gen,
            device=self.device,
        ).to(dtype=torch.float32)
        pi = (pi + epsilon) / (pi.sum() + epsilon * pi.numel())

        K = self.llp_min_per_group
        if K > 0:
            required = K * D
            if batch_size < required:
                raise ValueError(
                    f"LLP requires batch_size >= llp.min_per_group * num_groups "
                    f"({batch_size} < {required} for K={K}, groups={D})."
                )
            base = torch.arange(D, device=self.device, dtype=torch.long).repeat_interleave(K)
            remaining = batch_size - base.numel()
            if remaining > 0:
                extra = torch.multinomial(pi, num_samples=remaining, replacement=True).to(dtype=torch.long)
                group_idx = torch.cat([base, extra], dim=0)
            else:
                group_idx = base
            perm = torch.randperm(group_idx.numel(), device=self.device)
            group_idx = group_idx[perm]
        else:
            group_idx = torch.multinomial(pi, num_samples=batch_size, replacement=True).to(dtype=torch.long)

        age_raw = torch.div(group_idx, D_gen, rounding_mode="floor")
        gender_raw = group_idx % D_gen

        stats = {
            "llp_demo_source_pi": 1.0,
            "llp_num_groups": float(D),
            "llp_min_per_group": float(K),
        }
        return age_raw, gender_raw, stats

    # ------------------------------------------------------------------ #
    #  Attribute construction
    # ------------------------------------------------------------------ #
    def _build_batch(
        self,
        cbg: str,
        age_raw: Optional[torch.Tensor] = None,
        gender_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build attribute tensor for a batch from the *training* CBG cache."""
        return self._build_batch_from(
            cbg, self.batch_size, self.cache,
            age_raw=age_raw, gender_raw=gender_raw,
        )

    def _build_batch_from(
        self,
        cbg: str,
        batch_size: int,
        cache: CBGConditionCache,
        age_raw: Optional[torch.Tensor] = None,
        gender_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build attribute tensor for a batch from an arbitrary CBG cache.

        Returns (B, F) where F = 4 (coords only) or 6 (coords + age + gender)
        depending on whether the VAE has demographic conditioning enabled.
        """
        batch = cache.sample(cbg, batch_size, device=self.device)
        attrs_parts = [batch.work, batch.home]  # each (B, 2)

        if getattr(self.vae, "use_demo_condition", False):
            # Use provided demo ids (from pi sampling) or fall back to cache
            if age_raw is not None and gender_raw is not None:
                a = age_raw
                g = gender_raw
            else:
                a = batch.age_bin
                g = batch.gender_id
            # Shift by +1 so 0 is reserved as null/padding id (same convention as DiT).
            age_f = (a.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            gender_f = (g.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            attrs_parts.extend([age_f, gender_f])

        attrs = torch.cat(attrs_parts, dim=-1)  # (B, 4) or (B, 6)
        return attrs

    # ------------------------------------------------------------------ #
    #  Decoding
    # ------------------------------------------------------------------ #
    def _decode_latents_to_poi_probs(
        self, latents: torch.Tensor, num_special_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode VAE-generated latents through BART decoder to get POI probabilities."""
        num_special_tokens = num_special_tokens if num_special_tokens is not None else self.num_special_tokens
        # Unproject from PCA if needed
        if self.latent_pca is not None:
            latents = self.latent_pca.unproject(latents)

        encoder_outputs = BaseModelOutput(last_hidden_state=latents)

        bs, seq_len = latents.shape[0], latents.shape[1]
        bos_id = getattr(self.autoencoder.config, "decoder_start_token_id", None)
        if bos_id is None:
            bos_id = getattr(self.autoencoder.config, "bos_token_id", 1)
        decoder_input_ids = torch.full(
            (bs, seq_len), bos_id, device=self.device, dtype=torch.long
        )

        decoder_out = self.autoencoder(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = decoder_out.logits  # (B, T, V)

        # Ban special tokens before softmax (same as DiT version)
        if num_special_tokens > 0:
            logits[..., :num_special_tokens] = logits[..., :num_special_tokens] - 1e4

        poi_probs = F.softmax(logits, dim=-1)

        # Drop special tokens so last dim matches POI-only vocab in p_poi.csv
        if num_special_tokens > 0:
            poi_probs = poi_probs[..., num_special_tokens:]

        return poi_probs

    # ------------------------------------------------------------------ #
    #  LLP mixture KL (ported from run_cbg_conditioned_training.py)
    # ------------------------------------------------------------------ #
    def _llp_mixture_kl(
        self,
        poi_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        cbg: str,
        batch_age: torch.Tensor,
        batch_gender: torch.Tensor,
        epsilon: float = 1e-8,
        cache: Optional[CBGConditionCache] = None,
        poi_store: Optional[POIMarginalStore] = None,
        num_special_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """LLP mixture KL:
        1) Build per-group POI distributions P_hat[d] from current batch predictions.
        2) Mix them with pi_cbg,d to obtain P_mix.
        3) Compute divergence( P_target_cbg || P_mix ).
        """
        cache = cache or self.cache
        poi_store = poi_store or self.poi_store
        B, T, V = poi_probs.shape
        device = poi_probs.device
        D_age = max(int(self.num_age_bins), 1)
        D_gen = max(int(self.num_genders), 1)
        D = D_age * D_gen

        # Group indices: d = age * num_genders + gender (raw ids are 0-based)
        group_idx = (batch_age.long().clamp_min(0) * D_gen + batch_gender.long().clamp_min(0)).clamp(0, D - 1)
        one_hot = F.one_hot(group_idx, num_classes=D).to(dtype=poi_probs.dtype, device=device)  # (B, D)

        # Build attention weights: (B, T, D)
        if attention_mask is not None:
            bt_weights = attention_mask.to(dtype=poi_probs.dtype, device=device).unsqueeze(-1) * one_hot.unsqueeze(1)
        else:
            bt_weights = one_hot.unsqueeze(1).expand(B, T, D).to(dtype=poi_probs.dtype, device=device)

        # Weighted token mass per group: (D, V)
        group_mass = (poi_probs.unsqueeze(2) * bt_weights.unsqueeze(-1)).sum(dim=(0, 1))
        # Token counts per group: (D,)
        group_counts = bt_weights.sum(dim=(0, 1)).clamp_min(1.0)
        P_hat = group_mass / group_counts.unsqueeze(-1)

        # Fallback for empty groups: replace with overall batch marginal
        overall = aggregate_poi_distribution(poi_probs, attention_mask=attention_mask, epsilon=epsilon)  # (V,)
        empty_groups = (group_counts <= 1.0)
        if empty_groups.any():
            P_hat[empty_groups] = overall.unsqueeze(0).expand(int(empty_groups.sum().item()), V)

        # pi vector for this CBG: (D,)
        pi_cbg = cache.as_torch_distribution(
            cbg,
            num_age_bins=D_age,
            num_genders=D_gen,
            device=device,
        ).to(dtype=poi_probs.dtype)
        pi_cbg = (pi_cbg + epsilon) / (pi_cbg.sum() + epsilon * pi_cbg.numel())

        # Mixture distribution: (V,)
        mix = (pi_cbg.unsqueeze(-1) * P_hat).sum(dim=0)
        mix = (mix + epsilon) / (mix.sum() + epsilon * mix.numel())

        # Target P_cbg
        target = poi_store.get_distribution(cbg, device=device).to(dtype=mix.dtype)
        target = _align_target_to_poi_dim(
            target,
            int(mix.shape[0]),
            num_special_tokens=(
                self.num_special_tokens if num_special_tokens is None else int(num_special_tokens)
            ),
            context=f"LLP aggregate target for CBG {cbg}",
        )
        target = (target + epsilon) / (target.sum() + epsilon * target.numel())

        loss = _distribution_loss(target, mix, loss_type=self.aggregate_loss_type, epsilon=epsilon)
        stats = {f"agg_{self.aggregate_loss_type}": float(loss.item())}
        return loss, stats

    # ------------------------------------------------------------------ #
    #  Aggregate loss dispatch
    # ------------------------------------------------------------------ #
    def _compute_aggregate_loss(
        self,
        poi_probs: torch.Tensor,
        cbg: str,
        age_raw: Optional[torch.Tensor],
        gender_raw: Optional[torch.Tensor],
        cache: Optional[CBGConditionCache] = None,
        poi_store: Optional[POIMarginalStore] = None,
        num_special_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute aggregate loss: LLP mixture KL when enabled, else flat divergence."""
        cache = cache or self.cache
        poi_store = poi_store or self.poi_store
        special_tokens = self.num_special_tokens if num_special_tokens is None else int(num_special_tokens)
        if self.llp_enabled and self.num_age_bins > 0 and self.num_genders > 0:
            if age_raw is None or gender_raw is None:
                raise RuntimeError("LLP is enabled but no demographic IDs were provided.")
            loss, stats = self._llp_mixture_kl(
                poi_probs,
                attention_mask=None,
                cbg=cbg,
                batch_age=age_raw,
                batch_gender=gender_raw,
                cache=cache,
                poi_store=poi_store,
                num_special_tokens=special_tokens,
            )
        else:
            # Flat aggregate divergence (no demographic stratification)
            target = poi_store.get_distribution(cbg, device=poi_probs.device)
            target = _align_target_to_poi_dim(
                target,
                int(poi_probs.shape[-1]),
                num_special_tokens=special_tokens,
                context=f"aggregate target for CBG {cbg}",
            )

            if self.aggregate_loss_type == "kl":
                loss, stats = poi_marginal_kl_loss(
                    poi_probs,
                    target,
                    attention_mask=None,
                    epsilon=self.aggregate_loss_eps,
                )
            else:
                pred = aggregate_poi_distribution(
                    poi_probs, attention_mask=None, epsilon=self.aggregate_loss_eps
                )
                loss = _distribution_loss(
                    target.to(device=pred.device, dtype=pred.dtype),
                    pred,
                    loss_type=self.aggregate_loss_type,
                    epsilon=self.aggregate_loss_eps,
                )
                stats = {f"agg_{self.aggregate_loss_type}": float(loss.item())}

        return loss, stats

    # ------------------------------------------------------------------ #
    #  Original VAE reconstruction/KL loss
    # ------------------------------------------------------------------ #
    def _next_vae_loss_batch(self) -> Dict[str, Any]:
        if self.vae_loss_loader is None or self.vae_loss_iter is None:
            raise RuntimeError("Original VAE loss requested but no dataloader was initialized.")
        try:
            batch = next(self.vae_loss_iter)
        except StopIteration:
            self.vae_loss_iter = iter(self.vae_loss_loader)
            batch = next(self.vae_loss_iter)
        return _move_batch_to_device(batch, self.device)

    def _vae_beta_for_step(self, step: int) -> float:
        if self.vae_kl_anneal_steps > 0:
            frac = min(1.0, float(max(step + 1, 0)) / float(self.vae_kl_anneal_steps))
            return frac * self.vae_kl_beta_max
        return self.vae_kl_beta_max

    def _compute_vae_loss_from_batch(
        self,
        batch: Dict[str, Any],
        step: int,
        *,
        apply_conditional_dropout: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.vae_loss_args is None:
            raise RuntimeError("Original VAE loss requested but args were not initialized.")

        attrs = _prepare_vae_loss_attrs(batch, self.vae_loss_args, self.vae)
        target_latents = _extract_target_latents_for_vae_loss(
            batch,
            self.autoencoder,
            self.vae_loss_args,
            self.latent_pca,
        )

        attr_embeds_for_model = None
        if self.vae_loss_args.data_type == "unified":
            is_conditional = batch.get("is_conditional")
            if is_conditional is not None and attrs is not None:
                should_be_conditional = is_conditional.bool()
                if apply_conditional_dropout and self.vae_loss_args.conditional_dropout > 0:
                    dropout_mask = (
                        torch.rand(is_conditional.shape, device=is_conditional.device)
                        < self.vae_loss_args.conditional_dropout
                    )
                    should_be_conditional = should_be_conditional & ~dropout_mask
                if should_be_conditional.any():
                    attr_embeds_for_model = attrs.clone()
                    attr_embeds_for_model[~should_be_conditional] = 0
            elif attrs is not None:
                attr_embeds_for_model = attrs
        elif attrs is not None and attrs.abs().sum() > 0:
            attr_embeds_for_model = attrs

        beta_kl = self._vae_beta_for_step(step)
        loss_dict = self.vae.compute_loss(
            x=target_latents,
            attr_embeds=attr_embeds_for_model,
            attention_mask=batch.get("attention_mask"),
            beta_kl=beta_kl,
        )

        return loss_dict["total_loss"], {
            "vae_loss": float(loss_dict["total_loss"].item()),
            "vae_recon_loss": float(loss_dict["recon_loss"].item()),
            "vae_kl_loss": float(loss_dict["kl_loss"].item()),
            "vae_kl_beta": float(beta_kl),
        }

    def _compute_original_vae_loss(self, step: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = self._next_vae_loss_batch()
        return self._compute_vae_loss_from_batch(
            batch,
            step,
            apply_conditional_dropout=True,
        )

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def validate(self, step: Optional[int] = None) -> Dict[str, float]:
        """Run held-out aggregate/LLP and optional VAE reconstruction/KL validation."""
        can_validate_agg = (
            self.val_enabled
            and self.val_cache is not None
            and self.val_poi_store is not None
            and bool(self.val_cbgs)
        )
        can_validate_vae = (
            self.use_vae_loss
            and self.vae_loss_val_loader is not None
            and self.vae_loss_args is not None
        )
        if not can_validate_agg and not can_validate_vae:
            return {}

        self.vae.eval()
        total_agg = 0.0
        n = 0
        num_cbgs = len(self.val_cbgs) if self.val_cbgs is not None else 0
        sum_agg_by_idx = torch.zeros(num_cbgs, device=self.device, dtype=torch.float32)
        count_by_idx = torch.zeros(num_cbgs, device=self.device, dtype=torch.float32)

        total_vae = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_beta = 0.0
        n_vae = 0

        with torch.no_grad():
            if can_validate_agg:
                assert self.val_cache is not None
                assert self.val_poi_store is not None
                assert self.val_cbgs is not None
                for _ in range(self.val_num_batches):
                    cbg_idx = random.randrange(num_cbgs)
                    cbg = self.val_cbgs[cbg_idx]

                    # Sample demo IDs for this val batch.
                    age_raw = None
                    gender_raw = None
                    if (
                        self.llp_enabled
                        and self.val_llp_demo_source == "pi"
                        and self.num_age_bins > 0
                        and self.num_genders > 0
                    ):
                        age_raw, gender_raw, _ = self._sample_demo_ids_from_pi(
                            cbg, self.val_batch_size, cache=self.val_cache,
                        )

                    attrs = self._build_batch_from(
                        cbg, self.val_batch_size, self.val_cache,
                        age_raw=age_raw, gender_raw=gender_raw,
                    )

                    latents = self.vae.generate(
                        batch_size=self.val_batch_size,
                        attr_embeds=attrs,
                        device=self.device,
                    )

                    poi_probs = self._decode_latents_to_poi_probs(
                        latents, num_special_tokens=self.val_num_special_tokens,
                    )

                    agg_loss, _ = self._compute_aggregate_loss(
                        poi_probs, cbg, age_raw, gender_raw,
                        cache=self.val_cache, poi_store=self.val_poi_store,
                        num_special_tokens=self.val_num_special_tokens,
                    )
                    total_agg += float(agg_loss.item())
                    n += 1

                    if self.val_log_by_cbg:
                        sum_agg_by_idx[cbg_idx] += float(agg_loss.item())
                        count_by_idx[cbg_idx] += 1.0

            if can_validate_vae:
                for batch_idx, batch in enumerate(self.vae_loss_val_loader):
                    if batch_idx >= self.vae_loss_val_batches:
                        break
                    batch = _move_batch_to_device(batch, self.device)
                    _, stats = self._compute_vae_loss_from_batch(
                        batch,
                        max(int(step if step is not None else self.start_step), 0),
                        apply_conditional_dropout=False,
                    )
                    total_vae += stats["vae_loss"]
                    total_recon += stats["vae_recon_loss"]
                    total_kl += stats["vae_kl_loss"]
                    total_beta += stats["vae_kl_beta"]
                    n_vae += 1

        self.vae.train()

        out: Dict[str, float] = {}
        if n > 0:
            out["val_agg_loss"] = total_agg / n
        if n_vae > 0:
            out["val_vae_loss"] = total_vae / n_vae
            out["val_vae_recon_loss"] = total_recon / n_vae
            out["val_vae_kl_loss"] = total_kl / n_vae
            out["val_vae_kl_beta"] = total_beta / n_vae

        if self.val_log_by_cbg and num_cbgs > 0:
            max_to_log = self.val_max_cbgs_to_log if self.val_max_cbgs_to_log > 0 else (num_cbgs if num_cbgs <= 8 else 0)
            logged = 0
            for i, cbg in enumerate(self.val_cbgs):
                if logged >= max_to_log:
                    break
                cnt = float(count_by_idx[i].item())
                if cnt <= 0:
                    continue
                out[f"val_agg_loss_by_cbg/{cbg}"] = float(sum_agg_by_idx[i].item()) / cnt
                logged += 1

        return out

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def train(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is not installed, but training.wandb is true.")
            init_kwargs: Dict[str, Any] = {
                "project": self.wandb_project,
                "config": self.config,
            }
            if self.wandb_name:
                init_kwargs["name"] = self.wandb_name
            wandb.init(**init_kwargs)

        self.vae.train()
        running_agg_loss = 0.0
        running_vae_loss = 0.0
        running_vae_recon_loss = 0.0
        running_vae_kl_loss = 0.0

        pbar = tqdm(range(self.start_step, self.start_step + self.steps), desc="CBG finetune (VAE)")

        for step in pbar:
            cbg = self._sample_cbg()

            # Decide demographic IDs for conditioning + LLP grouping.
            age_raw = None
            gender_raw = None
            if self.llp_enabled and self.llp_demo_source == "pi" and self.num_age_bins > 0 and self.num_genders > 0:
                age_raw, gender_raw, _ = self._sample_demo_ids_from_pi(cbg, self.batch_size)

            attrs = self._build_batch(cbg, age_raw=age_raw, gender_raw=gender_raw)

            # VAE generation: sample from prior and decode.
            # Differentiable through reparameterization trick in generate().
            latents = self.vae.generate(
                batch_size=self.batch_size,
                attr_embeds=attrs,
                device=self.device,
            )  # (B, T, D)

            # Decode to POI probabilities
            poi_probs = self._decode_latents_to_poi_probs(latents)

            # Aggregate loss (LLP or flat depending on config)
            agg_loss, agg_stats = self._compute_aggregate_loss(
                poi_probs, cbg, age_raw, gender_raw
            )

            total_loss = self.lambda_agg * agg_loss
            vae_stats: Dict[str, float] = {}
            if self.use_vae_loss and self.lambda_vae > 0:
                vae_loss, vae_stats = self._compute_original_vae_loss(step)
                total_loss = total_loss + self.lambda_vae * vae_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)
            self.optimizer.step()

            running_agg_loss += agg_loss.item()
            if vae_stats:
                running_vae_loss += vae_stats["vae_loss"]
                running_vae_recon_loss += vae_stats["vae_recon_loss"]
                running_vae_kl_loss += vae_stats["vae_kl_loss"]

            # Logging
            if (step + 1) % self.log_every == 0:
                avg_agg = running_agg_loss / self.log_every
                postfix = {"agg_loss": f"{avg_agg:.4f}"}
                msg = f"Step {step + 1} | agg_loss: {avg_agg:.4f}"
                avg_vae = avg_recon = avg_kl = None
                if self.use_vae_loss and self.lambda_vae > 0:
                    avg_vae = running_vae_loss / self.log_every
                    avg_recon = running_vae_recon_loss / self.log_every
                    avg_kl = running_vae_kl_loss / self.log_every
                    postfix["vae_loss"] = f"{avg_vae:.4f}"
                    msg += f" | vae_loss: {avg_vae:.4f} | recon: {avg_recon:.4f} | kl: {avg_kl:.4f}"
                pbar.set_postfix(**postfix)
                print(f"{msg} | cbg: {cbg}")

                if self.use_wandb:
                    log_dict = {
                        "agg_loss": avg_agg,
                        "total_loss": float(total_loss.item()),
                        "step": step + 1,
                    }
                    log_dict.update(agg_stats)
                    log_dict.update(vae_stats)
                    if avg_vae is not None:
                        log_dict.update({
                            "vae_loss_avg": float(avg_vae),
                            "vae_recon_loss_avg": float(avg_recon),
                            "vae_kl_loss_avg": float(avg_kl),
                        })
                    if self.log_agg_by_cbg and isinstance(cbg, str) and cbg:
                        log_dict[f"agg_loss_by_cbg/{cbg}"] = float(agg_loss.item())
                    wandb.log(log_dict)
                running_agg_loss = 0.0
                running_vae_loss = 0.0
                running_vae_recon_loss = 0.0
                running_vae_kl_loss = 0.0

            # Validation (no early stopping)
            if self.val_enabled and self.val_every > 0 and (step + 1) % self.val_every == 0:
                val_metrics = self.validate(step + 1)
                if val_metrics:
                    msg = " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()
                                    if not k.startswith("val_agg_loss_by_cbg/")])
                    print(f"[val {step + 1:05d}] {msg}")
                    if self.use_wandb and wandb.run is not None:
                        wandb.log(val_metrics, step=step + 1)

            # Checkpointing
            if (step + 1) % self.save_every == 0:
                ckpt_path = self.output_dir / f"vae_cbg_step_{step + 1}.pt"
                torch.save({
                    "model": self.vae.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": step + 1,
                    "config": self.config,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Final save
        final_path = self.output_dir / "vae_cbg_final.pt"
        torch.save({
            "model": self.vae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.start_step + self.steps,
            "config": self.config,
        }, final_path)
        print(f"Saved final model: {final_path}")

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="CBG aggregate fine-tuning for VAE backbone")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device_str = args.device or config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = AggregateTrainerVAE(config, device)
    trainer.train()


if __name__ == "__main__":
    main()
