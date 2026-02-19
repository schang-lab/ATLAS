"""
CBG-conditioned aggregate POI trainer implementation.

AggregateTrainer is the main training class; typically invoked via
run_entrypoint(AggregateTrainer) from the run_cbg_conditioned_training entrypoint.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import BaseModelOutput

# Ensure trajectory-generation root is on path (lib is 3 levels deep)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.data import (
    CBGConditionCache,
    CategoryMapSpec,
    CategoryTransitionStore,
    POICategoryMap,
    POIMarginalStore,
)
from src.diffusion_model import GaussianDiffusion
from src.latent_pca import LatentPCA
from src.losses import aggregate_poi_distribution
from src.training import trajectory_dataset

from lib.run_cbg_bootstrap import (
    _posterior_sample,
    build_autoencoder,
    build_condition_encoder,
    build_dit,
    build_noise_scheduler,
    sample_latents,
    select_cbgs,
)
from lib.run_cbg_losses import (
    aggregate_feature_losses,
    apply_poi_token_mask_to_dist,
    apply_poi_token_mask_to_probs,
    category_transition_probs,
    distribution_loss,
    normalize_dist,
    poi_mask_index_tensor,
    transition_mask,
)
from lib.run_cbg_validation import validate as run_cbg_validate
from lib.run_cbg_train_step import training_step as run_cbg_training_step


class AggregateTrainer:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        diff_cfg = config["diffusion"]
        self.prediction_type = diff_cfg.get("prediction_type", "epsilon")
        # Match timestep sampling behavior used in the main DiT trainer
        self.timestep_sampling = diff_cfg.get("timestep_sampling", "logsnr")
        self.seq_len = int(diff_cfg["seq_len"])
        self.latent_dim = int(diff_cfg["latent_dim"])

        self.dit = build_dit(diff_cfg, device)
        self.autoencoder = build_autoencoder(config["autoencoder"], device)
        self.condition_encoder = None
        self.scheduler = build_noise_scheduler(diff_cfg).to(device)

        # Cache demographic config from the model for LLP grouping
        base_dit = getattr(self.dit, "module", self.dit)
        self.num_age_bins = int(getattr(getattr(base_dit, "attr_embed", None), "num_age_bins", 0))
        self.num_genders = int(getattr(getattr(base_dit, "attr_embed", None), "num_genders", 0))

        # Optional PCA unprojection for decoding (supports latent_dim < AE d_model)
        lp_cfg = config.get("latent_pca", {}) or {}
        self.latent_pca = None
        pca_path = lp_cfg.get("path")
        if pca_path:
            self.latent_pca = LatentPCA(pca_path, device=device)

        # Optional diffusion MSE branch (supervised diffusion loss)
        mse_cfg = config.get("diffusion_mse", {}) or {}
        self.use_diffusion_mse = bool(mse_cfg.get("enabled", False))
        self.lambda_mse = float(mse_cfg.get("weight", 1.0)) if self.use_diffusion_mse else 0.0
        self.mse_loader = None
        self.mse_iter = None
        if self.use_diffusion_mse:
            data_dir = mse_cfg.get("data_dir")
            if not data_dir:
                raise ValueError("diffusion_mse.enabled is true but diffusion_mse.data_dir is not set in config")
            # Allow independent anchor batch size but default to training batch size
            train_cfg = config["training"]
            default_mse_bs = int(train_cfg.get("batch_size", 128))
            mse_bs = int(mse_cfg.get("batch_size", default_mse_bs))
            num_workers = int(mse_cfg.get("num_workers", 4))
            data_type = str(mse_cfg.get("data_type", "unified"))
            self.mse_conditional_dropout = float(mse_cfg.get("conditional_dropout", 0.0))
            # MSE demo conditioning mode:
            # - "null": ignore demo dims (non-cheating baseline; default)
            # - "pi": sample demo ids from π_cbg (uses cbg_cache_dir demographic mixture)
            # - "data": use demo ids from the supervised dataset (requires all_attr_results_with_demo.npy)
            demo_source_raw = mse_cfg.get("demo_source", None)
            if demo_source_raw is None:
                # Backward-compat: keep support for diffusion_mse.randomize_demo_from_cbg
                self.mse_demo_source = "pi" if bool(mse_cfg.get("randomize_demo_from_cbg", False)) else "null"
            else:
                self.mse_demo_source = str(demo_source_raw).lower().strip()
            if self.mse_demo_source not in {"null", "pi", "data"}:
                raise ValueError(
                    f"diffusion_mse.demo_source must be one of ['null','pi','data'] (got {self.mse_demo_source!r})"
                )
            # Optional: randomly assign demo ids for MSE branch from current CBG's demo distribution.
            # (Deprecated in favor of diffusion_mse.demo_source: pi)
            self.mse_randomize_demo = bool(mse_cfg.get("randomize_demo_from_cbg", False))
            # Minimal args for trajectory_dataset
            args = SimpleNamespace(
                BATCH_SIZE=mse_bs,
                NUM_WORKERS=num_workers,
                # Ensure tokenized sequence length matches this finetune run's seq_len.
                # Otherwise the Phase-1 BART encoder can crash on position embeddings when
                # the dataset defaults to longer sequences (e.g., 512).
                sequence_length=int(self.seq_len),
                training_phase="phase1",
                ablation_mode="neither",
                enable_length_condition=False,
                rebuild_attention_masks=True,
                force_full_attention_mask=False,
            )
            self.mse_loader, _, _, _ = trajectory_dataset(args, testset=False, data_dir=data_dir, data_type=data_type)
            self.mse_iter = iter(self.mse_loader)

        data_cfg = config["data"]
        self.cache = CBGConditionCache(data_cfg["cbg_cache_dir"])
        self.poi_store = POIMarginalStore(data_cfg["poi_marginal_csv"])
        self.cbgs = select_cbgs(self.cache, self.poi_store, data_cfg.get("allowed_cbgs"))
        # Number of special tokens at the start of the vocab that should be excluded
        # from POI marginals (must match the value used when building p_poi.csv).
        self.num_special_tokens = int(data_cfg.get("num_special_tokens", 0))

        # Optional per-CBG trajectory length distributions (JSON produced by build_length_dists.py).
        self.length_dists: Optional[Dict[str, np.ndarray]] = None
        self.length_city_probs: Optional[np.ndarray] = None
        self.length_min = int(data_cfg.get("min_traj_len", 7))
        self.length_max = int(data_cfg.get("max_traj_length", self.seq_len))
        length_json = data_cfg.get("length_dists_json")
        if length_json is not None:
            self._load_length_distributions(length_json)

        train_cfg = config["training"]
        lr = float(train_cfg.get("lr", 1e-5))
        self.lambda_agg = float(train_cfg.get("lambda_agg", 1.0))
        # Optional: skip expensive aggregate sampling/decoding when lambda_agg==0
        # (useful for supervised-only baselines via diffusion_mse).
        self.skip_agg_when_lambda_zero = bool(train_cfg.get("skip_aggregate_when_lambda_zero", False))
        self.guidance_scale = float(train_cfg.get("guidance_scale", 1.0))
        self.steps = int(train_cfg.get("steps", 1000))
        self.batch_size = int(train_cfg.get("batch_size", 128))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.save_every = int(train_cfg.get("save_every", 500))
        self.output_dir = Path(train_cfg.get("output_dir", "./cbg_condition_finetune"))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        # Optional Weights & Biases logging toggle (config.training.wandb: true/false).
        self.use_wandb = bool(train_cfg.get("wandb", False))
        # Optional per-CBG aggregate loss logging to wandb (keyed as agg_loss_by_cbg/<cbg>).
        self.log_agg_by_cbg = bool(train_cfg.get("log_agg_by_cbg", False))
        # LLP (Label Proportion Learning) configuration (opt-in)
        llp_cfg = train_cfg.get("llp", {}) or {}
        # Backward-compat: allow a flat boolean toggle training.llp_enabled
        self.llp_enabled = bool(llp_cfg.get("enabled", train_cfg.get("llp_enabled", False)))
        # condition_mode kept for future extension; currently only "full" is used here
        self.llp_condition_mode = str(llp_cfg.get("condition_mode", "full"))
        # LLP demo id source:
        # - "cache": use per-trajectory demo ids stored in cbg cache (supervised by instance labels)
        # - "pi": sample demo ids i.i.d. from π_cbg and use those both for conditioning and LLP grouping
        self.llp_demo_source = str(llp_cfg.get("demo_source", "cache")).lower().strip()
        if self.llp_demo_source not in {"cache", "pi"}:
            raise ValueError(f"training.llp.demo_source must be 'cache' or 'pi' (got {self.llp_demo_source!r})")
        # Optional: enforce that each demo group appears at least K times in the batch (K>=0).
        # Useful to avoid empty-group fallbacks in LLP when batch sizes are small.
        self.llp_min_per_group = int(llp_cfg.get("min_per_group", 0))
        if self.llp_min_per_group < 0:
            self.llp_min_per_group = 0
        # Guard against accidentally enabling LLP without demo conditioning in the DiT config.
        if self.llp_enabled and (self.num_age_bins <= 0 or self.num_genders <= 0):
            raise ValueError(
                "LLP is enabled but the DiT model has no demographic conditioning configured. "
                "Update your DiT config (diffusion.config) to set use_demo_condition: true "
                "and set num_age_bins / num_genders to match your cache."
            )
        # Optional: simple x0 training over last K near-zero steps
        self.simple_x0 = bool(train_cfg.get("simple_x0", False))
        self.simple_x0_last_k = int(train_cfg.get("simple_x0_last_k", 1))
        if self.simple_x0_last_k <= 0:
            self.simple_x0_last_k = 1
        # Optional: entropy-style regularizer on per-trajectory POI histograms to encourage diversity.
        ent_cfg = train_cfg.get("entropy_reg", {}) or {}
        self.use_entropy_reg = bool(ent_cfg.get("enabled", False))
        self.lambda_entropy = float(ent_cfg.get("weight", 0.0)) if self.use_entropy_reg else 0.0
        # Optional: expected-unique-POI regularizer (Approach A) to push each trajectory
        # toward visiting more distinct POIs.
        uniq_cfg = train_cfg.get("unique_reg", {}) or {}
        self.use_unique_reg = bool(uniq_cfg.get("enabled", False))
        self.lambda_unique = float(uniq_cfg.get("weight", 0.0)) if self.use_unique_reg else 0.0
        # Aggregate loss type (applies to both standard aggregate and LLP mixture branch).
        # Supported: "kl" (default), "js", "tv", "hellinger".
        agg_loss_cfg = train_cfg.get("aggregate_loss", {}) or {}
        self.aggregate_loss_type = str(agg_loss_cfg.get("type", train_cfg.get("aggregate_loss_type", "kl"))).lower()
        if self.aggregate_loss_type not in {"kl", "js", "tv", "hellinger"}:
            raise ValueError(
                f"training.aggregate_loss.type must be one of kl|js|tv|hellinger (got {self.aggregate_loss_type!r})"
            )
        self.aggregate_loss_eps = float(agg_loss_cfg.get("epsilon", 1e-8))

        # ------------------------------------------------------------------
        # Aggregate feature space (Option A/D):
        # - "poi" (default): POI histogram
        # - "category": POI->category mapping then histogram loss (Option A)
        # - "category_transition": category bigram distribution loss (Option D)
        # - "category+transition": sum of category + transition losses
        # ------------------------------------------------------------------
        feat = str(train_cfg.get("aggregate_feature", "poi")).lower().strip()
        if feat in {"cat", "category_hist"}:
            feat = "category"
        if feat in {"cat_trans", "category_bigram", "transition"}:
            feat = "category_transition"
        if feat in {"category+category_transition", "category_transition+category", "cat+trans"}:
            feat = "category+transition"
        if feat not in {"poi", "category", "category_transition", "category+transition"}:
            raise ValueError(
                "training.aggregate_feature must be one of "
                "poi|category|category_transition|category+transition "
                f"(got {feat!r})"
            )
        self.aggregate_feature = feat
        self.category_weight = float(train_cfg.get("category_weight", 1.0))
        self.transition_weight = float(train_cfg.get("transition_weight", 1.0))

        # Category mapping + transition targets (optional).
        # Category mapping uses `poi_map_feature.csv` and the tokenizer vocab to align IDs.
        self.category_map: Optional[POICategoryMap] = None
        self.category_transition_store: Optional[CategoryTransitionStore] = None
        poi_map_csv = data_cfg.get("poi_map_feature_csv")
        poi_cat_col = str(data_cfg.get("poi_category_column", "top_category"))
        data_vocab_path = data_cfg.get("vocab_path")
        trans_npz = data_cfg.get("category_transition_npz")
        # Drop these POI tokens from category-based aggregates by default to avoid collapse.
        category_drop_poi_tokens = data_cfg.get("category_drop_poi_tokens", None)
        if category_drop_poi_tokens is None:
            category_drop_poi_tokens = ["POI_HOME", "POI_WORK", "POI_OTHER"]
        category_drop_poi_tokens = tuple(str(x) for x in list(category_drop_poi_tokens))

        if self.aggregate_feature != "poi":
            if not poi_map_csv:
                raise ValueError("data.poi_map_feature_csv is required when training.aggregate_feature != 'poi'")
            if not data_vocab_path:
                raise ValueError("data.vocab_path is required when training.aggregate_feature != 'poi'")
            if trans_npz:
                self.category_transition_store = CategoryTransitionStore(str(trans_npz))
                cat_order = self.category_transition_store.categories
            else:
                cat_order = None
            self.category_map = POICategoryMap(
                CategoryMapSpec(
                    poi_map_csv=str(poi_map_csv),
                    vocab_path=str(data_vocab_path),
                    num_special_tokens=int(self.num_special_tokens),
                    category_column=poi_cat_col,
                    drop_poi_tokens=category_drop_poi_tokens,
                ),
                categories=cat_order,
            )
            if self.aggregate_feature in {"category_transition", "category+transition"} and self.category_transition_store is None:
                raise ValueError(
                    "training.aggregate_feature requests transitions but data.category_transition_npz is not set."
                )

        # If using transitions, ensure training CBG list overlaps the transition store.
        if self.category_transition_store is not None and self.aggregate_feature in {"category_transition", "category+transition"}:
            have = set(self.category_transition_store.available_cbgs())
            self.cbgs = [c for c in self.cbgs if c in have]
            if not self.cbgs:
                raise ValueError("No overlapping CBGs between cache/poi_marginals and category_transition_npz.")

        # Optional: randomly drop (home, work) coords for the aggregate/LLP branch only.
        # This is a targeted ablation to reduce over-reliance on coords while keeping the MSE
        # branch as a stabilizing anchor.
        self.aggregate_coord_dropout = float(train_cfg.get("aggregate_coord_dropout", 0.0) or 0.0)
        if not (0.0 <= self.aggregate_coord_dropout <= 1.0):
            raise ValueError("training.aggregate_coord_dropout must be in [0, 1]")

        # Optional: exclude specific POI tokens from the aggregate / LLP histogram matching objective.
        # Motivation: avoid POI_HOME/POI_WORK/POI_OTHER dominating the aggregate objective and collapsing diversity.
        poi_mask_cfg = train_cfg.get("poi_token_mask", {}) or {}
        self.poi_token_mask_enabled = bool(poi_mask_cfg.get("enabled", False))
        self.poi_token_mask_eps = float(poi_mask_cfg.get("epsilon", 1e-12))
        self.poi_token_mask_renormalize = bool(poi_mask_cfg.get("renormalize", True))
        self.poi_token_mask_indices: List[int] = []
        self.poi_token_mask_tokens: List[str] = []
        if self.poi_token_mask_enabled:
            drop_tokens = poi_mask_cfg.get("drop_tokens", None)
            drop_poi_indices = poi_mask_cfg.get("drop_poi_indices", None)
            # If enabled but nothing specified, default to the common special POI tokens.
            if not drop_tokens and not drop_poi_indices:
                drop_tokens = ["POI_HOME", "POI_WORK", "POI_OTHER"]

            indices: List[int] = []
            if drop_poi_indices:
                indices.extend([int(x) for x in list(drop_poi_indices)])

            if drop_tokens:
                vocab_path = poi_mask_cfg.get("vocab_path") or data_cfg.get("vocab_path")
                if not vocab_path:
                    raise ValueError(
                        "training.poi_token_mask.drop_tokens requires either "
                        "training.poi_token_mask.vocab_path or data.vocab_path"
                    )
                vocab_lines = Path(str(vocab_path)).read_text(encoding="utf-8").splitlines()
                vocab = [ln.strip() for ln in vocab_lines if ln.strip()]
                token_to_id = {t: i for i, t in enumerate(vocab)}
                for t in list(drop_tokens):
                    tok = str(t)
                    if tok not in token_to_id:
                        raise ValueError(f"training.poi_token_mask: token {tok!r} not found in vocab at {vocab_path}")
                    full_id = int(token_to_id[tok])
                    poi_idx = int(full_id - int(self.num_special_tokens))
                    indices.append(poi_idx)
                    self.poi_token_mask_tokens.append(tok)

            # Validate + dedupe
            V_eff = int(self.poi_store.vocab_size())
            uniq = sorted({int(x) for x in indices})
            bad = [x for x in uniq if x < 0 or x >= V_eff]
            if bad:
                raise ValueError(
                    f"training.poi_token_mask has POI indices out of range [0, {V_eff}): {bad}. "
                    "Note: drop_poi_indices are in POI-only indexing (after removing num_special_tokens)."
                )
            if len(uniq) >= V_eff:
                raise ValueError("training.poi_token_mask would mask all POI tokens; refuse to continue.")
            self.poi_token_mask_indices = uniq

        # Optional validation loop (no early stopping). This validates the aggregate/LLP objective
        # on a separate set of caches + POI marginals (typically built from val split).
        val_cfg = train_cfg.get("validation", {}) or {}
        self.val_enabled = bool(val_cfg.get("enabled", False))
        self.val_every = int(val_cfg.get("eval_every", 0) or 0)
        self.val_num_batches = int(val_cfg.get("num_batches", 10) or 10)
        self.val_batch_size = int(val_cfg.get("batch_size", self.batch_size) or self.batch_size)
        self.val_log_by_cbg = bool(val_cfg.get("log_by_cbg", False))
        self.val_max_cbgs_to_log = int(val_cfg.get("max_cbgs_to_log", 0) or 0)
        if self.val_max_cbgs_to_log < 0:
            self.val_max_cbgs_to_log = 0
        if self.val_num_batches < 1:
            self.val_num_batches = 1
        if self.val_every < 0:
            self.val_every = 0
        # Optional: track resume start step for logging and checkpoint numbering
        self.start_step = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate sampler (controls the expensive reverse sampling used by the aggregate/LLP branch).
        # - ddpm: original ancestral sampling over all diffusion.timesteps
        # - ddim: reduced-step DDIM sampling (much faster). Uses eta=0 by default.
        sampler_cfg = train_cfg.get("aggregate_sampler", {}) or {}
        self.aggregate_sampler = str(sampler_cfg.get("method", "ddpm")).lower().strip()
        self.aggregate_ddim_steps = int(sampler_cfg.get("ddim_steps", 50) or 50)
        self.aggregate_ddim_eta = float(sampler_cfg.get("ddim_eta", sampler_cfg.get("eta", 0.0)) or 0.0)

        train_params = list(self.dit.parameters())
        self.optimizer = optim.Adam(train_params, lr=lr)

        # Simple linear warmup schedule followed by constant LR (similar spirit to main trainer).
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        if warmup_steps > 0:
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step + 1) / float(max(1, warmup_steps))
                return 1.0

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.lr_scheduler = None

        self.attention_mask = torch.ones(
            self.batch_size,
            self.seq_len,
            dtype=torch.long,
            device=self.device,
        )

        # Validation data objects (optional)
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
            self.val_num_special_tokens = int(val_cfg.get("num_special_tokens", self.num_special_tokens))
            if self.poi_token_mask_enabled and int(self.val_num_special_tokens) != int(self.num_special_tokens):
                raise ValueError(
                    "training.poi_token_mask is enabled but validation.num_special_tokens differs from "
                    "train data.num_special_tokens; masking indices would be inconsistent."
                )
            # Optional override: validate using demo labels from cache instead of π (diagnostic).
            v_demo = str(val_cfg.get("llp_demo_source", "") or "").lower().strip()
            if v_demo:
                if v_demo not in {"inherit", "cache", "pi"}:
                    raise ValueError(
                        "training.validation.llp_demo_source must be one of inherit|cache|pi "
                        f"(got {v_demo!r})."
                    )
                self.val_llp_demo_source = self.llp_demo_source if v_demo == "inherit" else v_demo
            # Optional: coord dropout for validation aggregate branch (default: 0; usually keep val clean).
            self.val_aggregate_coord_dropout = float(val_cfg.get("aggregate_coord_dropout", 0.0) or 0.0)
            if not (0.0 <= self.val_aggregate_coord_dropout <= 1.0):
                raise ValueError("training.validation.aggregate_coord_dropout must be in [0, 1]")
        else:
            self.val_aggregate_coord_dropout = 0.0

    def _poi_mask_index_tensor(self, *, device: torch.device) -> Optional[torch.Tensor]:
        return poi_mask_index_tensor(self, device=device)

    def _apply_poi_token_mask_to_probs(
        self,
        poi_probs: torch.Tensor,
        *,
        stats: Optional[Dict[str, float]] = None,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        return apply_poi_token_mask_to_probs(self, poi_probs, stats=stats, epsilon=epsilon)

    def _apply_poi_token_mask_to_dist(
        self,
        dist: torch.Tensor,
        *,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        return apply_poi_token_mask_to_dist(self, dist, epsilon=epsilon)

    def _transition_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return transition_mask(attention_mask)

    def _category_transition_probs(self, cat_probs: torch.Tensor) -> torch.Tensor:
        return category_transition_probs(cat_probs)

    def _aggregate_feature_losses(
        self,
        *,
        cbg: str,
        poi_probs: torch.Tensor,
        attention_mask: torch.Tensor,
        age_raw: torch.Tensor,
        gender_raw: torch.Tensor,
        poi_store: Optional[POIMarginalStore] = None,
        cache: Optional[CBGConditionCache] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return aggregate_feature_losses(
            self,
            cbg=cbg,
            poi_probs=poi_probs,
            attention_mask=attention_mask,
            age_raw=age_raw,
            gender_raw=gender_raw,
            poi_store=poi_store,
            cache=cache,
        )

    def _entropy_regularizer(
        self,
        poi_probs: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a simple entropy-based regularizer over per-trajectory POI histograms.
        Encourages trajectories to visit a more diverse set of POIs, without touching
        the aggregate marginals.

        Returns an unscaled scalar loss term (-mean_entropy). The caller is
        responsible for multiplying by self.lambda_entropy.
        """
        if not self.use_entropy_reg or self.lambda_entropy <= 0.0:
            # Return a detached zero so callers can safely add it.
            return poi_probs.new_zeros((), requires_grad=False)

        # poi_probs: [B, T, V], attn_mask: [B, T]
        mask = attn_mask.unsqueeze(-1).to(dtype=poi_probs.dtype)  # [B, T, 1]
        masked_probs = poi_probs * mask
        token_counts = mask.sum(dim=1).clamp_min(1.0)             # [B, 1]

        # Per-trajectory POI histogram over time: [B, V]
        hist = masked_probs.sum(dim=1) / token_counts
        hist_sum = hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hist = hist / hist_sum

        entropy = -(hist * (hist + 1e-8).log()).sum(dim=-1)       # [B]
        mean_entropy = entropy.mean()

        # Loss = -H (we want higher entropy → lower loss). Caller will scale.
        return -mean_entropy

    def _unique_regularizer(
        self,
        poi_probs: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expected unique-POI regularizer (Approach A).
        For each trajectory, approximate the expected number of distinct POIs visited:
            E[unique] = sum_k (1 - prod_t (1 - p_{t,k}))
        where p_{t,k} is the probability of POI k at timestep t (masked by attn_mask).

        Returns an unscaled scalar loss term (-mean_expected_unique). The caller is
        responsible for multiplying by self.lambda_unique.
        """
        if not self.use_unique_reg or self.lambda_unique <= 0.0:
            return poi_probs.new_zeros((), requires_grad=False)

        # poi_probs: [B, T, V], attn_mask: [B, T]
        mask = attn_mask.unsqueeze(-1).to(dtype=poi_probs.dtype)  # [B, T, 1]
        p = poi_probs * mask

        # Clamp for numerical stability: avoid log(<=0).
        p_clamped = p.clamp_max(1.0 - 1e-6)
        log_no_visit = torch.log1p(-p_clamped)                    # log(1 - p)
        # Sum over time, then exponentiate: prod_t (1 - p_{t,k})
        log_no_visit_sum = log_no_visit.sum(dim=1)                # [B, V]
        no_visit = torch.exp(log_no_visit_sum)
        visited = 1.0 - no_visit                                  # [B, V]
        expected_unique = visited.sum(dim=-1)                     # [B]
        mean_unique = expected_unique.mean()

        # Loss = -E[unique] (we want more unique POIs → lower loss). Caller will scale.
        return -mean_unique

    def _load_length_distributions(self, json_path: str) -> None:
        """
        Load per-CBG trajectory length distributions from a JSON file produced by build_length_dists.py.
        Mirrors the logic in population_pipeline.step3_sample_lengths_from_json.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            dists = json.load(f)
        if not isinstance(dists, dict) or not dists:
            raise ValueError(f"Empty or invalid length distributions JSON: {json_path}")

        self.length_dists = {}
        cbg_keys = list(dists.keys())
        probs_stack: List[np.ndarray] = []
        counts: List[float] = []
        for cbg in cbg_keys:
            entry = dists[cbg]
            probs = np.asarray(entry["probs"], dtype=float)
            if probs.ndim != 1:
                raise ValueError(f"Length probs for CBG {cbg} must be 1D.")
            # Normalize per-CBG to guard against numeric drift
            s = probs.sum()
            if s <= 0:
                probs = np.ones_like(probs, dtype=float) / max(probs.size, 1)
            else:
                probs = probs / s
            self.length_dists[cbg] = probs
            probs_stack.append(probs)
            counts.append(float(entry.get("count", 0.0)))

        probs_stack_arr = np.vstack(probs_stack)
        counts_arr = np.asarray(counts, dtype=float)
        weight_sum = counts_arr.sum()
        if weight_sum <= 0:
            city_probs = probs_stack_arr.mean(axis=0)
        else:
            city_probs = (probs_stack_arr * counts_arr[:, None]).sum(axis=0) / weight_sum
        # Final normalize
        s = city_probs.sum()
        if s <= 0:
            city_probs = np.ones_like(city_probs, dtype=float) / max(city_probs.size, 1)
        else:
            city_probs = city_probs / s

        self.length_city_probs = city_probs

        # Align max_traj_length with JSON shape (bins-1), as in step3_sample_lengths_from_json.
        num_bins = city_probs.shape[0]
        max_idx = num_bins - 1
        if self.length_max != max_idx:
            self.length_max = max_idx

    def _sample_lengths_for_cbg(self, cbg: str, batch_size: int) -> torch.Tensor:
        """
        Sample discrete trajectory lengths for a given CBG using its distribution,
        falling back to a city-wide average when needed.
        """
        if self.length_dists is None or self.length_city_probs is None:
            # Fallback: uniform over [1, seq_len]
            lengths = torch.randint(
                low=1,
                high=self.seq_len + 1,
                size=(batch_size,),
                device=self.device,
                dtype=torch.long,
            )
            return lengths

        probs = self.length_dists.get(cbg, self.length_city_probs)
        probs = np.asarray(probs, dtype=float)
        s = probs.sum()
        if s <= 0:
            probs = np.ones_like(probs, dtype=float) / max(probs.size, 1)
        else:
            probs = probs / s

        idx = np.random.choice(len(probs), size=batch_size, p=probs).astype(np.int64)
        # Mirror step3: clamp to [min_traj_len, max_traj_length]
        idx = np.clip(idx, self.length_min, self.length_max)
        lengths = torch.from_numpy(idx).to(self.device, dtype=torch.long)
        # Ensure lengths do not exceed seq_len used in the decoder.
        lengths = lengths.clamp_(1, self.seq_len)
        return lengths

    def _sample_length_mask(self, cbg: str, batch_size: int) -> torch.Tensor:
        """
        Sample a per-sample sequence length (using per-CBG distributions when available)
        and build an attention mask for the aggregate branch.
        """
        lengths = self._sample_lengths_for_cbg(cbg, batch_size)
        positions = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        mask = (positions < lengths.unsqueeze(1)).long()
        return mask

    def sample_cbg(self) -> str:
        return random.choice(self.cbgs)

    def _normalize_dist(self, x: torch.Tensor, *, epsilon: float) -> torch.Tensor:
        return normalize_dist(x, epsilon=epsilon)

    def _distribution_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        *,
        loss_type: str,
        epsilon: float,
    ) -> torch.Tensor:
        return distribution_loss(target, pred, loss_type=loss_type, epsilon=epsilon)

    def _sample_demo_ids_from_pi(
        self,
        cbg: str,
        batch_size: int,
        *,
        cache: Optional[CBGConditionCache] = None,
        epsilon: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Sample raw (0-based) demo ids (age_bin, gender_id) from π_cbg.

        If training.llp.min_per_group > 0, enforce >=K samples per group by construction
        (requires batch_size >= K * (num_age_bins * num_genders)).
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

        K = int(getattr(self, "llp_min_per_group", 0) or 0)
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
            # Shuffle so groups are not in contiguous blocks (helps avoid any accidental ordering effects).
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

    def _llp_mixture_kl(
        self,
        poi_probs: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        cbg: str,
        batch_age: torch.Tensor,
        batch_gender: torch.Tensor,
        cache: Optional[CBGConditionCache] = None,
        poi_store: Optional[POIMarginalStore] = None,
        target_dist: Optional[torch.Tensor] = None,
        apply_poi_token_mask: bool = True,
        epsilon: float = 1e-8,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        LLP mixture KL:
          1) Build per-group POI distributions P_hat[d] from current batch predictions.
          2) Mix them with π_cbg,d to obtain P_mix.
          3) Compute KL( P_target_cbg || P_mix ).
        """
        src_cache = cache or self.cache
        src_store = poi_store or self.poi_store

        # Shapes
        B, T, V = poi_probs.shape
        device = poi_probs.device
        # Require configured demo grouping
        D_age = max(int(self.num_age_bins), 1)
        D_gen = max(int(self.num_genders), 1)
        D = D_age * D_gen

        # Group indices d = age * num_genders + gender  (raw ids are 0-based)
        group_idx = (batch_age.long().clamp_min(0) * D_gen + batch_gender.long().clamp_min(0)).clamp(0, D - 1)
        one_hot = torch.nn.functional.one_hot(group_idx, num_classes=D).to(dtype=poi_probs.dtype, device=device)  # [B, D]

        # Broadcast weights over time and vocab
        # bt_weights: [B, T, D]
        bt_weights = attention_mask.to(dtype=poi_probs.dtype, device=device).unsqueeze(-1) * one_hot.unsqueeze(1)
        # Weighted token mass per group: [D, V]
        group_mass = (poi_probs.unsqueeze(2) * bt_weights.unsqueeze(-1)).sum(dim=(0, 1))
        # Token counts per group: [D]
        group_counts = bt_weights.sum(dim=(0, 1)).clamp_min(1.0)
        P_hat = group_mass / group_counts.unsqueeze(-1)

        # Fallback for empty groups: replace with overall batch marginal
        overall = aggregate_poi_distribution(poi_probs, attention_mask=attention_mask, epsilon=epsilon)  # [V]
        empty_groups = (group_counts <= 1.0)
        if empty_groups.any():
            P_hat[empty_groups] = overall.unsqueeze(0).expand(int(empty_groups.sum().item()), V)

        # π vector for this CBG: [D]
        pi_cbg = src_cache.as_torch_distribution(
            cbg,
            num_age_bins=D_age,
            num_genders=D_gen,
            device=device,
        ).to(dtype=poi_probs.dtype)
        pi_cbg = (pi_cbg + epsilon) / (pi_cbg.sum() + epsilon * pi_cbg.numel())

        # Mixture distribution: [V]
        mix = (pi_cbg.unsqueeze(-1) * P_hat).sum(dim=0)
        mix = (mix + epsilon) / (mix.sum() + epsilon * mix.numel())
        if apply_poi_token_mask:
            mix = self._apply_poi_token_mask_to_dist(mix, epsilon=epsilon)

        # Target P_cbg (can be injected for non-POI feature spaces)
        if target_dist is None:
            target = src_store.get_distribution(cbg, device=device).to(dtype=mix.dtype)
        else:
            target = target_dist.to(device=device, dtype=mix.dtype)
        target = (target + epsilon) / (target.sum() + epsilon * target.numel())
        if apply_poi_token_mask:
            target = self._apply_poi_token_mask_to_dist(target, epsilon=epsilon)

        loss_type = str(getattr(self, "aggregate_loss_type", "kl"))
        loss = self._distribution_loss(target, mix, loss_type=loss_type, epsilon=epsilon)
        stats_key = f"agg_{loss_type}"
        stats = {stats_key: float(loss.item())}
        if loss_type == "kl":
            stats["agg_kl"] = float(loss.item())
        return loss, stats

    def _aggregate_loss_for(
        self,
        *,
        cbg: str,
        cache: CBGConditionCache,
        poi_store: POIMarginalStore,
        batch_size: int,
        num_special_tokens: int,
        llp_demo_source: Optional[str] = None,
        coord_dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute aggregate loss (plain KL or LLP mixture KL) for one sampled batch.
        This is used for validation, and can also be used for training.
        """
        batch = cache.sample(cbg, batch_size, device=self.device)
        stats: Dict[str, float] = {"cbg": cbg}

        # Decide which demo ids to use for conditioning + LLP grouping.
        # Raw ids are 0-based; embeddings expect 1-based (0 reserved for null/pad).
        demo_source = (llp_demo_source or self.llp_demo_source).lower().strip()
        if demo_source not in {"cache", "pi"}:
            raise ValueError(f"llp_demo_source must be 'cache' or 'pi' (got {demo_source!r})")

        if self.llp_enabled and demo_source == "pi" and self.num_age_bins > 0 and self.num_genders > 0:
            age_raw, gender_raw, llp_demo_stats = self._sample_demo_ids_from_pi(cbg, batch_size, cache=cache)
            stats.update(llp_demo_stats)
        else:
            age_raw = batch.age_bin.to(device=self.device)
            gender_raw = batch.gender_id.to(device=self.device)
            stats["llp_demo_source_pi"] = 0.0

        # Build conditioning attributes for DiT.
        attrs_parts = [batch.work, batch.home]
        base_dit = getattr(self.dit, "module", self.dit)
        if getattr(base_dit.attr_embed, "use_demo_condition", False):
            age_f = (age_raw.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            gender_f = (gender_raw.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            attrs_parts.extend([age_f, gender_f])
        attrs = torch.cat(attrs_parts, dim=-1)
        # Optional: drop coords for some samples (demo remains, if present).
        p = float(coord_dropout_p or 0.0)
        if p > 0.0:
            if p < 0.0 or p > 1.0:
                raise ValueError("coord_dropout_p must be in [0, 1]")
            mask = torch.rand(attrs.size(0), device=attrs.device) < p
            if mask.any():
                attrs = attrs.clone()
                attrs[mask, :4] = 0.0
                stats["coord_dropout_p"] = float(p)
                stats["coord_dropout_n"] = float(mask.sum().item())

        if not self.simple_x0:
            latents = sample_latents(
                diffusion=self.scheduler,
                dit=self.dit,
                cond=attrs,
                seq_len=self.seq_len,
                latent_dim=self.latent_dim,
                prediction_type=self.prediction_type,
                guidance_scale=self.guidance_scale,
                sampler=self.aggregate_sampler,
                ddim_steps=self.aggregate_ddim_steps,
                ddim_eta=self.aggregate_ddim_eta,
                device=self.device,
            )
            decode_latents = self.latent_pca.unproject(latents) if self.latent_pca is not None else latents
            encoder_outputs = BaseModelOutput(last_hidden_state=decode_latents)
            agg_attn_mask = self._sample_length_mask(cbg, latents.size(0))
            bos_id = self.autoencoder.config.decoder_start_token_id or self.autoencoder.config.bos_token_id
            if bos_id is None:
                raise ValueError("Autoencoder config must define bos_token_id or decoder_start_token_id.")
            decoder_input_ids = torch.full(
                (latents.size(0), self.seq_len),
                bos_id,
                device=self.device,
                dtype=torch.long,
            )
            decoder_out = self.autoencoder(
                encoder_outputs=encoder_outputs,
                attention_mask=agg_attn_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = decoder_out.logits
            if num_special_tokens > 0:
                logits[..., :num_special_tokens] = logits[..., :num_special_tokens] - 1e4
            poi_probs = torch.softmax(logits, dim=-1)
            if num_special_tokens > 0:
                poi_probs = poi_probs[..., num_special_tokens:]
            poi_probs = self._apply_poi_token_mask_to_probs(poi_probs, stats=stats, epsilon=self.aggregate_loss_eps)

            agg_loss, feat_stats = self._aggregate_feature_losses(
                cbg=cbg,
                poi_probs=poi_probs,
                attention_mask=agg_attn_mask,
                age_raw=age_raw,
                gender_raw=gender_raw,
                poi_store=poi_store,
                cache=cache,
            )
            stats.update(feat_stats)
            ent_loss = self._entropy_regularizer(poi_probs, agg_attn_mask)
            uniq_loss = self._unique_regularizer(poi_probs, agg_attn_mask)
        else:
            timesteps = self.scheduler.num_timesteps
            K = min(max(1, self.simple_x0_last_k), timesteps)
            latents = torch.randn(attrs.size(0), self.seq_len, self.latent_dim, device=self.device)
            for idx in range(timesteps - 1, K - 1, -1):
                t = torch.full((latents.size(0),), idx, device=self.device, dtype=torch.long)
                model_out = GaussianDiffusion.classifier_free_guidance(
                    denoiser=self.dit,
                    x_t=latents,
                    t=t,
                    conditional_attrs=attrs,
                    guidance_scale=self.guidance_scale,
                )
                preds = self.scheduler.model_predictions(model_out, latents, t, prediction_type=self.prediction_type)
                latents = _posterior_sample(self.scheduler, preds.x_start, latents, t)

            predicted_x0_list: List[torch.Tensor] = []
            x_t = latents
            for idx in range(K - 1, -1, -1):
                t = torch.full((x_t.size(0),), idx, device=self.device, dtype=torch.long)
                model_out = GaussianDiffusion.classifier_free_guidance(
                    denoiser=self.dit,
                    x_t=x_t,
                    t=t,
                    conditional_attrs=attrs,
                    guidance_scale=self.guidance_scale,
                )
                preds = self.scheduler.model_predictions(model_out, x_t, t, prediction_type=self.prediction_type)
                predicted_x0_list.append(preds.x_start)
                if idx > 0:
                    posterior_mean, _, posterior_log_var = self.scheduler.q_posterior(x_start=preds.x_start, x_t=x_t, t=t)
                    noise = torch.randn_like(x_t)
                    x_t = posterior_mean + torch.exp(0.5 * posterior_log_var) * noise

            agg_attn_mask = self._sample_length_mask(cbg, attrs.size(0))
            bos_id = self.autoencoder.config.decoder_start_token_id or self.autoencoder.config.bos_token_id
            if bos_id is None:
                raise ValueError("Autoencoder config must define bos_token_id or decoder_start_token_id.")
            decoder_input_ids = torch.full((attrs.size(0), self.seq_len), bos_id, device=self.device, dtype=torch.long)

            agg_losses: List[torch.Tensor] = []
            ent_losses: List[torch.Tensor] = []
            uniq_losses: List[torch.Tensor] = []
            for x0 in predicted_x0_list:
                decode_latents = self.latent_pca.unproject(x0) if self.latent_pca is not None else x0
                encoder_outputs = BaseModelOutput(last_hidden_state=decode_latents)
                decoder_out = self.autoencoder(
                    encoder_outputs=encoder_outputs,
                    attention_mask=agg_attn_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = decoder_out.logits
                if num_special_tokens > 0:
                    logits[..., :num_special_tokens] = logits[..., :num_special_tokens] - 1e4
                poi_probs = torch.softmax(logits, dim=-1)
                if num_special_tokens > 0:
                    poi_probs = poi_probs[..., num_special_tokens:]
                poi_probs = self._apply_poi_token_mask_to_probs(poi_probs, stats=stats, epsilon=self.aggregate_loss_eps)

                loss_k, _ = self._aggregate_feature_losses(
                    cbg=cbg,
                    poi_probs=poi_probs,
                    attention_mask=agg_attn_mask,
                    age_raw=age_raw,
                    gender_raw=gender_raw,
                    poi_store=poi_store,
                    cache=cache,
                )
                agg_losses.append(loss_k)
                ent_losses.append(self._entropy_regularizer(poi_probs, agg_attn_mask))
                uniq_losses.append(self._unique_regularizer(poi_probs, agg_attn_mask))

            agg_loss = torch.stack(agg_losses).mean()
            ent_loss = torch.stack(ent_losses).mean() if ent_losses else agg_loss.new_zeros((), requires_grad=False)
            uniq_loss = torch.stack(uniq_losses).mean() if uniq_losses else agg_loss.new_zeros((), requires_grad=False)

        total_loss = self.lambda_agg * agg_loss
        stats["agg_loss"] = float(agg_loss.detach().item())

        if self.use_entropy_reg and self.lambda_entropy > 0.0 and ent_loss is not None:
            total_loss = total_loss + self.lambda_entropy * ent_loss
            stats["entropy_loss"] = float(ent_loss.detach().item())
        if self.use_unique_reg and self.lambda_unique > 0.0 and uniq_loss is not None:
            total_loss = total_loss + self.lambda_unique * uniq_loss
            stats["unique_loss"] = float(uniq_loss.detach().item())

        return total_loss, stats

    def validate(self, accelerator: Accelerator) -> Dict[str, float]:
        return run_cbg_validate(self, accelerator)

    def training_step(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        return run_cbg_training_step(self)

    def save_checkpoint(self, global_step: int, accelerator: Optional[Accelerator] = None) -> None:
        if accelerator is not None:
            model = accelerator.unwrap_model(self.dit)
        else:
            model = self.dit
        payload = {
            "dit": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": global_step,
        }
        ckpt_path = self.output_dir / f"cbg_finetune_step_{global_step}.pt"
        torch.save(payload, ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")

    def run(self, accelerator: Accelerator) -> None:
        train_cfg = self.config["training"]
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))

        print(f"[INFO] Starting CBG-conditioned fine-tuning for {self.steps} optimizer steps "
              f"(grad_accumulation={grad_accum}).")
        global_step = self.start_step
        self.optimizer.zero_grad()

        for step in range(1, self.steps + 1):
            step_loss = 0.0
            last_stats: Optional[Dict[str, float]] = None

            for _ in range(grad_accum):
                loss, stats = self.training_step()
                last_stats = stats
                loss = loss / grad_accum
                accelerator.backward(loss)
                step_loss += float(loss.detach().item())

            accelerator.clip_grad_norm_(self.dit.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process and (global_step % self.log_every == 0) and last_stats is not None:
                agg_val = last_stats.get("agg_loss", 0.0)
                cbg = last_stats.get("cbg", "N/A")
                print(f"[step {global_step:05d}] loss={step_loss:.4f} agg={agg_val:.4f} cbg={cbg}")

                # Log to Weights & Biases on the main process if enabled.
                if self.use_wandb and wandb.run is not None:
                    log_payload = {
                        "loss": step_loss,
                        "agg_loss": agg_val,
                        "cbg": cbg,
                        "step": global_step,
                    }
                    if "mse_loss" in last_stats:
                        log_payload["mse_loss"] = last_stats["mse_loss"]
                    if "entropy_loss" in last_stats:
                        log_payload["entropy_loss"] = last_stats["entropy_loss"]
                    if "unique_loss" in last_stats:
                        log_payload["unique_loss"] = last_stats["unique_loss"]
                    # Log current learning rate if available.
                    if len(self.optimizer.param_groups) > 0 and "lr" in self.optimizer.param_groups[0]:
                        log_payload["lr"] = self.optimizer.param_groups[0]["lr"]
                    # Optionally include per-CBG aggregate loss as its own metric key.
                    if getattr(self, "log_agg_by_cbg", False) and isinstance(cbg, str) and cbg:
                        log_payload[f"agg_loss_by_cbg/{cbg}"] = agg_val
                    wandb.log(log_payload, step=global_step)

            # Optional validation (no early stopping)
            if self.val_enabled and self.val_every > 0 and (global_step % self.val_every == 0):
                val_metrics = self.validate(accelerator)
                if accelerator.is_main_process and val_metrics:
                    msg = " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                    print(f"[val {global_step:05d}] {msg}")
                    if self.use_wandb and wandb.run is not None:
                        wandb.log(val_metrics, step=global_step)

            if accelerator.is_main_process and (global_step % self.save_every == 0):
                self.save_checkpoint(global_step, accelerator)

        if accelerator.is_main_process:
            self.save_checkpoint(global_step, accelerator)
            print("[INFO] Training finished.")
