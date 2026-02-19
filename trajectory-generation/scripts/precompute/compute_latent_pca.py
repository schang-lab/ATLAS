#!/usr/bin/env python3
"""Compute PCA artifacts for autoencoder latents.

This utility mirrors the preprocessing performed in latent_length_tests E4 by:
  * encoding trajectories with the Phase 1/2 autoencoder
  * computing token-level mean and variance across the latent channels
  * sampling standardized latents and fitting a PCA basis (optionally with whitening)

The resulting tensors (mean, std, PCA components, variance statistics) are
stored in a Torch checkpoint for later reuse during DiT training.

Example (change to your own folder paths):
CUDA_VISIBLE_DEVICES=0 python compute_latent_pca.py \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --output_path /path/to/output/pca32/train.pt \
  --data_dir /path/to/YOUR_DATA_FOLDER \
  --data_type controlled,uncontrolled \
  --split train \
  --training_phase phase1 \
  --pca_components 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Repo path setup (allow launching from anywhere)
# --------------------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
TRAJGEN_ROOT = SCRIPT_PATH.parents[2]

if str(TRAJGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAJGEN_ROOT))

AUTOENCODER_ROOT = REPO_ROOT / "autoencoder"
if AUTOENCODER_ROOT.exists() and str(AUTOENCODER_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOENCODER_ROOT))

# Provide a minimal stub for timm if the dependency is absent (only needed for DiT definitions).
try:  # pragma: no cover - executed only when timm missing
    import timm  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    LOG_STUB = logging.getLogger("latent_pca.timm_stub")
    LOG_STUB.info("timm not available; injecting lightweight stubs for dataset import")

    import importlib.machinery

    timm_module = types.ModuleType("timm")
    models_module = types.ModuleType("timm.models")
    vt_module = types.ModuleType("timm.models.vision_transformer")

    timm_module.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None)
    models_module.__spec__ = importlib.machinery.ModuleSpec("timm.models", loader=None)
    vt_module.__spec__ = importlib.machinery.ModuleSpec("timm.models.vision_transformer", loader=None)

    class _Placeholder:  # Minimal callable used only for type checking during import
        def __init__(self, *args, **kwargs):
            pass

    vt_module.PatchEmbed = _Placeholder
    vt_module.Attention = _Placeholder
    vt_module.Mlp = _Placeholder

    models_module.vision_transformer = vt_module
    timm_module.models = models_module

    sys.modules["timm"] = timm_module
    sys.modules["timm.models"] = models_module
    sys.modules["timm.models.vision_transformer"] = vt_module

# Now safe to import project modules
from transformers import BartConfig, BartForConditionalGeneration, BertTokenizerFast  # noqa: E402

from auto_encoder.traj_compressed_ae import BARTLatentCompression  # noqa: E402


LOG = logging.getLogger("latent_pca")


# --------------------------------------------------------------------------------------
# Helper containers
# --------------------------------------------------------------------------------------


def parse_sequence(seq: object) -> List[str]:
    if isinstance(seq, str):
        return seq.strip().split()
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [str(x) for x in seq]
    raise TypeError(f"Unsupported sequence type: {type(seq)!r}")


def load_tokenizer(train_dir: Path) -> BertTokenizerFast:
    LOG.info("Loading tokenizer vocabulary from %s", train_dir)
    vocab_path = train_dir / "tokenizer_vocab.pkl"
    vocab: List[str]
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab_txt = train_dir / "tokenizer" / "vocab.txt"
        if not vocab_txt.exists():
            raise FileNotFoundError(
                f"Tokenizer vocabulary not found (expected {vocab_path} or {vocab_txt})"
            )
        vocab = []
        with open(vocab_txt, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    vocab.append(token)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for token in vocab:
            tmp.write(f"{token}\n")
        temp_vocab_file = tmp.name

    try:
        tokenizer = BertTokenizerFast(vocab_file=temp_vocab_file, do_lower_case=False)
        tokenizer.add_special_tokens(
            {
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
            }
        )
    finally:
        os.unlink(temp_vocab_file)

    LOG.info("Tokenizer built with vocab size %d", len(tokenizer))
    return tokenizer


def build_segment_maps(segment_df: pd.DataFrame, ablation_mode: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[int, int]]]:
    lat_min, lat_max = segment_df["lat"].min(), segment_df["lat"].max()
    lon_min, lon_max = segment_df["lon"].min(), segment_df["lon"].max()

    segment_df["norm_lat"] = 2 * (segment_df["lat"] - lat_min) / (lat_max - lat_min) - 1
    segment_df["norm_lon"] = 2 * (segment_df["lon"] - lon_min) / (lon_max - lon_min) - 1

    use_categories = ablation_mode in {"subcat_only", "both"}
    if use_categories:
        unique_top = segment_df["top_category"].dropna().unique()
        unique_sub = segment_df["sub_category"].dropna().unique()
        top_category_to_id = {cat: i for i, cat in enumerate(unique_top)}
        sub_category_to_id = {cat: i for i, cat in enumerate(unique_sub)}
    else:
        top_category_to_id = {}
        sub_category_to_id = {}

    coord_map: Dict[str, Tuple[float, float]] = {}
    category_map: Dict[str, Tuple[int, int]] = {}

    for row in segment_df.itertuples():
        poi = str(row.poi_id)
        coord_map[poi] = (float(row.norm_lat), float(row.norm_lon))

        if use_categories:
            top_id = top_category_to_id.get(row.top_category, 0) if pd.notna(row.top_category) else 0
            sub_id = sub_category_to_id.get(row.sub_category, 0) if pd.notna(row.sub_category) else 0
        else:
            top_id = 0
            sub_id = 0
        category_map[poi] = (int(top_id), int(sub_id))

    return coord_map, category_map


class LatentDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[str]],
        attention_masks: List[List[int]],
        tokenizer: BertTokenizerFast,
        coord_map: Dict[str, Tuple[float, float]],
        category_map: Dict[str, Tuple[int, int]],
        training_phase: str,
        ablation_mode: str,
        max_length: int = 512,
    ) -> None:
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.tokenizer = tokenizer
        self.coord_map = coord_map
        self.category_map = category_map
        self.training_phase = training_phase
        self.ablation_mode = ablation_mode
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq = list(self.sequences[idx])
        attn = list(self.attention_masks[idx])

        if len(seq) > self.max_length:
            seq = seq[: self.max_length]
            attn = attn[: self.max_length]

        token_ids: List[int] = []
        lat: List[float] = []
        lon: List[float] = []
        sub_categories: List[int] = []

        for token in seq:
            token_str = str(token)
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None:
                token_id = self.tokenizer.unk_token_id
            token_ids.append(token_id)

            coord = self.coord_map.get(token_str, (0.0, 0.0))
            lat.append(coord[0])
            lon.append(coord[1])

            cat = self.category_map.get(token_str, (0, 0))
            sub_categories.append(cat[1])

        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(self.tokenizer.pad_token_id)
            lat.append(0.0)
            lon.append(0.0)
            sub_categories.append(0)
            attn.append(0)

        attention_mask = attn[: self.max_length]

        item: Dict[str, Tensor] = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.training_phase == "phase2" and self.ablation_mode in {"coords_only", "both"}:
            item["lat"] = torch.tensor(lat, dtype=torch.float32)
            item["lon"] = torch.tensor(lon, dtype=torch.float32)

        if self.training_phase == "phase2" and self.ablation_mode in {"subcat_only", "both"}:
            item["sub_categories"] = torch.tensor(sub_categories, dtype=torch.long)

        return item


@dataclass
class PCAArtifacts:
    """Container for the PCA tensors and metadata we persist to disk."""

    mean: Tensor
    std: Tensor
    components: Tensor
    explained_variance: Tensor
    explained_variance_ratio: Tensor
    whitening: bool
    latent_dim: int
    component_dim: int
    token_count: int
    sample_count: int
    metadata: Dict[str, object]

    def to_serializable(self) -> Dict[str, object]:
        return {
            "mean": self.mean.cpu(),
            "std": self.std.cpu(),
            "components": self.components.cpu(),
            "explained_variance": self.explained_variance.cpu(),
            "explained_variance_ratio": self.explained_variance_ratio.cpu(),
            "whitening": self.whitening,
            "latent_dim": self.latent_dim,
            "component_dim": self.component_dim,
            "token_count": self.token_count,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCA artifacts for AE latents")

    parser.add_argument("--autoencoder_path", type=str, required=True,
                        help="Directory containing the trained autoencoder checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the PCA artifact (.pt)")
    parser.add_argument("--data_dir", type=str, default="split_data_new",
                        help="Root directory containing domain subfolders")
    parser.add_argument("--data_type", type=str, default="controlled,uncontrolled",
                        help="Comma-separated domains to include (e.g. 'controlled,uncontrolled' or 'unified')")
    parser.add_argument("--split", type=str, default="train",
                        help="Comma-separated data splits to include (e.g. 'train' or 'train,val')")
    parser.add_argument("--training_phase", type=str, choices=["phase1", "phase2"],
                        default="phase1", help="Autoencoder training phase")
    parser.add_argument("--ablation_mode", type=str,
                        choices=["coords_only", "subcat_only", "both", "neither", "pure"],
                        default="both", help="Feature configuration to mirror during encoding")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for latent extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length used for encoding (will be capped to the autoencoder's max_position_embeddings if smaller)",
    )
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Optional limit on number of sequences to process")
    parser.add_argument("--token_cap", type=int, default=500_000,
                        help="Maximum number of tokens sampled for PCA (reservoir sampling)")
    parser.add_argument("--pca_components", type=int, default=64,
                        help="Maximum number of PCA components to keep")
    parser.add_argument("--variance_threshold", type=float, default=None,
                        help="Optional target cumulative variance (0-1). Overrides n_components when set")
    parser.add_argument("--whiten", action="store_true", default=True,
                        help="Enable PCA whitening (recommended)")
    parser.add_argument("--no_whiten", dest="whiten", action="store_false",
                        help="Disable PCA whitening")
    parser.add_argument("--skip_pca", action="store_true", default=False,
                        help="Skip PCA rotation and only record mean/std (identity components)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (defaults to cuda if available)")

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Core utilities
# --------------------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def resolve_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_autoencoder(args: argparse.Namespace, device: torch.device) -> BARTLatentCompression:
    ae_path = Path(args.autoencoder_path)
    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder path not found: {ae_path}")

    if args.training_phase == "phase1":
        # For phase1, we expect a plain BART checkpoint directory
        autoencoder = BartForConditionalGeneration.from_pretrained(str(ae_path))
        autoencoder.to(device)
        autoencoder.eval()
        LOG.info("Loaded phase1 BART autoencoder from %s", ae_path)
        # Wrap to provide uniform interface in extract_latents
        return autoencoder

    config_path = ae_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing BART config at {config_path}")

    ae_config = BartConfig.from_json_file(str(config_path))

    args_path = ae_path / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json alongside autoencoder at {args_path}")

    with open(args_path, "r", encoding="utf-8") as f:
        ae_args = json.load(f)

    use_coords = ae_args.get("use_coords", False) and args.ablation_mode in {"coords_only", "both"}
    use_subcategories = ae_args.get("num_sub_categories") is not None and args.ablation_mode in {"subcat_only", "both"}

    autoencoder = BARTLatentCompression.from_pretrained(
        str(ae_path),
        config=ae_config,
        num_encoder_latents=ae_args.get("num_encoder_latents"),
        num_decoder_latents=ae_args.get("num_decoder_latents"),
        dim_ae=ae_args.get("dim_ae"),
        num_layers=ae_args.get("num_layers", 2),
        l2_normalize_latents=ae_args.get("l2_normalize_latents", True),
        use_coords=use_coords,
        num_sub_categories=ae_args.get("num_sub_categories") if use_subcategories else None,
        use_position_embedding=ae_args.get("use_position_embedding", True),
        transformer_decoder=ae_args.get("transformer_decoder", False),
        no_compression=ae_args.get("no_compression", False),
    )

    autoencoder.to(device)
    autoencoder.eval()
    LOG.info("Loaded autoencoder from %s (dim=%d)", ae_path, ae_args.get("dim_ae"))
    return autoencoder


def build_dataloader(args: argparse.Namespace, *, max_length: int) -> DataLoader:
    domain_names = [d.strip() for d in args.data_type.split(",") if d.strip()]
    if not domain_names:
        raise ValueError("No data domains provided via --data_type")

    split_names = [s.strip() for s in args.split.split(",") if s.strip()]
    if not split_names:
        raise ValueError("No splits provided via --split")

    coord_map: Dict[str, Tuple[float, float]] = {}
    category_map: Dict[str, Tuple[int, int]] = {}
    sequences: List[List[str]] = []
    attention_masks: List[List[int]] = []
    tokenizer: Optional[BertTokenizerFast] = None

    for domain in domain_names:
        for split in split_names:
            split_dir = Path(args.data_dir) / domain / split
            if not split_dir.exists():
                LOG.warning("Skipping missing split directory: %s", split_dir)
                continue

            LOG.info("Loading %s/%s data from %s", domain, split, split_dir)

            segment_df = pd.read_csv(split_dir / "poi_map_feature.csv")
            seg_coord_map, seg_category_map = build_segment_maps(segment_df, args.ablation_mode)
            coord_map.update(seg_coord_map)
            category_map.update(seg_category_map)

            if tokenizer is None:
                tokenizer = load_tokenizer(split_dir)

            traj_path = split_dir / "final_segments_all_train_data.pkl"
            if not traj_path.exists():
                raise FileNotFoundError(f"Expected trajectory file not found: {traj_path}")

            traj_df = pd.read_pickle(traj_path)
            for row in traj_df.itertuples():
                sequences.append(parse_sequence(row.unique_id_seq))
                attention_masks.append(list(row.attention_mask))

    if tokenizer is None:
        raise RuntimeError("Tokenizer could not be constructed; no valid splits were loaded")

    if args.max_sequences is not None and args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
        attention_masks = attention_masks[: args.max_sequences]

    dataset = LatentDataset(
        sequences,
        attention_masks,
        tokenizer,
        coord_map,
        category_map,
        training_phase=args.training_phase,
        ablation_mode=args.ablation_mode,
        max_length=max_length,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    LOG.info("Prepared dataset with %d samples from domains=%s splits=%s", len(dataset), domain_names, split_names)
    return loader


def _maybe_stack_coords(batch: Dict[str, Tensor], device: torch.device) -> Optional[Tensor]:
    if "lat" in batch and "lon" in batch:
        return torch.stack([batch["lat"].to(device), batch["lon"].to(device)], dim=-1)
    return None


def extract_latents(
    batch: Dict[str, Tensor],
    autoencoder: BARTLatentCompression,
    training_phase: str,
    ablation_mode: str,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        if training_phase == "phase1":
            encoder_outputs = autoencoder.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            latents = encoder_outputs.last_hidden_state
        else:
            encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            segment_coords = None
            sub_categories = None

            if ablation_mode in {"coords_only", "both"}:
                segment_coords = _maybe_stack_coords(batch, device)

            if ablation_mode in {"subcat_only", "both"} and "sub_categories" in batch:
                sub_categories = batch["sub_categories"].to(device)

            if getattr(autoencoder, "no_compression", False):
                enhanced = autoencoder._add_features_no_compression(
                    encoder_outputs, attention_mask, segment_coords, sub_categories
                )
                latents = enhanced["last_hidden_state"]
            else:
                latents = autoencoder.get_diffusion_latent(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    segment_coords=segment_coords,
                    sub_categories=sub_categories,
                )

    return latents.detach(), attention_mask.detach()


def compute_channel_stats(
    dataloader: torch.utils.data.DataLoader,
    autoencoder: BARTLatentCompression,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    running_sum: Optional[Tensor] = None
    running_sq: Optional[Tensor] = None
    token_count = 0
    processed_sequences = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Pass 1: moments", leave=False)):
        latents, attention_mask = extract_latents(batch, autoencoder, args.training_phase, args.ablation_mode, device)
        latents = latents.to(torch.float32).cpu()
        mask = attention_mask.to(torch.bool).cpu()

        if running_sum is None:
            running_sum = torch.zeros(latents.size(-1), dtype=torch.float64)
            running_sq = torch.zeros(latents.size(-1), dtype=torch.float64)

        masked = latents[mask]
        token_count += masked.size(0)
        running_sum += masked.sum(dim=0, dtype=torch.float64)
        running_sq += (masked.double() ** 2).sum(dim=0)

        processed_sequences += latents.size(0)
        if args.max_sequences and processed_sequences >= args.max_sequences:
            LOG.info("Reached --max_sequences=%d during statistics pass", args.max_sequences)
            break

    if token_count == 0:
        raise RuntimeError("No tokens encountered while computing statistics")

    mean = (running_sum / token_count).to(torch.float32)
    var = (running_sq / token_count) - (mean.double() ** 2)
    std = torch.sqrt(var.clamp_min(1e-8)).to(torch.float32)

    return mean, std, token_count


def reservoir_sampling(
    reservoir: np.ndarray,
    next_index: int,
    rng: np.random.Generator,
    tokens: np.ndarray,
) -> Tuple[int, int]:
    total_seen = next_index
    filled = min(reservoir.shape[0], total_seen)

    for token in tokens:
        total_seen += 1
        if filled < reservoir.shape[0]:
            reservoir[filled] = token
            filled += 1
        else:
            j = rng.integers(0, total_seen)
            if j < reservoir.shape[0]:
                reservoir[j] = token

    return total_seen, filled


def sample_standardized_tokens(
    dataloader: torch.utils.data.DataLoader,
    autoencoder: BARTLatentCompression,
    mean: Tensor,
    std: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    latent_dim = mean.numel()
    rng = np.random.default_rng(args.seed)

    if args.token_cap is None:
        token_buffer: Optional[np.ndarray] = None
    else:
        token_buffer = np.empty((args.token_cap, latent_dim), dtype=np.float32)

    samples_list = []
    total_seen = 0
    processed_sequences = 0

    for batch in tqdm(dataloader, desc="Pass 2: sampling", leave=False):
        latents, attention_mask = extract_latents(batch, autoencoder, args.training_phase, args.ablation_mode, device)
        latents = latents.to(torch.float32).cpu()
        mask = attention_mask.to(torch.bool).cpu()

        standardized = (latents - mean) / std
        tokens = standardized[mask].numpy()

        if tokens.size == 0:
            continue

        if token_buffer is None:
            samples_list.append(tokens)
        else:
            total_seen, _ = reservoir_sampling(token_buffer, total_seen, rng, tokens)

        processed_sequences += latents.size(0)
        if args.max_sequences and processed_sequences >= args.max_sequences:
            LOG.debug("Reached --max_sequences=%d during sampling pass", args.max_sequences)
            break

    if token_buffer is not None:
        filled = min(total_seen, token_buffer.shape[0]) if total_seen else 0
        samples = token_buffer[:filled]
        sample_count = filled
    else:
        if not samples_list:
            raise RuntimeError("No tokens collected for PCA")
        samples = np.concatenate(samples_list, axis=0)
        sample_count = samples.shape[0]

    LOG.info("Collected %d standardized tokens for PCA", sample_count)
    return samples, sample_count


def fit_pca(
    samples: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[PCA, int]:
    if samples.shape[0] == 0:
        raise RuntimeError("Cannot fit PCA without samples")

    if args.variance_threshold is not None:
        n_components = args.variance_threshold
    else:
        n_components = min(args.pca_components, samples.shape[1], samples.shape[0])

    pca = PCA(
        n_components=n_components,
        whiten=args.whiten,
        svd_solver="auto",
        random_state=args.seed,
    )
    pca.fit(samples)

    # Optionally clamp number of components
    actual_components = pca.components_.shape[0]
    if args.variance_threshold is not None and isinstance(n_components, float) and args.pca_components:
        keep = min(args.pca_components, actual_components)
        if keep < actual_components:
            pca.components_ = pca.components_[:keep]
            pca.explained_variance_ = pca.explained_variance_[:keep]
            pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:keep]
            actual_components = keep

    LOG.info(
        "Fitted PCA with %d components (variance %.3f)",
        actual_components,
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return pca, actual_components


def save_artifacts(
    artifacts: PCAArtifacts,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.to_serializable(), output_path)
    LOG.info("Saved PCA artifacts to %s", output_path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    LOG.info("Using device: %s", device)

    autoencoder = load_autoencoder(args, device)

    # Cap encoding max_length to the model's positional embedding limit to avoid CUDA indexing asserts.
    model_max_pos = int(getattr(getattr(autoencoder, "config", None), "max_position_embeddings", 0) or 0)
    effective_max_length = int(args.max_length or 0)
    if effective_max_length <= 0:
        effective_max_length = model_max_pos if model_max_pos > 0 else 512
    if model_max_pos > 0 and effective_max_length > model_max_pos:
        LOG.warning(
            "Requested --max_length=%d exceeds autoencoder.config.max_position_embeddings=%d; "
            "capping to %d to avoid position-embedding indexing errors.",
            effective_max_length, model_max_pos, model_max_pos,
        )
        effective_max_length = model_max_pos

    dataloader = build_dataloader(args, max_length=effective_max_length)

    mean, std, token_count = compute_channel_stats(dataloader, autoencoder, args, device)
    LOG.info("Computed channel statistics over %d tokens", token_count)

    samples, sample_count = sample_standardized_tokens(dataloader, autoencoder, mean, std, args, device)

    if args.skip_pca:
        LOG.info("Skipping PCA rotation; storing identity components only")
        dim = mean.numel()
        component_dim = dim
        components = torch.eye(dim, dtype=torch.float32)
        explained_variance = torch.ones(dim, dtype=torch.float32)
        explained_ratio = torch.ones(dim, dtype=torch.float32) / dim
        args.whiten = False
    else:
        pca, component_dim = fit_pca(samples, args)
        components = torch.from_numpy(pca.components_.astype(np.float32))
        explained_variance = torch.from_numpy(pca.explained_variance_.astype(np.float32))
        explained_ratio = torch.from_numpy(pca.explained_variance_ratio_.astype(np.float32))

    artifact = PCAArtifacts(
        mean=mean,
        std=std,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_ratio,
        whitening=args.whiten,
        latent_dim=mean.numel(),
        component_dim=component_dim,
        token_count=token_count,
        sample_count=sample_count,
        metadata={
            "autoencoder_path": os.path.abspath(args.autoencoder_path),
            "data_dir": args.data_dir,
            "data_type": args.data_type,
            "training_phase": args.training_phase,
            "ablation_mode": args.ablation_mode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "token_cap": args.token_cap,
            "variance_threshold": args.variance_threshold,
            "pca_components": args.pca_components,
            "skip_pca": args.skip_pca,
            "seed": args.seed,
        },
    )

    save_artifacts(artifact, Path(args.output_path))


if __name__ == "__main__":
    main()
