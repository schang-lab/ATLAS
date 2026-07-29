#!/usr/bin/env python3
"""
VAE-backbone inference for Embee trajectory generation.

Mirrors inference_dit_only_embee.py but replaces DiT + diffusion sampling
with VAE prior sampling. The BART decoder, POI conversion, and output format
are identical.

Usage:
    python inference_vae_embee.py \
        --vae_config /path/to/config_vae_phase1.yml \
        --vae_checkpoint /path/to/vae_final.pt \
        --autoencoder_path /path/to/pretrained_autoencoder \
        --split_dir /path/to/split_data \
        --output_dir /path/to/output \
        --num_samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PARENT_DIR)

from src.vae import TrajectoryVAE
from src.latent_pca import LatentPCA
from src.utils import batch_count_lengths

from transformers import BartConfig, LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput


# ---- Helpers (reused from inference_dit_only_embee.py) ----

def _normalize_sequence(seq) -> List[str]:
    if isinstance(seq, str):
        return seq.strip().split()
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [str(token) for token in seq]
    raise TypeError(f"Unsupported trajectory format: {type(seq)}")


def load_length_ids(split_dir: Union[str, Path]) -> Optional[np.ndarray]:
    path = Path(split_dir) / "trajectory_length_ids.npy"
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True).astype(np.int64)
    except Exception as exc:
        print(f"Warning: failed to load {path} ({exc})")
        return None


def select_length_subset(length_cache: Optional[np.ndarray], count: int,
                         indices: Optional[np.ndarray] = None) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if length_cache is None or len(length_cache) == 0:
        print("Warning: missing cached trajectory lengths; defaulting to zeros")
        return np.zeros(count, dtype=np.int64)
    if indices is not None:
        return length_cache[indices]
    if count <= len(length_cache):
        return length_cache[:count]
    return length_cache[np.random.choice(len(length_cache), count, replace=True)]


def _load_demo_pairs_from_attrs_with_demo(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    arr = np.load(str(path), allow_pickle=True)
    demo = arr[:, -2:].astype(np.int64)
    valid = (demo[:, 0] >= 0) & (demo[:, 1] >= 0)
    demo = demo[valid]
    if demo.shape[0] == 0:
        raise ValueError(f"No valid demo rows found in {path}")
    return demo


def _sample_demo_pairs_from_real_demo(*, n: int, real_demo_pairs: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(real_demo_pairs.shape[0], size=int(n), replace=True)
    return real_demo_pairs[idx].astype(np.int64)


def load_poi_mapping(tokenizer_path: str, poi_coords_path: str) -> Tuple[Dict[str, int], Dict[str, Tuple[float, float]]]:
    vocab_file = os.path.join(tokenizer_path, "vocab.txt")
    vocab: Dict[str, int] = {}
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        first_line = lines[0].strip() if lines else ""
        if "\t" in first_line or " " in first_line:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vocab[parts[0]] = int(parts[1])
        else:
            for token_id, line in enumerate(lines):
                token = line.strip()
                if token:
                    vocab[token] = token_id
    poi_coords: Dict[str, Tuple[float, float]] = {}
    if os.path.exists(poi_coords_path):
        import pandas as pd
        poi_df = pd.read_csv(poi_coords_path)
        for _, row in poi_df.iterrows():
            poi_coords[str(row["poi_id"])] = (float(row["lat"]), float(row["lon"]))
    return vocab, poi_coords


class ForbiddenTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, forbidden_token_ids: List[int]):
        super().__init__()
        self.forbidden_token_ids = sorted({int(t) for t in forbidden_token_ids})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.forbidden_token_ids:
            return scores
        for tid in self.forbidden_token_ids:
            if 0 <= tid < scores.size(-1):
                scores[:, tid] = float("-inf")
        return scores


class UnkTokenPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, unk_token_ids: List[int], penalty: float = 5.0):
        super().__init__()
        self.unk_token_ids = sorted(set(int(t) for t in unk_token_ids))
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for tid in self.unk_token_ids:
            if 0 <= tid < scores.size(-1):
                scores[:, tid] -= self.penalty
        return scores


def convert_poi_sequences_to_coordinates_embee(
    poi_sequences: torch.Tensor,
    vocab: Dict[str, int],
    poi_coords: Dict[str, Tuple[float, float]],
    attrs_4d: np.ndarray,
    max_length: int,
    poi_home_token: str,
    poi_work_token: str,
    poi_other_token: str,
) -> np.ndarray:
    batch_size, seq_len = poi_sequences.shape
    bart_special_ids = {0, 1, 2, 3, 4}
    id_to_token = {v: k for k, v in vocab.items()}
    out = np.zeros((batch_size, 2, max_length), dtype=np.float32)
    kept = injected = skipped_other = skipped_special = missing_map = 0

    for b in range(batch_size):
        work_lat, work_lon, home_lat, home_lon = attrs_4d[b, 0:4].tolist()
        out_idx = 0
        for t in range(seq_len):
            if out_idx >= max_length:
                break
            tid = int(poi_sequences[b, t].item())
            if tid in bart_special_ids:
                skipped_special += 1
                continue
            tok = id_to_token.get(tid, "[UNK]")
            if tok == poi_other_token:
                skipped_other += 1
                continue
            if tok == poi_home_token:
                out[b, 0, out_idx] = float(home_lat)
                out[b, 1, out_idx] = float(home_lon)
                out_idx += 1
                injected += 1
                continue
            if tok == poi_work_token:
                out[b, 0, out_idx] = float(work_lat)
                out[b, 1, out_idx] = float(work_lon)
                out_idx += 1
                injected += 1
                continue
            if tok in poi_coords:
                lat, lon = poi_coords[tok]
                out[b, 0, out_idx] = float(lat)
                out[b, 1, out_idx] = float(lon)
                out_idx += 1
                kept += 1
            else:
                missing_map += 1

    total_seen = kept + missing_map + injected
    if total_seen > 0:
        miss_rate = missing_map / total_seen
        print(
            f"[Embee VAE] coord conversion: kept={kept}, injected={injected}, "
            f"other={skipped_other}, special={skipped_special}, "
            f"missing={missing_map}, miss_rate={miss_rate:.3f}"
        )
    return out


def convert_poi_sequences_to_poi_ids_embee(
    poi_sequences: torch.Tensor,
    vocab: Dict[str, int],
    max_length: int,
    poi_other_token: str,
) -> List[List[str]]:
    batch_size, seq_len = poi_sequences.shape
    filter_special_ids = {0, 1, 2, 4}
    id_to_token = {v: k for k, v in vocab.items()}
    sequences: List[List[str]] = []

    for b in range(batch_size):
        out: List[str] = []
        for t in range(seq_len):
            if len(out) >= max_length:
                break
            tid = int(poi_sequences[b, t].item())
            if tid in filter_special_ids:
                continue
            tok = id_to_token.get(tid, "[UNK]")
            if tok == poi_other_token:
                continue
            out.append(tok)
        sequences.append(out)
    return sequences


@torch.no_grad()
def sample_vae_with_autoencoder(
    vae_model: TrajectoryVAE,
    autoencoder,
    attr_embeds: Optional[torch.Tensor],
    max_traj_length: int,
    min_traj_length: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    generation_config: dict,
    actual_batch_size: Optional[int] = None,
    latent_scale: float = 1.0,
    latent_pca: Optional[LatentPCA] = None,
    length_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample trajectories using VAE prior sampling + BART decoder."""
    device = next(vae_model.parameters()).device

    if attr_embeds is not None:
        attr_embeds = attr_embeds.to(device=device, dtype=torch.float32)
    batch_size = actual_batch_size or (attr_embeds.shape[0] if attr_embeds is not None else 1)

    # --- VAE generation: sample from prior N(0,I) and decode ---
    vae_model.eval()
    latents = vae_model.generate(
        batch_size=batch_size,
        attr_embeds=attr_embeds,
        device=device,
    )  # (B, T, D_in) in PCA space

    # Unproject from PCA if needed
    ae_latents = latent_pca.unproject(latents) if latent_pca is not None else latents
    decoder_inputs = ae_latents * latent_scale if latent_scale != 1.0 else ae_latents

    bs = decoder_inputs.shape[0]
    seq_len = decoder_inputs.shape[1]
    attention_mask = torch.ones(bs, seq_len, device=device, dtype=torch.bool)

    # Length conditioning
    length_tensor = None
    if length_ids is not None:
        if isinstance(length_ids, torch.Tensor):
            length_tensor = length_ids.to(device=device, dtype=torch.long).view(-1)
        else:
            length_tensor = torch.as_tensor(length_ids, device=device, dtype=torch.long).view(-1)

    # Logits processors
    forbidden_token_ids = generation_config.get("forbidden_token_ids", [])
    logits_processors = []
    if forbidden_token_ids:
        logits_processors.append(ForbiddenTokensLogitsProcessor(forbidden_token_ids=forbidden_token_ids))

    decoder_start_id = getattr(autoencoder.config, "decoder_start_token_id", None)
    decoder_offset = 1 if decoder_start_id is not None else 0
    eos_token_id = getattr(autoencoder.config, "eos_token_id", None)
    eos_offset = 1 if eos_token_id is not None else 0
    extra_specials = decoder_offset + eos_offset
    pad_token_id = getattr(autoencoder.config, "pad_token_id", 0)

    decoder_start_token_id = int(decoder_start_id) if decoder_start_id is not None else 1
    eos_token_id_for_generate = int(eos_token_id) if eos_token_id is not None else 2
    pad_token_id_for_generate = int(pad_token_id) if pad_token_id is not None else 0

    bart_max_pos = getattr(autoencoder.config, "max_position_embeddings", 64)

    def _run_generation(hidden_states, attn_mask, target_tokens=None):
        max_allowed = min(max_traj_length + extra_specials, int(bart_max_pos))
        content_cap = max(1, int(max_allowed - extra_specials))

        if target_tokens is not None:
            desired = max(1, min(int(target_tokens), content_cap))
            max_length_local = int(desired + extra_specials)
            min_length_local = int(max_length_local)
        else:
            max_length_local = int(max_allowed)
            min_length_local = int(max(min_traj_length + extra_specials, extra_specials + 1))
            if min_length_local > max_length_local:
                min_length_local = max_length_local

        return autoencoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=attn_mask,
            max_length=max_length_local,
            min_length=min_length_local,
            num_beams=num_beams if not do_sample else 1,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=pad_token_id_for_generate,
            eos_token_id=eos_token_id_for_generate,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=True,
            repetition_penalty=generation_config.get("repetition_penalty", 1.1),
            length_penalty=generation_config.get("length_penalty", 1.2),
            no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 0),
            logits_processor=LogitsProcessorList(logits_processors),
            output_scores=True,
            return_dict_in_generate=True,
        )

    if length_tensor is None:
        gen_out = _run_generation(decoder_inputs, attention_mask)
        generated_ids = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
    else:
        # Group by length for efficient generation
        length_values = length_tensor.cpu().tolist()
        index_groups: Dict[int, List[int]] = {}
        for idx, lv in enumerate(length_values):
            index_groups.setdefault(int(lv), []).append(int(idx))

        per_sample_sequences = [None] * bs
        for raw_length, sample_indices in index_groups.items():
            target_tokens = max(1, min(int(raw_length), max_traj_length))
            subset_hidden = decoder_inputs[sample_indices]
            subset_mask = attention_mask[sample_indices]
            subset_output = _run_generation(subset_hidden, subset_mask, target_tokens=target_tokens)
            subset_seqs = subset_output.sequences if hasattr(subset_output, "sequences") else subset_output
            for local_idx, global_idx in enumerate(sample_indices):
                per_sample_sequences[global_idx] = subset_seqs[local_idx]

        max_len = max(s.size(0) for s in per_sample_sequences)
        generated_ids = torch.full((bs, max_len), pad_token_id_for_generate, device=device, dtype=torch.long)
        for i, seq in enumerate(per_sample_sequences):
            generated_ids[i, :seq.size(0)] = seq

    return latents, generated_ids


def main():
    parser = argparse.ArgumentParser(description="VAE inference for Embee trajectory generation")

    # Model
    parser.add_argument("--vae_config", type=str, required=True, help="VAE config YAML")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE model checkpoint (.pt)")
    parser.add_argument("--autoencoder_path", type=str, required=True, help="BART autoencoder path")
    parser.add_argument("--training_phase", type=str, default="phase1", choices=["phase1", "phase2"])
    parser.add_argument("--latent_pca_path", type=str, default=None)
    parser.add_argument("--latent_scale", type=float, default=1.0)

    # Data
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--poi_coords_csv", type=str, default=None)
    parser.add_argument("--attrs_npy", type=str, default=None, help="Attributes file (N x 4+)")
    parser.add_argument("--attrs_with_demo_npy", type=str, default=None, help="Attributes with demo (N x 6)")

    # Generation
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_traj_length", type=int, default=512)
    parser.add_argument("--min_traj_length", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--length_penalty", type=float, default=1.2)

    # Length conditioning
    parser.add_argument("--use_length_condition", action="store_true", default=False)

    # Demo conditioning
    parser.add_argument("--use_demo_condition", action="store_true", default=False)
    parser.add_argument("--demo_seed", type=int, default=42)

    # POI token names (Embee convention)
    parser.add_argument("--poi_home_token", type=str, default="POI_HOME")
    parser.add_argument("--poi_work_token", type=str, default="POI_WORK")
    parser.add_argument("--poi_other_token", type=str, default="POI_OTHER")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="vae")
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE config
    with open(args.vae_config, "r") as f:
        config = yaml.safe_load(f)
    vae_params = config.get("VAE", {})

    # Load latent PCA if specified
    latent_pca = None
    if args.latent_pca_path:
        latent_pca = LatentPCA(args.latent_pca_path, device)
        vae_params["in_channels"] = latent_pca.component_dim
        print(f"PCA loaded: {latent_pca.latent_dim} -> {latent_pca.component_dim}")

    # Create and load VAE
    vae_model = TrajectoryVAE(**vae_params).to(device)
    state_dict = torch.load(args.vae_checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    vae_model.load_state_dict(state_dict)
    vae_model.eval()
    print(f"VAE loaded from {args.vae_checkpoint}")
    print(f"  latent_code_dim={vae_model.latent_code_dim}, image_size={vae_model.image_size}, in_channels={vae_model.in_channels}")

    # Load autoencoder
    from auto_encoder.traj_compressed_ae import BARTLatentCompression
    from transformers import BartForConditionalGeneration

    if args.training_phase == "phase1":
        autoencoder = BartForConditionalGeneration.from_pretrained(args.autoencoder_path).to(device)
    else:
        autoencoder = BARTLatentCompression.from_pretrained(args.autoencoder_path).to(device)
    autoencoder.eval()
    print(f"Autoencoder loaded from {args.autoencoder_path}")

    # Load POI mapping — tokenizer lives in split_dir/tokenizer/, coords in poi_map_feature.csv
    tokenizer_dir = os.path.join(args.split_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        tokenizer_dir = args.split_dir  # fallback: vocab.txt directly in split_dir
    poi_coords_csv = args.poi_coords_csv or os.path.join(args.split_dir, "poi_map_feature.csv")
    if not os.path.exists(poi_coords_csv):
        poi_coords_csv = os.path.join(args.split_dir, "poi_coords.csv")  # legacy fallback
    vocab, poi_coords = load_poi_mapping(tokenizer_dir, poi_coords_csv)
    print(f"Loaded vocab: {len(vocab)} tokens, POI coords: {len(poi_coords)} POIs")

    # Load attributes
    attrs_path = args.attrs_with_demo_npy or args.attrs_npy
    if attrs_path is None:
        # Try default paths
        for candidate in ["all_attr_results_with_demo.npy", "all_attr_results.npy"]:
            p = os.path.join(args.split_dir, candidate)
            if os.path.exists(p):
                attrs_path = p
                break

    if attrs_path and os.path.exists(attrs_path):
        all_attrs = np.load(attrs_path, allow_pickle=True)
        print(f"Loaded attributes: {all_attrs.shape} from {attrs_path}")
    else:
        print("Warning: no attributes file found, using zeros")
        all_attrs = np.zeros((args.num_samples, 4), dtype=np.float32)

    # Load length ids if needed
    length_cache = load_length_ids(args.split_dir) if args.use_length_condition else None

    # Load demo pairs if needed
    real_demo_pairs = None
    if args.use_demo_condition and args.attrs_with_demo_npy:
        real_demo_pairs = _load_demo_pairs_from_attrs_with_demo(args.attrs_with_demo_npy)
        print(f"Loaded {real_demo_pairs.shape[0]} demo pairs for sampling")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate in batches
    num_samples = args.num_samples
    all_poi_sequences = []
    all_coord_trajectories = []
    all_attrs_used = []

    num_generated = 0
    batch_idx = 0

    generation_config = {
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": 0,
    }

    pbar = tqdm(total=num_samples, desc="Generating trajectories")

    while num_generated < num_samples:
        bs = min(args.batch_size, num_samples - num_generated)

        # Sample attributes for this batch
        if num_generated + bs <= all_attrs.shape[0]:
            batch_attrs = all_attrs[num_generated : num_generated + bs].copy()
        else:
            indices = np.random.choice(all_attrs.shape[0], bs, replace=True)
            batch_attrs = all_attrs[indices].copy()

        # Prepare attr tensor
        attr_tensor = torch.tensor(batch_attrs[:, :4], dtype=torch.float32, device=device)

        # Add length conditioning
        batch_length_ids = None
        if args.use_length_condition and length_cache is not None:
            batch_length_ids = select_length_subset(length_cache, bs)
            length_tensor = torch.tensor(batch_length_ids, dtype=torch.float32, device=device).unsqueeze(-1)
            attr_tensor = torch.cat([attr_tensor, length_tensor], dim=-1)

        # Add demo conditioning
        if args.use_demo_condition and batch_attrs.shape[1] >= 6:
            age_raw = torch.tensor(batch_attrs[:, -2], dtype=torch.float32, device=device).clamp_min(0) + 1
            gender_raw = torch.tensor(batch_attrs[:, -1], dtype=torch.float32, device=device).clamp_min(0) + 1
            # Handle missing values
            missing = (batch_attrs[:, -2] < 0) | (batch_attrs[:, -1] < 0)
            age_raw[missing] = 0
            gender_raw[missing] = 0
            attr_tensor = torch.cat([attr_tensor, age_raw.unsqueeze(-1), gender_raw.unsqueeze(-1)], dim=-1)
        elif args.use_demo_condition and real_demo_pairs is not None:
            demo = _sample_demo_pairs_from_real_demo(n=bs, real_demo_pairs=real_demo_pairs, seed=args.demo_seed + batch_idx)
            age_shifted = torch.tensor(demo[:, 0] + 1, dtype=torch.float32, device=device)
            gender_shifted = torch.tensor(demo[:, 1] + 1, dtype=torch.float32, device=device)
            attr_tensor = torch.cat([attr_tensor, age_shifted.unsqueeze(-1), gender_shifted.unsqueeze(-1)], dim=-1)

        # Generate
        latents, generated_ids = sample_vae_with_autoencoder(
            vae_model=vae_model,
            autoencoder=autoencoder,
            attr_embeds=attr_tensor,
            max_traj_length=args.max_traj_length,
            min_traj_length=args.min_traj_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            generation_config=generation_config,
            actual_batch_size=bs,
            latent_scale=args.latent_scale,
            latent_pca=latent_pca,
            length_ids=batch_length_ids,
        )

        # Convert to POI sequences
        poi_seqs = convert_poi_sequences_to_poi_ids_embee(
            generated_ids, vocab, args.max_traj_length, args.poi_other_token
        )
        all_poi_sequences.extend(poi_seqs)

        # Convert to coordinates
        coord_traj = convert_poi_sequences_to_coordinates_embee(
            generated_ids, vocab, poi_coords, batch_attrs[:bs, :4],
            args.max_traj_length, args.poi_home_token, args.poi_work_token, args.poi_other_token,
        )
        all_coord_trajectories.append(coord_traj)
        all_attrs_used.append(batch_attrs[:bs])

        num_generated += bs
        batch_idx += 1
        pbar.update(bs)

    pbar.close()

    # Save outputs
    coord_array = np.concatenate(all_coord_trajectories, axis=0)
    attrs_array = np.concatenate(all_attrs_used, axis=0)

    output_prefix = args.output_prefix
    np.save(os.path.join(args.output_dir, f"{output_prefix}_trajectories.npy"), coord_array)
    np.save(os.path.join(args.output_dir, f"{output_prefix}_attrs.npy"), attrs_array)

    # Save POI sequences as pickle
    with open(os.path.join(args.output_dir, f"{output_prefix}_poi_sequences.pkl"), "wb") as f:
        pickle.dump(all_poi_sequences, f)

    # Save generation config
    gen_info = {
        "model": "VAE",
        "vae_config": args.vae_config,
        "vae_checkpoint": args.vae_checkpoint,
        "num_samples": num_generated,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "latent_code_dim": vae_model.latent_code_dim,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
    }
    with open(os.path.join(args.output_dir, f"{output_prefix}_generation_info.json"), "w") as f:
        json.dump(gen_info, f, indent=2)

    print(f"\nSaved {num_generated} trajectories to {args.output_dir}")
    print(f"  Coordinates: {coord_array.shape}")
    print(f"  Attributes: {attrs_array.shape}")
    print(f"  POI sequences: {len(all_poi_sequences)} trajectories")


if __name__ == "__main__":
    main()
