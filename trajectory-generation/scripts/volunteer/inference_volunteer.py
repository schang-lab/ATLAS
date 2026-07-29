#!/usr/bin/env python3
"""
Generate trajectories using trained VOLUNTEER-ATLAS model.

Outputs are compatible with the existing evaluation pipeline:
  - generated_sequences.npy: (N, T) int — POI token IDs
  - generated_coordinates.npy: (N, 2, T) float — lat/lon coordinates
  - sampled_attributes.npy: (N, 6+) float — Embee attr layout:
        [work_lat, work_lon, home_lat, home_lon, ..., age, gender]
  - generated_poi_sequences.pkl: list of list of token strings

Example:
    python trajectory-generation/scripts/volunteer/inference_volunteer.py \
        --checkpoint runs/volunteer_atlas/phase2_final.pt \
        --config trajectory-generation/scripts/volunteer/config_volunteer.yaml \
        --split test \
        --output_dir runs/volunteer_atlas/inference_test
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from volunteer_model import VolunteerVAE

import yaml


EMBEE_ATTR_LAYOUT = "work_lat,work_lon,home_lat,home_lon,...,age_bin,gender_id"


def load_poi_mapping(tokenizer_dir: str, poi_csv: str):
    """Load vocab and POI coordinate mapping."""
    # Vocab: token string -> token id
    vocab_path = os.path.join(tokenizer_dir, "vocab.txt")
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f]
    id_to_token = {i: t for i, t in enumerate(vocab)}

    # POI coordinates
    poi_df = pd.read_csv(poi_csv)
    poi_coords = {}
    for _, row in poi_df.iterrows():
        poi_coords[row["poi_id"]] = (row["lat"], row["lon"])

    return vocab, id_to_token, poi_coords


def tokens_to_coords(
    token_ids: np.ndarray,
    id_to_token: Dict[int, str],
    poi_coords: Dict[str, tuple],
    home: np.ndarray,
    work: np.ndarray,
    max_len: int,
) -> np.ndarray:
    """Convert token IDs to (lat, lon) coordinates.

    Returns: (2, max_len) array of [lat_seq, lon_seq].
    """
    lats = np.zeros(max_len)
    lons = np.zeros(max_len)

    for t in range(min(len(token_ids), max_len)):
        tid = int(token_ids[t])
        token = id_to_token.get(tid, "[PAD]")

        if token == "POI_HOME":
            lats[t], lons[t] = home[0], home[1]
        elif token == "POI_WORK":
            lats[t], lons[t] = work[0], work[1]
        elif token == "POI_OTHER":
            # Use home as fallback for POI_OTHER
            lats[t], lons[t] = home[0], home[1]
        elif token in poi_coords:
            lats[t], lons[t] = poi_coords[token]
        elif token.startswith("POI::") and token in poi_coords:
            lats[t], lons[t] = poi_coords[token]
        else:
            # Special tokens ([PAD], [CLS], [SEP], etc.) or unknown POIs
            lats[t], lons[t] = 0.0, 0.0

    return np.stack([lats, lons], axis=0)  # (2, max_len)


def get_pad_id(vocab: List[str]) -> int:
    """PAD token id, matching VolunteerTrajectoryDataset's resolution order."""
    tok2id = {t: i for i, t in enumerate(vocab)}
    return int(tok2id.get("[PAD]", tok2id.get("<pad>", 0)))


def load_real_lengths(split_dir: Path, max_len: int) -> np.ndarray:
    """Per-trajectory real lengths from the split's attention_mask (1=valid).

    Returns an (N,) int array aligned row-for-row with all_attr_results_with_demo.npy,
    clipped to [1, max_len]. This matches the number of valid (non-pad) positions the
    model was trained on.
    """
    with open(split_dir / "final_segments_all_train_data.pkl", "rb") as f:
        segments_df = pickle.load(f)
    lengths = []
    for _, row in segments_df.iterrows():
        L = int(np.asarray(row["attention_mask"]).sum())
        lengths.append(int(min(max(L, 1), max_len)))
    return np.asarray(lengths, dtype=np.int64)


def load_length_dist_json(path: str, max_len: int) -> np.ndarray:
    """Pool a per-region length_dists JSON into one probability vector of size max_len+1.

    Schema (from build_length_dists_from_llp_world.py):
        {region_id: {"probs": [p0..p_maxlen], "count": N}, ...}
    Regions are pooled weighted by their trajectory count.
    """
    import json
    with open(path) as f:
        dists = json.load(f)
    pooled = np.zeros(max_len + 1, dtype=np.float64)
    total = 0.0
    for region in dists.values():
        probs = np.asarray(region.get("probs", []), dtype=np.float64)
        count = float(region.get("count", 0))
        if probs.size == 0 or count <= 0:
            continue
        # Re-bin to [0, max_len] in case the JSON used a different max_length.
        p = np.zeros(max_len + 1, dtype=np.float64)
        n = min(probs.size, max_len + 1)
        p[:n] = probs[:n]
        if probs.size > max_len + 1:  # fold overflow mass into max_len
            p[max_len] += probs[max_len + 1:].sum()
        s = p.sum()
        if s > 0:
            pooled += (p / s) * count
            total += count
    if total <= 0:
        pooled[:] = 1.0 / (max_len + 1)
    else:
        pooled /= total
    pooled[0] = 0.0  # never generate length-0 trajectories
    s = pooled.sum()
    pooled = pooled / s if s > 0 else np.full(max_len + 1, 1.0 / (max_len + 1))
    return pooled


def split_embee_attrs(attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split Embee attrs into semantic model inputs.

    Embee attrs are stored as:
        [work_lat, work_lon, home_lat, home_lon, ..., age_bin, gender_id]

    Demo IDs are read from the last two columns to match the Embee inference
    scripts, which may insert optional fields such as length_id before demos.
    """
    if attrs.ndim != 2 or attrs.shape[1] < 6:
        raise ValueError(
            f"Expected attrs with shape [N,6+] using layout [{EMBEE_ATTR_LAYOUT}], got {attrs.shape}."
        )
    work = attrs[:, 0:2]
    home = attrs[:, 2:4]
    age_bin = attrs[:, -2]
    gender_id = attrs[:, -1]
    return home, work, age_bin, gender_id


def main():
    parser = argparse.ArgumentParser(description="VOLUNTEER-ATLAS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to generate. Default: same as split size.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # --- Trajectory-length control -------------------------------------------
    # The model decodes all max_len positions in parallel (no EOS), so without
    # this every trajectory has exactly max_len stops. These options impose a
    # target length per trajectory and PAD the tail.
    parser.add_argument(
        "--length_mode", type=str, default="realdist",
        choices=["none", "realdist", "length_dists_json", "fixed"],
        help=(
            "none: keep full max_len (old behavior). "
            "realdist (default): use each conditioning trajectory's real length "
            "from the split's attention_mask (aligned per-sample). "
            "length_dists_json: sample lengths from a saved per-region length_dists JSON. "
            "fixed: use --target_length for every trajectory."
        ),
    )
    parser.add_argument("--length_dists_json", type=str, default=None,
                        help="Path to length_dists JSON (required for --length_mode length_dists_json).")
    parser.add_argument("--target_length", type=int, default=None,
                        help="Target length for --length_mode fixed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model_cfg = cfg["model"]
    model = VolunteerVAE(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load data for conditioning attributes
    data_cfg = cfg["data"]
    split_dir = Path(data_cfg["split_data_dir"]) / args.split
    attrs = np.load(split_dir / "all_attr_results_with_demo.npy", allow_pickle=True).astype(np.float32)
    print(
        f"Loaded {attrs.shape[0]} conditioning attributes from {args.split} split "
        f"using Embee layout [{EMBEE_ATTR_LAYOUT}]"
    )

    # Load POI mapping for coordinate conversion
    tokenizer_dir = str(split_dir / "tokenizer")
    poi_csv = str(split_dir / "poi_map_feature.csv")
    vocab, id_to_token, poi_coords = load_poi_mapping(tokenizer_dir, poi_csv)

    # Determine number of samples
    num_samples = args.num_samples or attrs.shape[0]
    if num_samples > attrs.shape[0]:
        # Oversample with replacement
        indices = np.random.choice(attrs.shape[0], num_samples, replace=True)
    else:
        indices = np.arange(num_samples)

    max_len = model_cfg.get("max_seq_len", 64)
    pad_id = get_pad_id(vocab)

    # --- Resolve a target length per generated trajectory --------------------
    # The decoder emits all max_len positions in parallel (no EOS), so we impose
    # a length here and PAD the tail. See --length_mode.
    target_lengths = np.full(num_samples, max_len, dtype=np.int64)
    if args.length_mode == "realdist":
        real_lengths = load_real_lengths(split_dir, max_len)  # aligned to attrs rows
        target_lengths = real_lengths[indices]
        print(f"Length control: realdist (per-conditioning-trajectory real length, "
              f"median={int(np.median(target_lengths))}, min={int(target_lengths.min())}, "
              f"max={int(target_lengths.max())})")
    elif args.length_mode == "length_dists_json":
        if not args.length_dists_json:
            parser.error("--length_mode length_dists_json requires --length_dists_json PATH")
        probs = load_length_dist_json(args.length_dists_json, max_len)
        target_lengths = np.random.choice(np.arange(max_len + 1), size=num_samples, p=probs)
        target_lengths = np.clip(target_lengths, 1, max_len).astype(np.int64)
        print(f"Length control: length_dists_json {args.length_dists_json} "
              f"(sampled median={int(np.median(target_lengths))})")
    elif args.length_mode == "fixed":
        if args.target_length is None:
            parser.error("--length_mode fixed requires --target_length N")
        target_lengths = np.full(num_samples, int(np.clip(args.target_length, 1, max_len)),
                                 dtype=np.int64)
        print(f"Length control: fixed length={int(target_lengths[0])}")
    else:
        print("Length control: none (full max_len trajectories)")

    # Generate in batches
    all_sequences = []
    all_coords = []
    all_attrs = []
    all_poi_sequences = []

    for start in tqdm(range(0, num_samples, args.batch_size), desc="Generating"):
        end = min(start + args.batch_size, num_samples)
        batch_idx = indices[start:end]
        B = len(batch_idx)

        batch_attrs = attrs[batch_idx]
        home_np, work_np, age_np, gender_np = split_embee_attrs(batch_attrs)
        home = torch.tensor(home_np, dtype=torch.float32, device=device)
        work = torch.tensor(work_np, dtype=torch.float32, device=device)
        age_bin = torch.tensor(age_np, dtype=torch.long, device=device)
        gender_id = torch.tensor(gender_np, dtype=torch.long, device=device)

        with torch.no_grad():
            gen_out = model.generate(
                age_bin=age_bin,
                gender_id=gender_id,
                home=home,
                work=work,
                max_len=max_len,
                temperature=args.temperature,
            )

        loc_ids = gen_out["loc_ids"].cpu().numpy()  # (B, T)

        # Impose target lengths: keep first L tokens, PAD the tail.
        batch_lengths = target_lengths[start:end]
        pos_idx = np.arange(loc_ids.shape[1])[None, :]           # (1, T)
        keep = pos_idx < batch_lengths[:, None]                  # (B, T)
        loc_ids = np.where(keep, loc_ids, pad_id)

        all_sequences.append(loc_ids)

        # Convert to coordinates
        for i in range(B):
            coords = tokens_to_coords(
                loc_ids[i], id_to_token, poi_coords,
                home_np[i], work_np[i],
                max_len,
            )
            all_coords.append(coords)

            # Token strings for POI sequence
            poi_seq = [id_to_token.get(int(tid), "[UNK]") for tid in loc_ids[i]]
            all_poi_sequences.append(poi_seq)

        all_attrs.append(batch_attrs)

    # Concatenate and save
    os.makedirs(args.output_dir, exist_ok=True)

    seq_out = np.concatenate(all_sequences, axis=0)  # (N, T)
    coord_out = np.stack(all_coords, axis=0)          # (N, 2, T)
    attrs_out = np.concatenate(all_attrs, axis=0)     # (N, 6)

    np.save(os.path.join(args.output_dir, "generated_sequences.npy"), seq_out)
    np.save(os.path.join(args.output_dir, "generated_coordinates.npy"), coord_out)
    np.save(os.path.join(args.output_dir, "sampled_attributes.npy"), attrs_out)
    np.save(os.path.join(args.output_dir, "sampled_attributes_4d.npy"), attrs_out[:, :4])
    np.save(os.path.join(args.output_dir, "generated_lengths.npy"), target_lengths)

    with open(os.path.join(args.output_dir, "generated_poi_sequences.pkl"), "wb") as f:
        pickle.dump(all_poi_sequences, f)

    generation_params = {
        "num_samples": int(num_samples),
        "split": args.split,
        "temperature": float(args.temperature),
        "max_len": int(max_len),
        "checkpoint": args.checkpoint,
        "model_type": "volunteer_vae",
        "attr_layout": EMBEE_ATTR_LAYOUT,
        "length_mode": args.length_mode,
        "length_dists_json": args.length_dists_json,
        "target_length": args.target_length,
        "pad_id": int(pad_id),
        "median_generated_length": int(np.median(target_lengths)),
    }
    with open(os.path.join(args.output_dir, "generation_params.json"), "w") as f:
        import json
        json.dump(generation_params, f, indent=2)

    print(f"Saved {num_samples} generated trajectories to {args.output_dir}")
    print(f"  generated_sequences.npy: {seq_out.shape}")
    print(f"  generated_coordinates.npy: {coord_out.shape}")
    print(f"  sampled_attributes.npy: {attrs_out.shape}")


if __name__ == "__main__":
    main()
