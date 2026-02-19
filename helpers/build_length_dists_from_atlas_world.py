#!/usr/bin/env python3
"""
Build per-region trajectory length distributions JSON from an LLP-world directory.

This outputs the JSON schema expected by:
  trajectory-generation/scripts/training/run_cbg_conditioned_training.py
  (config key: data.length_dists_json)

The LLP-world directory should contain per-region subfolders like:
  <world_root>/<region_id>/generated_sequences.npy

This script does NOT require PyYAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _iter_region_dirs(world_root: Path) -> List[Path]:
    return sorted([p for p in world_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _sequence_length_from_ids(ids: np.ndarray, *, num_special_tokens: int) -> int:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
        ids = ids.reshape(-1)
    if num_special_tokens > 0:
        ids = ids[ids >= int(num_special_tokens)]
    return int(ids.size)


def _load_lengths_for_region(
    region_dir: Path,
    *,
    num_special_tokens: int,
    max_length: int,
    min_length: int,
    clip: bool,
) -> np.ndarray:
    seq_path = region_dir / "generated_sequences.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")
    seqs = np.load(seq_path, allow_pickle=True)
    lengths: List[int] = []
    for x in seqs.tolist():
        if x is None:
            continue
        L = _sequence_length_from_ids(np.asarray(x), num_special_tokens=num_special_tokens)
        if clip:
            L = int(min(max(L, 0), max_length))
        if L < min_length:
            continue
        lengths.append(L)
    return np.asarray(lengths, dtype=np.int64)


def _hist_to_probs(lengths: np.ndarray, *, max_length: int) -> Tuple[np.ndarray, int]:
    if lengths.size == 0:
        probs = np.ones(max_length + 1, dtype=float) / float(max_length + 1)
        return probs, 0
    hist = np.bincount(lengths, minlength=max_length + 1).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        probs = np.ones(max_length + 1, dtype=float) / float(max_length + 1)
    else:
        probs = hist / total
    return probs, int(lengths.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-root", type=str, required=True, help="e.g. /path/to/atlas_world/world_train_demogroups")
    ap.add_argument("--out-json", type=str, required=True, help="e.g. /path/to/atlas_world/length_dists_train.json")
    ap.add_argument("--max-length", type=int, required=True, help="Model seq_len / DiT image_size (e.g. 64)")
    ap.add_argument("--min-length", type=int, default=1, help="Drop trajectories shorter than this length (default: 1)")
    ap.add_argument("--num-special-tokens", type=int, default=5, help="Filter token ids < num_special_tokens from length")
    ap.add_argument("--clip", action="store_true", default=True, help="Clip lengths to [0, max-length] (default: true)")
    ap.add_argument("--no-clip", action="store_false", dest="clip", help="Do not clip; will error if any length > max-length")
    args = ap.parse_args()

    world_root = Path(args.world_root)
    out_json = Path(args.out_json)
    max_length = int(args.max_length)
    min_length = int(args.min_length)
    num_special = int(args.num_special_tokens)

    if max_length <= 0:
        raise ValueError("--max-length must be > 0")
    if min_length < 0:
        min_length = 0
    if num_special < 0:
        num_special = 0

    dists: Dict[str, Dict[str, object]] = {}
    total_count = 0

    for region_dir in _iter_region_dirs(world_root):
        region_id = region_dir.name
        lengths = _load_lengths_for_region(
            region_dir,
            num_special_tokens=num_special,
            max_length=max_length,
            min_length=min_length,
            clip=bool(args.clip),
        )
        if (not args.clip) and lengths.size > 0 and int(lengths.max()) > max_length:
            raise ValueError(f"{region_id}: found length {int(lengths.max())} > max_length={max_length}")
        probs, count = _hist_to_probs(lengths, max_length=max_length)
        dists[region_id] = {"probs": probs.tolist(), "count": int(count)}
        total_count += int(count)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dists, f, indent=2)

    print(f"[DONE] Wrote length distributions: {out_json}")
    print(f"       regions={len(dists)} total_traj_used={total_count} max_length={max_length} min_length={min_length}")


if __name__ == "__main__":
    main()

