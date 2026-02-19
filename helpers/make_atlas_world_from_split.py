#!/usr/bin/env python3
"""
Build an LLP-world directory from Carlos `split_data_carlos_w1_demo`.

Writes a directory layout consumable by:
  - trajectory-generation/scripts/build_poi_marginals.py
  - trajectory-generation/scripts/precompute/cache_cbg_conditionals.py

This script does NOT require PyYAML.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SPECIAL_TOKENS_DEFAULT = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def _read_vocab(vocab_path: Path) -> Tuple[Dict[str, int], int]:
    lines = vocab_path.read_text(encoding="utf-8").splitlines()
    lines = [ln.rstrip("\n") for ln in lines]
    # Heuristic: treat as "token id" file only if (a) every line has 2 fields and (b) the 2nd parses as int.
    as_pairs = True
    pairs: List[Tuple[str, int]] = []
    for ln in lines[:200]:
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 2:
            as_pairs = False
            break
        try:
            _ = int(parts[1])
        except Exception:
            as_pairs = False
            break
    if as_pairs:
        for ln in lines:
            if not ln:
                continue
            tok, idx = ln.split()[0], int(ln.split()[1])
            pairs.append((tok, idx))
        vocab = {tok: idx for tok, idx in pairs}
    else:
        vocab = {tok: i for i, tok in enumerate(lines) if tok}

    unk_id = vocab.get("[UNK]")
    if unk_id is None:
        unk_id = vocab.get("<unk>")
    if unk_id is None:
        raise ValueError("Could not find [UNK] or <unk> in vocab.")
    return vocab, int(unk_id)


def _parse_sequence_value(value) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        # Prefer parsing list-like repr (keeps tokens with spaces intact).
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        # If it's a plain token string (can contain spaces), do NOT split.
        return [s]
    return [str(value)]


def _trim_by_attention_mask(tokens: List[str], attn_mask_value) -> List[str]:
    if attn_mask_value is None:
        return tokens
    if isinstance(attn_mask_value, np.ndarray):
        mask = attn_mask_value.tolist()
    elif isinstance(attn_mask_value, (list, tuple)):
        mask = list(attn_mask_value)
    else:
        return tokens
    try:
        valid_len = int(sum(int(x) for x in mask))
    except Exception:
        return tokens
    if valid_len <= 0:
        return []
    return tokens[:valid_len]


def _tokens_to_ids(tokens: Sequence[str], vocab: Dict[str, int], unk_id: int) -> Tuple[np.ndarray, int]:
    out: List[int] = []
    missing = 0
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is None:
            missing += 1
            idx = unk_id
        out.append(int(idx))
    return np.asarray(out, dtype=np.int64), int(missing)


@dataclass(frozen=True)
class LoadedSplit:
    sequences_tok: List[List[str]]
    sequences_ids: List[np.ndarray]
    attrs: np.ndarray  # [N,6] float32
    missing_token_count: int


def load_split(split_root: Path, vocab_path: Path) -> LoadedSplit:
    pkl_path = split_root / "final_segments_all_train_data.pkl"
    attrs_path = split_root / "all_attr_results_with_demo.npy"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}")
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing {attrs_path}")

    import pickle

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {pkl_path}, got {type(df)}")

    attrs = np.load(attrs_path).astype(np.float32, copy=False)
    if attrs.ndim != 2 or attrs.shape[1] < 6:
        raise ValueError(f"Expected attrs shape [N,6], got {attrs.shape}")
    if len(df) != attrs.shape[0]:
        raise ValueError(f"Row mismatch: pkl rows={len(df)} vs attrs rows={attrs.shape[0]}")

    vocab, unk_id = _read_vocab(vocab_path)

    sequences_tok: List[List[str]] = []
    sequences_ids: List[np.ndarray] = []
    missing_total = 0

    has_attn = "attention_mask" in df.columns
    for i in range(len(df)):
        raw = df.iloc[i]["unique_id_seq"]
        toks = _parse_sequence_value(raw)
        if has_attn:
            toks = _trim_by_attention_mask(toks, df.iloc[i]["attention_mask"])
        sequences_tok.append(toks)
        ids, missing = _tokens_to_ids(toks, vocab, unk_id)
        sequences_ids.append(ids)
        missing_total += missing

    return LoadedSplit(
        sequences_tok=sequences_tok,
        sequences_ids=sequences_ids,
        attrs=attrs,
        missing_token_count=missing_total,
    )


def _region_keys_home_grid(attrs: np.ndarray, lat_bins: int, lon_bins: int) -> Tuple[np.ndarray, List[str]]:
    home_lat = attrs[:, 2]
    home_lon = attrs[:, 3]
    # Use Series so qcut returns a Series with .isna and stable dtype handling.
    lat_bin = pd.qcut(pd.Series(home_lat), q=lat_bins, labels=False, duplicates="drop")
    lon_bin = pd.qcut(pd.Series(home_lon), q=lon_bins, labels=False, duplicates="drop")
    if lat_bin.isna().any() or lon_bin.isna().any():  # type: ignore[union-attr]
        raise ValueError("NaNs in qcut binning; check coordinate inputs.")
    lat_bin = lat_bin.astype(int).to_numpy()  # type: ignore[union-attr]
    lon_bin = lon_bin.astype(int).to_numpy()  # type: ignore[union-attr]
    n_lat = int(lat_bin.max() + 1) if lat_bin.size else 0
    n_lon = int(lon_bin.max() + 1) if lon_bin.size else 0
    keys = np.array([f"homegrid_r{r}_c{c}" for r, c in zip(lat_bin.tolist(), lon_bin.tolist())], dtype=object)
    info = [f"home_grid bins lat={n_lat} lon={n_lon} (requested lat={lat_bins} lon={lon_bins})"]
    return keys, info


def _region_keys_demo_group(attrs: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    age = attrs[:, 4].astype(int)
    gender = attrs[:, 5].astype(int)
    keys = np.array([f"demo_a{a}_g{g}" for a, g in zip(age.tolist(), gender.tolist())], dtype=object)
    info = ["demo_group regions = (age_bin, gender_id)"]
    return keys, info


def _write_region(
    out_dir: Path,
    region_id: str,
    indices: np.ndarray,
    sequences_ids: Sequence[np.ndarray],
    attrs: np.ndarray,
    *,
    max_per_region: Optional[int],
    seed: int,
) -> Dict[str, object]:
    rs = np.random.RandomState(seed)
    indices = np.asarray(indices, dtype=np.int64)
    if max_per_region is not None and indices.size > int(max_per_region):
        chosen = rs.choice(indices, size=int(max_per_region), replace=False)
        indices = np.sort(chosen.astype(np.int64))

    region_path = out_dir / region_id
    region_path.mkdir(parents=True, exist_ok=True)

    seqs = [sequences_ids[i] for i in indices.tolist()]
    # Store ragged sequences as object array for np.save.
    np.save(region_path / "generated_sequences.npy", np.array(seqs, dtype=object), allow_pickle=True)
    np.save(region_path / "all_attr_results.demographics.npy", attrs[indices].astype(np.float32, copy=False))
    np.save(region_path / "selected_indices.npy", np.arange(len(seqs), dtype=np.int64))

    # Simple summary stats
    region_attrs = attrs[indices]
    age = region_attrs[:, 4].astype(int)
    gender = region_attrs[:, 5].astype(int)
    demo_pairs = list(zip(age.tolist(), gender.tolist()))
    counts = pd.Series(demo_pairs).value_counts().sort_index()
    demo_json = {f"a{a}_g{g}": int(n) for (a, g), n in counts.items()}

    return {
        "region_id": region_id,
        "num_traj": int(len(seqs)),
        "home_lat_min": float(np.min(region_attrs[:, 2])),
        "home_lat_max": float(np.max(region_attrs[:, 2])),
        "home_lon_min": float(np.min(region_attrs[:, 3])),
        "home_lon_max": float(np.max(region_attrs[:, 3])),
        "demo_counts": demo_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", type=str, required=True, help="e.g. /path/to/YOUR_DATA_FOLDER/controlled/train")
    ap.add_argument("--tokenizer-vocab", type=str, required=True, help="e.g. /path/to/YOUR_DATA_FOLDER/controlled/tokenizer/vocab.txt")
    ap.add_argument("--out-root", type=str, required=True, help="Output world root, e.g. /path/to/atlas_world/world_train_demogroups")
    ap.add_argument("--region-mode", type=str, default="home_grid", choices=["home_grid", "demo_group"])
    ap.add_argument("--lat-bins", type=int, default=3)
    ap.add_argument("--lon-bins", type=int, default=3)
    ap.add_argument("--min-region-size", type=int, default=500)
    ap.add_argument("--max-per-region", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-missing-demo", action="store_true", help="Keep rows with age/gender == -1 (not recommended for ATLAS world training).")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    vocab_path = Path(args.tokenizer_vocab)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    loaded = load_split(split_root, vocab_path)
    attrs = loaded.attrs

    # Filter missing demo by default (Carlos uses -1 for missing).
    if args.keep_missing_demo:
        keep_mask = np.ones((attrs.shape[0],), dtype=bool)
    else:
        keep_mask = (attrs[:, 4] >= 0) & (attrs[:, 5] >= 0)

    kept_idx = np.where(keep_mask)[0].astype(np.int64)
    if kept_idx.size == 0:
        raise RuntimeError("No trajectories remain after filtering.")

    attrs_kept = attrs[kept_idx]
    seqs_kept = [loaded.sequences_ids[i] for i in kept_idx.tolist()]

    # Compute region keys on kept subset.
    if args.region_mode == "home_grid":
        keys, info = _region_keys_home_grid(attrs_kept, int(args.lat_bins), int(args.lon_bins))
    else:
        keys, info = _region_keys_demo_group(attrs_kept)

    # Group indices by region key (indices are in 0..len(kept)-1 coordinates here).
    key_series = pd.Series(keys.astype(str))
    groups = key_series.groupby(key_series).groups  # dict region_id -> Int64Index (positions)

    summaries: List[Dict[str, object]] = []
    written_regions: List[str] = []
    dropped_small = 0

    for region_id, pos_idx in sorted(groups.items(), key=lambda x: x[0]):
        pos = np.asarray(list(pos_idx), dtype=np.int64)
        if pos.size < int(args.min_region_size):
            dropped_small += 1
            continue
        summary = _write_region(
            out_dir=out_root,
            region_id=str(region_id),
            indices=pos,
            sequences_ids=seqs_kept,
            attrs=attrs_kept,
            max_per_region=args.max_per_region,
            seed=int(args.seed),
        )
        summaries.append(summary)
        written_regions.append(str(region_id))

    if not written_regions:
        raise RuntimeError("No regions written (all dropped by min-region-size?).")

    # Write summary artifacts at world root.
    pd.DataFrame(summaries).to_csv(out_root / "region_summary.csv", index=False)
    (out_root / "cbgs.txt").write_text("\n".join(written_regions) + "\n", encoding="utf-8")

    meta = {
        "split_root": str(split_root),
        "tokenizer_vocab": str(vocab_path),
        "out_root": str(out_root),
        "region_mode": str(args.region_mode),
        "region_info": info,
        "lat_bins_requested": int(args.lat_bins),
        "lon_bins_requested": int(args.lon_bins),
        "min_region_size": int(args.min_region_size),
        "max_per_region": None if args.max_per_region is None else int(args.max_per_region),
        "seed": int(args.seed),
        "keep_missing_demo": bool(args.keep_missing_demo),
        "input_rows": int(attrs.shape[0]),
        "kept_rows": int(kept_idx.size),
        "missing_token_total_replaced_with_unk": int(loaded.missing_token_count),
        "regions_written": int(len(written_regions)),
        "regions_dropped_too_small": int(dropped_small),
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Wrote LLP world to {out_root} with {len(written_regions)} regions.")
    print(f"       Summary: {out_root/'region_summary.csv'}")
    print(f"       Region ids: {out_root/'cbgs.txt'}")


if __name__ == "__main__":
    main()
