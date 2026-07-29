#!/usr/bin/env python3
"""
Create a "coord-null" version of a Embee controlled split by zeroing out (work/home) coords.

This is useful when you want:
  - unconditional pretraining on controlled POI sequences (attrs all zeros), and/or
  - demo-only finetuning (age/gender conditioning) without home/work confounding.

It copies the controlled split folder structure:
  <in_root>/controlled/<split>/
    final_segments_all_train_data.pkl
    all_attr_results.npy                  # [N,4] coords
    all_attr_results_with_demo.npy        # [N,6] coords + demo
    tokenizer/vocab.txt
    poi_map_feature.csv
    (other files are copied if present)

and writes:
  <out_root>/controlled/<split>/... with the same files, but:
    - all_attr_results.npy coords set to 0
    - all_attr_results_with_demo.npy first 4 dims set to 0 (demo kept)

No PyYAML required.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _zero_coords(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] < 4:
        raise ValueError(f"Expected [N,>=4], got {out.shape}")
    out[:, :4] = 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, required=True, help="split_data root (contains controlled/{train,val,test})")
    ap.add_argument("--out-root", type=str, required=True, help="Output split_data root")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    split = str(args.split)
    overwrite = bool(args.overwrite)

    in_dir = in_root / "controlled" / split
    out_dir = out_root / "controlled" / split
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy everything first (except attrs we will rewrite).
    for item in in_dir.iterdir():
        if item.name in {"all_attr_results.npy", "all_attr_results_with_demo.npy"}:
            continue
        dst = out_dir / item.name
        if item.is_dir():
            _copy_tree(item, dst, overwrite=overwrite)
        else:
            _copy_file(item, dst, overwrite=overwrite)

    # Rewrite attrs files.
    p4 = in_dir / "all_attr_results.npy"
    p6 = in_dir / "all_attr_results_with_demo.npy"
    if not p4.exists():
        raise FileNotFoundError(f"Missing {p4}")
    if not p6.exists():
        raise FileNotFoundError(f"Missing {p6}")

    arr4 = np.load(p4, allow_pickle=False)
    arr6 = np.load(p6, allow_pickle=False)
    out4 = _zero_coords(arr4)
    out6 = _zero_coords(arr6)

    np.save(out_dir / "all_attr_results.npy", out4, allow_pickle=False)
    np.save(out_dir / "all_attr_results_with_demo.npy", out6, allow_pickle=False)

    print(f"[DONE] Wrote coord-null split to {out_dir}")


if __name__ == "__main__":
    main()

