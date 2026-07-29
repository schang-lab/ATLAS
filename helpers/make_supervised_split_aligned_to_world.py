#!/usr/bin/env python3
"""
Create a "selected-only" controlled split aligned to an ATLAS world.

This is useful for a strong supervised baseline: finetune on per-trajectory labels
(home/work/age/gender) but using exactly the same *subset* of trajectories that
the ATLAS world keeps (e.g., dropping missing demographics).

For Embee demo-group worlds produced by `make_llp_world_from_split.py` with:
  - region_mode=demo_group
  - max_per_region=None
  - keep_missing_demo=false
the selection is equivalent to:
  keep age_bin>=0 and gender_id>=0 (and demo group in cbgs.txt).

Example (train, change to your own folder paths):
  python3 helpers/make_supervised_split_aligned_to_world.py \\
    --world-root /path/to/atlas_world/world_train_demogroups \\
    --out-data-dir /path/to/atlas_world/supervised_splits/demogroups_aligned \\
    --split train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _parse_demo_region_id(region_id: str) -> Optional[Tuple[int, int]]:
    # Expected: "demo_a{age}_g{gender}"
    s = region_id.strip()
    if not s.startswith("demo_a") or "_g" not in s:
        return None
    try:
        # demo_a3_g1
        rest = s[len("demo_a") :]
        age_str, gender_str = rest.split("_g", 1)
        return int(age_str), int(gender_str)
    except Exception:
        return None


def _safe_rmtree(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists: {path} (pass --overwrite to replace)")
        shutil.rmtree(path)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copytree_if_exists(src_dir: Path, dst_dir: Path) -> None:
    if src_dir.is_dir():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def _load_world_metadata(world_root: Path) -> Dict[str, object]:
    meta_path = world_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing world metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_allowed_demo_pairs(world_root: Path) -> Set[Tuple[int, int]]:
    cbgs_txt = world_root / "cbgs.txt"
    if not cbgs_txt.exists():
        raise FileNotFoundError(f"Missing {cbgs_txt}")
    pairs: Set[Tuple[int, int]] = set()
    for ln in cbgs_txt.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parsed = _parse_demo_region_id(ln)
        if parsed is not None:
            pairs.add(parsed)
    return pairs


def _select_indices_from_controlled_split(
    controlled_dir: Path,
    *,
    allowed_demo_pairs: Optional[Set[Tuple[int, int]]],
    keep_missing_demo: bool,
) -> np.ndarray:
    attrs_path = controlled_dir / "all_attr_results_with_demo.npy"
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing {attrs_path}")
    attrs = np.load(attrs_path, allow_pickle=False)
    if attrs.ndim != 2 or attrs.shape[1] < 6:
        raise ValueError(f"Expected attrs shape [N,6], got {attrs.shape}")
    age = attrs[:, 4].astype(np.int64, copy=False)
    gender = attrs[:, 5].astype(np.int64, copy=False)

    if keep_missing_demo:
        mask = np.ones((attrs.shape[0],), dtype=bool)
    else:
        mask = (age >= 0) & (gender >= 0)

    if allowed_demo_pairs:
        # Apply demo-group whitelist from cbgs.txt (only meaningful for region_mode=demo_group).
        pairs_arr = np.stack([age, gender], axis=1)
        allowed = np.array(sorted(allowed_demo_pairs), dtype=np.int64)
        # Vectorized membership: mark true if row matches any allowed pair.
        # Since Embee has small number of demo groups, a simple loop is fine and clear.
        allowed_mask = np.zeros((attrs.shape[0],), dtype=bool)
        for a, g in allowed.tolist():
            allowed_mask |= (pairs_arr[:, 0] == a) & (pairs_arr[:, 1] == g)
        mask &= allowed_mask

    idx = np.where(mask)[0].astype(np.int64)
    return idx


def build_selected_controlled_split(
    *,
    controlled_dir: Path,
    out_controlled_dir: Path,
    selected_idx: np.ndarray,
    overwrite: bool,
) -> None:
    _safe_rmtree(out_controlled_dir, overwrite=overwrite)
    out_controlled_dir.mkdir(parents=True, exist_ok=True)

    # Load primary artifacts
    traj_path = controlled_dir / "final_segments_all_train_data.pkl"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")
    df = pd.read_pickle(traj_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {traj_path}, got {type(df)}")

    N = len(df)
    if selected_idx.size == 0:
        raise ValueError("Selected indices are empty.")
    if selected_idx.min() < 0 or selected_idx.max() >= N:
        raise ValueError(f"Selected indices out of bounds for {controlled_dir} (N={N}).")

    df_sel = df.iloc[selected_idx].reset_index(drop=True)
    df_sel.to_pickle(out_controlled_dir / "final_segments_all_train_data.pkl")

    # Subset numpy arrays if present
    for name in (
        "all_attr_results.npy",
        "all_attr_results_with_demo.npy",
        "all_dwell.npy",
        "all_timestamp.npy",
        "trajectory_length_ids.npy",
        "all_coordinates.npy",
    ):
        src = controlled_dir / name
        if not src.exists():
            continue
        arr = np.load(src, allow_pickle=True)
        if arr.shape[0] != N:
            raise ValueError(f"{src} first dim {arr.shape[0]} != df length {N}")
        np.save(out_controlled_dir / name, arr[selected_idx])

    # Copy metadata/static files
    _copy_if_exists(controlled_dir / "poi_map_feature.csv", out_controlled_dir / "poi_map_feature.csv")
    _copy_if_exists(controlled_dir / "split_summary.pkl", out_controlled_dir / "split_summary.pkl")
    _copy_if_exists(controlled_dir / "split_summary.txt", out_controlled_dir / "split_summary.txt")
    _copytree_if_exists(controlled_dir / "tokenizer", out_controlled_dir / "tokenizer")

    # Save the selected indices for traceability (indices into the *original* controlled split).
    np.save(out_controlled_dir / "selected_indices_in_original_split.npy", selected_idx.astype(np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-root", type=str, required=True, help="e.g. llp_world_embee/world_train_demogroups")
    ap.add_argument("--out-data-dir", type=str, required=True, help="Base output dir; writes <out-data-dir>/controlled/<split>")
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument(
        "--controlled-dir",
        type=str,
        default=None,
        help="Override the controlled split dir; default uses world_root/metadata.json:split_root",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    world_root = Path(args.world_root)
    out_data_dir = Path(args.out_data_dir)
    split = str(args.split)

    meta = _load_world_metadata(world_root)
    keep_missing_demo = bool(meta.get("keep_missing_demo", False))
    region_mode = str(meta.get("region_mode", ""))
    max_per_region = meta.get("max_per_region", None)

    if args.controlled_dir is None:
        split_root = str(meta.get("split_root", "")).strip()
        if not split_root:
            raise ValueError("metadata.json missing split_root; pass --controlled-dir explicitly.")
        controlled_dir = Path(split_root)
    else:
        controlled_dir = Path(args.controlled_dir)
    if not controlled_dir.exists():
        raise FileNotFoundError(f"controlled_dir not found: {controlled_dir}")

    allowed_pairs = None
    if region_mode == "demo_group":
        allowed_pairs = _load_allowed_demo_pairs(world_root)
    if max_per_region is not None:
        raise ValueError(
            "This world was built with max_per_region set; exact per-region alignment would require "
            "tracking original row indices. Rebuild the world with max_per_region=null, or extend the world "
            "builder to save split-row indices."
        )

    selected_idx = _select_indices_from_controlled_split(
        controlled_dir,
        allowed_demo_pairs=allowed_pairs,
        keep_missing_demo=keep_missing_demo,
    )

    out_controlled_dir = out_data_dir / "controlled" / split
    build_selected_controlled_split(
        controlled_dir=controlled_dir,
        out_controlled_dir=out_controlled_dir,
        selected_idx=selected_idx,
        overwrite=bool(args.overwrite),
    )

    summary = {
        "world_root": str(world_root),
        "controlled_dir": str(controlled_dir),
        "out_controlled_dir": str(out_controlled_dir),
        "split": split,
        "num_selected": int(selected_idx.size),
        "keep_missing_demo": keep_missing_demo,
        "region_mode": region_mode,
    }
    (out_controlled_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[DONE] Wrote supervised split aligned to world:")
    print(f"  out_data_dir: {out_data_dir}")
    print(f"  out_controlled_dir: {out_controlled_dir}")
    print(f"  num_selected: {int(selected_idx.size)}")


if __name__ == "__main__":
    main()

