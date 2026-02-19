#!/usr/bin/env python3
"""
Create a "coord-null" copy of an LLP-world by zeroing out (work/home) coords.

Motivation:
  - Remove (home,work) confounding so you can test demo-only LLP / strong supervision.
  - Keep the POI sequences REAL (copied from the source world).

Input world layout:
  <world_root>/<region_id>/
    generated_sequences.npy
    all_attr_results.demographics.npy   # [N,6] -> [work_lat, work_lon, home_lat, home_lon, age_bin, gender_id]
    selected_indices.npy

Output world has the same layout, but attrs[:, :4] are replaced by either:
  - zeros (default), or
  - a constant coord vector you provide.

No PyYAML required.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-world-root", type=str, required=True)
    ap.add_argument("--out-world-root", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument(
        "--constant-coords",
        type=float,
        nargs=4,
        default=None,
        metavar=("WORK_LAT", "WORK_LON", "HOME_LAT", "HOME_LON"),
        help="If provided, use this constant coord vector instead of zeros.",
    )
    args = ap.parse_args()

    in_root = Path(args.in_world_root)
    out_root = Path(args.out_world_root)
    overwrite = bool(args.overwrite)

    if not in_root.exists():
        raise FileNotFoundError(f"Missing input world root: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy root metadata files if present.
    for fname in ["cbgs.txt", "region_summary.csv", "metadata.json"]:
        src = in_root / fname
        if src.exists():
            _copy_file(src, out_root / fname, overwrite=overwrite)

    # Process each region folder.
    region_dirs = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
    if not region_dirs:
        raise ValueError(f"No region dirs found under {in_root}")

    coord_vec: Optional[np.ndarray] = None
    if args.constant_coords is not None:
        coord_vec = np.asarray([float(x) for x in args.constant_coords], dtype=np.float32).reshape(1, 4)

    summaries: List[Dict[str, object]] = []
    cbg_ids: List[str] = []

    for region_dir in region_dirs:
        rid = region_dir.name
        seq_src = region_dir / "generated_sequences.npy"
        attr_src = region_dir / "all_attr_results.demographics.npy"
        sel_src = region_dir / "selected_indices.npy"
        if not (seq_src.exists() and attr_src.exists() and sel_src.exists()):
            continue

        out_region = out_root / rid
        out_region.mkdir(parents=True, exist_ok=True)

        _copy_file(seq_src, out_region / "generated_sequences.npy", overwrite=overwrite)
        _copy_file(sel_src, out_region / "selected_indices.npy", overwrite=overwrite)

        attrs = np.load(attr_src, allow_pickle=False).astype(np.float32, copy=False)
        if attrs.ndim != 2 or attrs.shape[1] < 6:
            raise ValueError(f"Bad attrs at {attr_src}: {attrs.shape}")
        out_attrs = attrs.copy()
        if coord_vec is None:
            out_attrs[:, :4] = 0.0
        else:
            out_attrs[:, :4] = coord_vec
        np.save(out_region / "all_attr_results.demographics.npy", out_attrs, allow_pickle=False)

        # Summaries (useful when region_summary.csv is missing or stale).
        age = out_attrs[:, 4].astype(np.int64)
        gender = out_attrs[:, 5].astype(np.int64)
        demo_pairs = [f"a{a}_g{g}" for a, g in zip(age.tolist(), gender.tolist())]
        demo_counts = pd.Series(demo_pairs).value_counts().to_dict()
        summaries.append(
            {
                "region_id": rid,
                "num_traj": int(out_attrs.shape[0]),
                "home_lat_min": float(np.min(out_attrs[:, 2])),
                "home_lat_max": float(np.max(out_attrs[:, 2])),
                "home_lon_min": float(np.min(out_attrs[:, 3])),
                "home_lon_max": float(np.max(out_attrs[:, 3])),
                "demo_counts": {str(k): int(v) for k, v in demo_counts.items()},
            }
        )
        cbg_ids.append(rid)

    if cbg_ids:
        # Always write fresh summary + cbgs list for the output world.
        pd.DataFrame(summaries).to_csv(out_root / "region_summary.csv", index=False)
        (out_root / "cbgs.txt").write_text("\n".join(cbg_ids) + "\n", encoding="utf-8")

    meta = {
        "in_world_root": str(in_root),
        "out_world_root": str(out_root),
        "coords_mode": "constant" if coord_vec is not None else "zeros",
        "constant_coords": None if coord_vec is None else [float(x) for x in args.constant_coords],
        "regions_written": int(len(cbg_ids)),
    }
    (out_root / "coords_null_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote coord-null world to {out_root} (regions={len(cbg_ids)})")


if __name__ == "__main__":
    main()

