#!/usr/bin/env python3
"""
Cache per-CBG conditioning tensors (home/work coordinates + demo labels).

Inputs (per CBG directory):
  - selected_indices.npy : indices matching the target marginal distribution
  - all_attr_results.demographics.npy : array [N, 6] -> [work_lat, work_lon, home_lat, home_lon, age_bin_id, gender_id]

Outputs (per CBG):
  - <out_dir>/<cbg>.npz with arrays:
        home:    [M, 2]  (lat, lon)
        work:    [M, 2]
        age_bin: [M]
        gender_id: [M]
        indices: [M] original indices for traceability
        demo_keys: [K] array of strings like "a{age}_g{gender}" (unique demos)
        demo_groups_ptr: [K+1] CSR pointers into demo_groups_idx
        demo_groups_idx: [M] concatenated subset indices per demo (0..M-1)
  - demo_marginals.csv summarizing empirical age/gender histograms per CBG
  - metadata.json describing the run

Config example (YAML):
  runtime:
    verbose: true
  llm_world:
    npy_root: /path/to/llm-demo-traj
    files:
      selected_indices: selected_indices.npy
      demographics: all_attr_results.demographics.npy
  groups:
    cbgs: []   # optional whitelist
  output:
    dir: /path/to/cache
    overwrite: false
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running this script from any working directory by adding `trajectory-generation/` to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

try:
    import yaml  # type: ignore
except Exception:
    print("ERROR: PyYAML is required. Please `pip install pyyaml`.", file=sys.stderr)
    raise


# -----------------------------
# Helpers
# -----------------------------
def _safe_mkdir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            return
    os.makedirs(path, exist_ok=True)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _list_subdirs(path: str) -> List[str]:
    try:
        names = []
        for entry in os.scandir(path):
            if entry.is_dir():
                names.append(entry.name)
        return sorted(names)
    except FileNotFoundError:
        return []


def _load_numpy(path: str, allow_pickle: bool = False):
    return np.load(path, allow_pickle=allow_pickle)


def _load_selected_indices(g_dir: str, files_cfg: Dict[str, str]) -> np.ndarray:
    p = os.path.join(g_dir, files_cfg.get("selected_indices", "selected_indices.npy"))
    if os.path.exists(p):
        return _load_numpy(p, allow_pickle=False).astype(np.int64)
    raise FileNotFoundError(f"selected_indices not found in {g_dir}")


def _load_demographics(g_dir: str, files_cfg: Dict[str, str]) -> np.ndarray:
    p = os.path.join(g_dir, files_cfg.get("demographics", "all_attr_results.demographics.npy"))
    if os.path.exists(p):
        return _load_numpy(p, allow_pickle=False)
    raise FileNotFoundError(f"demographics array not found in {g_dir}")


def _extract_condition_arrays(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Demographics array must be [N,6]; got {arr.shape}")
    work = arr[:, 0:2].astype(np.float32)
    home = arr[:, 2:4].astype(np.float32)
    age = arr[:, 4].astype(np.int64)
    gender = arr[:, 5].astype(np.int64)
    return home, work, age, gender


def _compute_demo_hist(cbg: str, age: np.ndarray, gender: np.ndarray) -> pd.DataFrame:
    if age.shape[0] == 0:
        return pd.DataFrame(columns=["cbg", "age_bin", "gender_id", "count", "prob"])
    combos = pd.DataFrame(
        {
            "age_bin": age,
            "gender_id": gender,
        }
    )
    grouped = combos.value_counts().reset_index(name="count")
    grouped["prob"] = grouped["count"] / grouped["count"].sum()
    grouped.insert(0, "cbg", cbg)
    return grouped


# -----------------------------
# Main routine
# -----------------------------
def _build_demo_groups(age: np.ndarray, gender: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR-style grouped indices for demos without Python loops.
    Returns (demo_keys, ptr, idx), where:
      - demo_keys: [K] unique demo strings "a{age}_g{gender}"
      - ptr: [K+1] pointers into idx
      - idx: [M] subset indices (0..M-1) concatenated per demo
    """
    if age.shape[0] == 0:
        return np.array([], dtype=object), np.array([0], dtype=np.int64), np.array([], dtype=np.int64)
    # Compose demo strings vectorized
    age_str = age.astype(str)
    gender_str = gender.astype(str)
    demo = np.char.add(np.char.add("a", age_str), np.char.add("_g", gender_str))
    # Group by unique demo
    keys, inverse = np.unique(demo, return_inverse=True)
    # Sort by group id so that indices for the same group are contiguous
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=keys.shape[0]).astype(np.int64)
    ptr = np.concatenate(([0], np.cumsum(counts)))
    return keys.astype(object), ptr, order.astype(np.int64)


def build_condition_cache(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    runtime_cfg = cfg.get("runtime", {}) or {}
    verbose = bool(runtime_cfg.get("verbose", True))
    llm_cfg = cfg.get("llm_world", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    groups_cfg = cfg.get("groups", {}) or {}

    npy_root = llm_cfg.get("npy_root")
    files_cfg = llm_cfg.get("files", {}) or {}
    out_dir = out_cfg.get("dir")
    overwrite = bool(out_cfg.get("overwrite", False))

    if not npy_root or not out_dir:
        raise ValueError("Config must set llm_world.npy_root and output.dir")
    _safe_mkdir(out_dir, overwrite=overwrite)

    configured_cbgs = [str(x) for x in (groups_cfg.get("cbgs", []) or [])]
    if configured_cbgs:
        cbg_entries = []
        for g in configured_cbgs:
            if os.path.isabs(g):
                cbg_entries.append((os.path.basename(os.path.normpath(g)) or g, g))
            else:
                cbg_entries.append((g, os.path.join(npy_root, g)))
    else:
        subdirs = _list_subdirs(npy_root)
        cbg_entries = [(name, os.path.join(npy_root, name)) for name in subdirs]
        if not cbg_entries and os.path.isdir(npy_root):
            cbg_name = os.path.basename(os.path.normpath(npy_root)) or "cbg"
            cbg_entries = [(cbg_name, npy_root)]
    if not cbg_entries:
        raise ValueError(f"No CBG directories found under {npy_root}")

    demo_histories: List[pd.DataFrame] = []
    written_files: List[str] = []

    for cbg_name, cbg_dir in cbg_entries:
        if not os.path.isdir(cbg_dir):
            if verbose:
                print(f"[WARN] Skip non-directory: {cbg_dir}")
            continue
        try:
            sel = _load_selected_indices(cbg_dir, files_cfg)
            demo_arr = _load_demographics(cbg_dir, files_cfg)
        except FileNotFoundError as exc:
            if verbose:
                print(f"[WARN] {exc}")
            continue

        N = demo_arr.shape[0]
        if np.any((sel < 0) | (sel >= N)):
            raise ValueError(f"CBG {cbg_name}: selected_indices out of bounds (max valid {N-1})")
        subset = demo_arr[sel]
        if subset.shape[0] == 0:
            if verbose:
                print(f"[INFO] CBG {cbg_name}: no selected trajectories, skipping")
            continue

        home, work, age, gender = _extract_condition_arrays(subset)
        demo_keys, demo_ptr, demo_idx = _build_demo_groups(age, gender)
        out_path = os.path.join(out_dir, f"{cbg_name}.npz")
        np.savez_compressed(
            out_path,
            cbg=cbg_name,
            home=home,
            work=work,
            age_bin=age,
            gender_id=gender,
            indices=sel.astype(np.int64),
            demo_keys=demo_keys,
            demo_groups_ptr=demo_ptr,
            demo_groups_idx=demo_idx,
        )
        written_files.append(out_path)

        demo_hist = _compute_demo_hist(cbg_name, age, gender)
        if not demo_hist.empty:
            demo_histories.append(demo_hist)

        if verbose:
            print(f"[INFO] Cached conditioning data for CBG {cbg_name}: {home.shape[0]} samples -> {out_path}")

    if not written_files:
        raise RuntimeError("No conditioning caches were written. Check config paths and selected indices.")

    demo_hist_df = pd.concat(demo_histories, ignore_index=True) if demo_histories else pd.DataFrame(
        columns=["cbg", "age_bin", "gender_id", "count", "prob"]
    )
    hist_path = os.path.join(out_dir, "demo_marginals.csv")
    demo_hist_df.to_csv(hist_path, index=False)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(config_path),
        "npy_root": os.path.abspath(npy_root),
        "num_cbgs": len(written_files),
        "outputs": {
            "cache_dir": os.path.abspath(out_dir),
            "demo_marginals": os.path.abspath(hist_path),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Wrote condition caches to {os.path.abspath(out_dir)} ({len(written_files)} CBGs)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache per-CBG conditioning tensors (home/work + demos).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_condition_cache(args.config)

