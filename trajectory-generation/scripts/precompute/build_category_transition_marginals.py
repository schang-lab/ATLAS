#!/usr/bin/env python3
"""
Build CBG -> category transition marginals for weak supervision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

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

from src.data import CategoryMapSpec, POICategoryMap


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


def _load_numpy(path: str, allow_pickle: bool = True):
    return np.load(path, allow_pickle=allow_pickle)


def _load_selected_indices(g_dir: str, files_cfg: Dict[str, str]) -> np.ndarray:
    p = os.path.join(g_dir, files_cfg.get("selected_indices", "selected_indices.npy"))
    if os.path.exists(p):
        return np.asarray(_load_numpy(p, allow_pickle=False), dtype=np.int64)
    raise FileNotFoundError(f"selected_indices not found in {g_dir}")


def _load_sequences_for_cbg(g_dir: str, files_cfg: Dict[str, str]) -> np.ndarray:
    p = os.path.join(g_dir, files_cfg.get("poi_sequences", "generated_sequences.npy"))
    if not os.path.exists(p):
        raise FileNotFoundError(f"generated sequences not found in {g_dir} (expected {p})")
    arr = _load_numpy(p, allow_pickle=True)
    return arr


def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(vec.sum())
    if s <= 0:
        return np.full_like(vec, 1.0 / max(vec.size, 1), dtype=np.float64)
    return (vec + eps) / (s + eps * vec.size)


def build_category_transition_marginals(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    verbose = bool(cfg.get("runtime", {}).get("verbose", True))
    np.random.seed(int(cfg.get("runtime", {}).get("seed", 42)))

    llm_cfg = cfg["llm_world"]
    out_cfg = cfg.get("output", {}) or {}
    cat_cfg = cfg.get("category_map", {}) or {}

    npy_root = str(llm_cfg.get("npy_root", ""))
    files_cfg = llm_cfg.get("files", {}) or {}
    vocab_path = str(llm_cfg.get("vocab_path", ""))
    num_special = int(llm_cfg.get("num_special_tokens", 0))
    if not npy_root or not vocab_path:
        raise ValueError("Config must set llm_world.npy_root and llm_world.vocab_path")

    map_csv = str(cat_cfg.get("csv_path", ""))
    cat_col = str(cat_cfg.get("category_column", "top_category"))
    drop_poi_tokens = cat_cfg.get("drop_poi_tokens", None)
    if drop_poi_tokens is None:
        drop_poi_tokens = ["POI_HOME", "POI_WORK", "POI_OTHER"]
    drop_poi_tokens = tuple(str(x) for x in list(drop_poi_tokens))
    if not map_csv:
        raise ValueError("Config must set category_map.csv_path (poi_map_feature.csv)")

    out_dir = str(out_cfg.get("dir", ""))
    overwrite = bool(out_cfg.get("overwrite", True))
    if not out_dir:
        raise ValueError("Config must set output.dir")
    _safe_mkdir(out_dir, overwrite=overwrite)

    # Build POI->category mapping in POI-only indexing (after removing num_special tokens)
    cat_map = POICategoryMap(
        CategoryMapSpec(
            poi_map_csv=map_csv,
            vocab_path=vocab_path,
            num_special_tokens=num_special,
            category_column=cat_col,
            drop_poi_tokens=drop_poi_tokens,
        )
    )
    categories = cat_map.categories
    C = cat_map.num_categories()
    K = C * C

    subdirs = _list_subdirs(npy_root)
    if not subdirs and os.path.isdir(npy_root):
        # Allow pointing directly at a single CBG dir
        subdirs = [os.path.basename(os.path.normpath(npy_root)) or "cbg"]
        npy_root = os.path.dirname(os.path.normpath(npy_root))

    if not subdirs:
        raise ValueError(f"No CBG subfolders found under {npy_root}")

    sum_by_cbg: Dict[str, np.ndarray] = {}
    traj_count_by_cbg: Dict[str, int] = {}

    for cbg in subdirs:
        g_dir = os.path.join(npy_root, cbg)
        if not os.path.isdir(g_dir):
            continue
        try:
            sel = _load_selected_indices(g_dir, files_cfg)
            seqs_all = _load_sequences_for_cbg(g_dir, files_cfg)
        except FileNotFoundError as exc:
            if verbose:
                print(f"[WARN] {exc}")
            continue

        if seqs_all is None:
            continue

        # Make indexable list/array of sequences.
        if isinstance(seqs_all, np.ndarray) and seqs_all.dtype == object:
            seqs = [np.asarray(seqs_all[i], dtype=np.int64) for i in sel.tolist()]
        elif isinstance(seqs_all, np.ndarray) and seqs_all.ndim == 2:
            seqs = [np.asarray(seqs_all[i], dtype=np.int64) for i in sel.tolist()]
        else:
            # best-effort fallback
            seqs = [np.asarray(seqs_all[i], dtype=np.int64) for i in sel.tolist()]

        used = 0
        for tok in seqs:
            # Filter to POI-only ids, matching build_poi_marginals behavior
            tok = np.asarray(tok, dtype=np.int64)
            mask = (tok >= num_special)
            if not np.any(mask):
                continue
            poi_ids = tok[mask] - num_special  # [L_poi]
            poi_ids = poi_ids[(poi_ids >= 0) & (poi_ids < cat_map.poi_vocab_size())]
            if poi_ids.size < 2:
                continue

            cat_ids = cat_map._poi_to_cat_np[poi_ids]  # pylint: disable=protected-access
            cat_ids = cat_ids[cat_ids >= 0]
            if cat_ids.size < 2:
                continue

            pairs = cat_ids[:-1] * C + cat_ids[1:]
            cnt = np.bincount(pairs.astype(np.int64, copy=False), minlength=K).astype(np.float64, copy=False)
            if cnt.sum() <= 0:
                continue
            s_vec = cnt / cnt.sum()
            if cbg not in sum_by_cbg:
                sum_by_cbg[cbg] = np.zeros(K, dtype=np.float64)
                traj_count_by_cbg[cbg] = 0
            sum_by_cbg[cbg] += s_vec
            traj_count_by_cbg[cbg] += 1
            used += 1

        if verbose:
            print(f"[INFO] {cbg}: selected={len(seqs)} used={used}")

    if not sum_by_cbg:
        raise RuntimeError("No transition aggregates collected. Check mapping/vocab alignment and inputs.")

    cbgs = sorted(sum_by_cbg.keys())
    probs = np.zeros((len(cbgs), K), dtype=np.float32)
    traj_counts = np.zeros((len(cbgs),), dtype=np.int64)
    for i, cbg in enumerate(cbgs):
        probs[i] = _normalize_vector(sum_by_cbg[cbg]).astype(np.float32, copy=False)
        traj_counts[i] = int(traj_count_by_cbg.get(cbg, 0))

    out_npz = os.path.join(out_dir, "p_cat_transition.npz")
    np.savez_compressed(
        out_npz,
        cbgs=np.asarray(cbgs, dtype=object),
        categories=np.asarray(categories, dtype=object),
        probs=probs,
        traj_count=traj_counts,
    )

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(config_path),
        "npy_root": os.path.abspath(cfg["llm_world"]["npy_root"]),
        "vocab_path": os.path.abspath(vocab_path),
        "num_special_tokens": int(num_special),
        "poi_map_feature_csv": os.path.abspath(map_csv),
        "category_column": cat_col,
        "num_cbgs": int(len(cbgs)),
        "num_categories": int(C),
        "outputs": {"p_cat_transition_npz": os.path.abspath(out_npz)},
    }
    with open(os.path.join(out_dir, "p_cat_transition.metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Wrote category-transition marginals: {out_npz}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CBG -> category transition marginals (npz).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_category_transition_marginals(args.config)


