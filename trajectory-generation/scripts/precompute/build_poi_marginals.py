#!/usr/bin/env python3
"""
Build CBG -> POI marginals for weakly-supervised trajectory generation.

Outputs:
  - p_poi.csv: rows (cbg, demo, poi_index, vocab_idx, token, weight_sum, prob, traj_count)
  - p_dem.csv: rows (cbg, demo, traj_count, prob)
  - metadata.json: run info and parameters

Config: trajectory-generation/configs/poi_marginals.yaml
CLI:
  python trajectory-generation/scripts/build_poi_marginals.py --config trajectory-generation/configs/poi_marginals.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

# Allow running this script from any working directory by adding `trajectory-generation/` to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

try:
    import yaml  # type: ignore
except Exception as exc:
    print("ERROR: PyYAML is required to load the config. Please `pip install pyyaml`.", file=sys.stderr)
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


def _write_parquet_or_csv(df: pd.DataFrame, out_path: str) -> str:
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".parquet":
        # Respect the provided extension
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            df.to_csv(out_path, index=False)
            return out_path
    else:
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            csv_path = base + ".csv"
            df.to_csv(csv_path, index=False)
            return csv_path


def _update_aggregates(
    sum_by_cbg: Dict[Tuple[str, str], np.ndarray],
    traj_count_by_cbg: Dict[Tuple[str, str], int],
    key: Tuple[str, str],
    s_vec: np.ndarray,
) -> None:
    if key not in sum_by_cbg:
        sum_by_cbg[key] = np.zeros_like(s_vec, dtype=np.float64)
        traj_count_by_cbg[key] = 0
    sum_by_cbg[key] += s_vec
    traj_count_by_cbg[key] += 1


def _finalize_p_poi(
    sum_by_cbg: Dict[Tuple[str, str], np.ndarray],
    traj_count_by_cbg: Dict[Tuple[str, str], int],
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for (g, d), vec in sum_by_cbg.items():
        total = vec.sum()
        if total <= 0:
            # Smooth to uniform to avoid degenerate zeros
            prob = np.full_like(vec, 1.0 / max(len(vec), 1), dtype=np.float64)
        else:
            prob = (vec + epsilon) / np.clip(total + epsilon * vec.size, 1e-12, None)
        traj_count = traj_count_by_cbg[(g, d)]
        for p_idx, p_prob in enumerate(prob):
            rows.append(
                {
                    "cbg": g,
                    "demo": d,
                    "poi_index": p_idx,
                    "weight_sum": float(vec[p_idx]),
                    "prob": float(p_prob),
                    "traj_count": int(traj_count),
                }
            )
    return pd.DataFrame(rows)


def _finalize_p_dem(
    traj_count_by_cbg: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    total = max(sum(traj_count_by_cbg.values()), 1)
    rows = []
    for (g, d), n in traj_count_by_cbg.items():
        rows.append(
            {
                "cbg": g,
                "demo": d,
                "traj_count": int(n),
                "prob": float(n / total),
            }
        )
    return pd.DataFrame(rows)

#
# ---- NPY / Vocab helpers ----
#
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


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_vocab(vocab_path: str) -> List[str]:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    with open(vocab_path, "r") as f:
        toks = [line.strip() for line in f.readlines()]
    return toks

def _load_selected_indices(g_dir: str, files_cfg: Dict[str, str]) -> np.ndarray:
    # primary key
    p = os.path.join(g_dir, files_cfg.get("selected_indices", "selected_indices.npy"))
    if os.path.exists(p):
        arr = _load_numpy(p, allow_pickle=False)
        return np.asarray(arr, dtype=np.int64)
    # common misspelling fallback
    alt = os.path.join(g_dir, "selected.npy")
    if os.path.exists(alt):
        arr = _load_numpy(alt, allow_pickle=False)
        return np.asarray(arr, dtype=np.int64)
    raise FileNotFoundError(f"selected_indices not found in {g_dir}")


def _has_required_cbg_files(g_dir: str, files_cfg: Dict[str, str]) -> bool:
    """
    Lightweight check to see if a directory looks like a CBG folder.
    Currently we only require selected_indices.* to exist because the rest
    of the loading code will raise clearer errors later if missing.
    """
    sel_path = os.path.join(g_dir, files_cfg.get("selected_indices", "selected_indices.npy"))
    return os.path.exists(sel_path)


def _infer_demo_from_demographics_array(arr: np.ndarray) -> np.ndarray:
    """
    Infer demographic labels from demographics array.
    Expected format: N x 6 [work_lat, work_lon, home_lat, home_lon, age_bin_id, gender_id]
    Returns: array of strings like "a{age_bin}_g{gender}"
    """
    if arr.ndim == 1:
        # Single sample: [work_lat, work_lon, home_lat, home_lon, age_bin_id, gender_id]
        if arr.shape[0] == 6:
            age_bin = int(arr[4])
            gender = int(arr[5])
            return np.array([f"a{age_bin}_g{gender}"])
        else:
            # Fallback: treat as single label
            return np.array([str(x) for x in arr.tolist()])
    if arr.ndim == 2:
        if arr.shape[1] == 6:
            # N x 6 format: [work_lat, work_lon, home_lat, home_lon, age_bin_id, gender_id]
            age_bins = arr[:, 4].astype(int)
            genders = arr[:, 5].astype(int)
            return np.array([f"a{a}_g{g}" for a, g in zip(age_bins.tolist(), genders.tolist())])

    raise ValueError(f"Unsupported demographics array shape: {arr.shape}")


def _infer_demo_from_sampled_attributes(
    arr: np.ndarray,
    age_key: str,
    gender_key: str,
) -> np.ndarray:
    def _compose(age_arr: np.ndarray, gender_arr: np.ndarray) -> np.ndarray:
        age_arr = age_arr.astype(int)
        gender_arr = gender_arr.astype(int)
        return np.array([f"a{a}_g{g}" for a, g in zip(age_arr.tolist(), gender_arr.tolist())])

    if arr.dtype.fields is not None:
        if (age_key in arr.dtype.fields) and (gender_key in arr.dtype.fields):
            return _compose(arr[age_key], arr[gender_key])
    if arr.dtype == object:
        out: List[str] = []
        for x in arr.tolist():
            if isinstance(x, dict) and (age_key in x) and (gender_key in x):
                out.append(f"a{int(x[age_key])}_g{int(x[gender_key])}")
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append(f"a{int(x[0])}_g{int(x[1])}")
            else:
                out.append(str(x))
        return np.array(out)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return _compose(arr[:, 0], arr[:, 1])
    return np.array([str(x) for x in arr.tolist()])


def _load_sequences_for_cbg(
    g_dir: str,
    files_cfg: Dict[str, str],
) -> List[np.ndarray]:
    """
    Load POI token sequences for a CBG:
      - Try poi_sequences (npy) -> expected [N, L] ints or object array of lists
      - Else poi_sequences_pkl (list of lists or np.ndarray)
      - Else sequences (npy) then filter tokens later by vocab
    Returns a list of 1D np.ndarray[int] for each trajectory.
    """
    # Prefer npy poi_sequences
    if "poi_sequences" in files_cfg:
        p = os.path.join(g_dir, files_cfg["poi_sequences"])
        if os.path.exists(p):
            arr = _load_numpy(p, allow_pickle=True)
            if arr.dtype == object:
                return [np.asarray(x, dtype=np.int64) for x in arr.tolist()]
            elif arr.ndim == 2:
                return [arr[i].astype(np.int64, copy=False) for i in range(arr.shape[0])]
    # Fallback: PKL
    if "poi_sequences_pkl" in files_cfg:
        p = os.path.join(g_dir, files_cfg["poi_sequences_pkl"])
        if os.path.exists(p):
            obj = _load_pkl(p)
            if isinstance(obj, list):
                return [np.asarray(x, dtype=np.int64) for x in obj]
            elif isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    return [np.asarray(x, dtype=np.int64) for x in obj.tolist()]
                elif obj.ndim == 2:
                    return [obj[i].astype(np.int64, copy=False) for i in range(obj.shape[0])]
    # Fallback: general sequences
    p = os.path.join(g_dir, files_cfg.get("sequences", "generated_sequences.npy"))
    if os.path.exists(p):
        arr = _load_numpy(p, allow_pickle=False)
        if arr.ndim == 2:
            return [arr[i].astype(np.int64, copy=False) for i in range(arr.shape[0])]
        if arr.dtype == object:
            return [np.asarray(x, dtype=np.int64) for x in arr.tolist()]
    raise FileNotFoundError(f"No sequences found in {g_dir} (checked poi_sequences, pkl, sequences).")


def _load_demo_labels_for_cbg(
    g_dir: str,
    files_cfg: Dict[str, str],
    demo_source: str,
    attr_keys: Dict[str, object],
) -> np.ndarray:
    demo_arr: Optional[np.ndarray] = None
    if demo_source in ("auto", "demographics"):
        p = os.path.join(g_dir, files_cfg.get("demographics", "all_attr_results.demographics.npy"))
        if os.path.exists(p):
            arr = _load_numpy(p, allow_pickle=True)
            demo_arr = _infer_demo_from_demographics_array(arr)
    if demo_arr is None and demo_source in ("auto", "sampled_attributes"):
        p = os.path.join(g_dir, files_cfg.get("sampled_attributes", "sampled_attributes.npy"))
        if os.path.exists(p):
            arr = _load_numpy(p, allow_pickle=True)
            demo_arr = _infer_demo_from_sampled_attributes(
                arr,
                str(attr_keys.get("age_key", "age_id")),
                str(attr_keys.get("gender_key", "gender_id")),
            )
    if demo_arr is None:
        raise RuntimeError(f"No demo labels found in {g_dir} (demographics/sample_attributes).")
    return demo_arr



def build_marginals(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    verbose = bool(cfg.get("runtime", {}).get("verbose", True))
    np.random.seed(int(cfg.get("runtime", {}).get("seed", 42)))

    llm_cfg = cfg["llm_world"]
    groups_cfg = cfg.get("groups", {}) or {}
    stats_cfg = cfg.get("stats", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    npy_root = llm_cfg.get("npy_root", None)
    demo_source = str(llm_cfg.get("demo_source", "auto")).lower()
    attr_keys = llm_cfg.get("attr_keys", {}) or {}
    epsilon = float(stats_cfg.get("epsilon", 1e-6))
    min_traj_per_group = int(stats_cfg.get("min_traj_per_group", 1))
    out_dir = str(out_cfg.get("dir"))
    overwrite = bool(out_cfg.get("overwrite", True))
    if not out_dir:
        raise ValueError("output.dir must be set in config")
    _safe_mkdir(out_dir, overwrite=overwrite)

    # Read selected_indices.npy, subset POI sequences + demographics, aggregate.
    files_cfg = llm_cfg.get("files", {}) or {}
    if not (npy_root and ("vocab_path" in llm_cfg) and ("selected_indices" in files_cfg)):
        raise ValueError(
            "Config must specify llm_world.npy_root, llm_world.vocab_path, "
            "and files.selected_indices. Please update configs/poi_marginals.yaml."
        )

    vocab_path = str(llm_cfg["vocab_path"])
    num_special = int(llm_cfg.get("num_special_tokens", 5))
    vocab_tokens = _load_vocab(vocab_path)
    V = len(vocab_tokens)
    if V <= num_special:
        raise ValueError(f"num_special_tokens ({num_special}) >= vocab size ({V})")
    V_eff = V - num_special
    if verbose:
        print(f"[INFO] Vocab loaded: {V} tokens, excluding first {num_special} -> {V_eff} POI tokens")

    configured_cbgs = [str(x) for x in (groups_cfg.get("cbgs", []) or [])]
    cbg_entries: List[Tuple[str, str]] = []
    if len(configured_cbgs) > 0:
        for g in configured_cbgs:
            if os.path.isabs(g):
                g_dir = g
                cbg_name = os.path.basename(os.path.normpath(g_dir)) or g
            else:
                g_dir = os.path.join(npy_root, g)
                cbg_name = g
            cbg_entries.append((cbg_name, g_dir))
    else:
        subdirs = _list_subdirs(npy_root)
        if len(subdirs) > 0:
            cbg_entries = [(name, os.path.join(npy_root, name)) for name in subdirs]
        elif _has_required_cbg_files(npy_root, files_cfg):
            cbg_name = (
                os.path.basename(os.path.normpath(npy_root))
                or os.path.basename(os.path.dirname(os.path.normpath(npy_root)))
                or "cbg"
            )
            cbg_entries = [(cbg_name, npy_root)]
        else:
            raise ValueError(
                f"No CBG subfolders found under {npy_root} and no files detected in the specified directory."
            )
    if len(cbg_entries) == 0:
        raise ValueError(f"No valid CBG directories resolved from configuration (npy_root={npy_root}).")

    configured_demos = [str(x) for x in (groups_cfg.get("demos", []) or [])]
    demo_universe: set = set(configured_demos)

    sum_by_cbg: Dict[Tuple[str, str], np.ndarray] = {}
    traj_count_by_cbg: Dict[Tuple[str, str], int] = {}

    for g, g_dir in cbg_entries:
        if not os.path.isdir(g_dir):
            if verbose:
                print(f"[WARN] Skip non-dir: {g_dir}")
            continue
        sel = _load_selected_indices(g_dir, files_cfg)  # [M]
        seqs_all = _load_sequences_for_cbg(g_dir, files_cfg)  # list length N
        demos_all = _load_demo_labels_for_cbg(g_dir, files_cfg, demo_source, attr_keys)

        N = len(seqs_all)
        if demos_all.shape[0] != N:
            raise ValueError(f"CBG {g}: sequences N {N} != demo N {demos_all.shape[0]}")
        if np.any((sel < 0) | (sel >= N)):
            raise ValueError(f"CBG {g}: selected_indices out of bounds (max valid {N-1})")

        seqs = [seqs_all[i] for i in sel.tolist()]
        demos = np.asarray(demos_all)[sel]
        if len(configured_demos) == 0:
            demo_universe.update(set(demos.tolist()))
            allowed: Optional[set[str]] = None
        else:
            allowed = set(configured_demos)

        prev_total_for_g = sum(n for (gg, _d), n in traj_count_by_cbg.items() if gg == g)
        for i, tokens in enumerate(seqs):
            d_str = str(demos[i])
            if allowed is not None and d_str not in allowed:
                continue
            tok = np.asarray(tokens, dtype=np.int64)
            mask = (tok >= num_special) & (tok < V)
            if not np.any(mask):
                continue
            poi_ids = tok[mask] - num_special
            counts = np.bincount(poi_ids, minlength=V_eff).astype(np.float64)
            if counts.sum() <= 0:
                continue
            s_vec = counts / counts.sum()
            _update_aggregates(sum_by_cbg, traj_count_by_cbg, (g, d_str), s_vec)
        if verbose:
            new_total_for_g = sum(n for (gg, _d), n in traj_count_by_cbg.items() if gg == g)
            used = new_total_for_g - prev_total_for_g
            print(f"[INFO] CBG {g}: selected {len(seqs)} sequences, used {used}")

    if len(sum_by_cbg) == 0:
        raise RuntimeError("No aggregates collected (empty after filtering).")

    sbg_f: Dict[Tuple[str, str], np.ndarray] = {}
    tbg_f: Dict[Tuple[str, str], int] = {}
    for key, vec in sum_by_cbg.items():
        n = traj_count_by_cbg[key]
        if n >= min_traj_per_group:
            sbg_f[key] = vec
            tbg_f[key] = n
    sum_by_cbg = sbg_f
    traj_count_by_cbg = tbg_f

    if len(sum_by_cbg) == 0:
        raise RuntimeError("No aggregates remain after applying min_traj_per_group filter.")

    p_poi_df = _finalize_p_poi(sum_by_cbg, traj_count_by_cbg, epsilon)
    p_poi_df["vocab_idx"] = p_poi_df["poi_index"] + num_special
    tokens = np.array(vocab_tokens, dtype=object)
    p_poi_df["token"] = tokens[p_poi_df["vocab_idx"].to_numpy()]
    p_dem_df = _finalize_p_dem(traj_count_by_cbg)

    p_poi_out = os.path.join(out_dir, "p_poi.csv")
    p_dem_out = os.path.join(out_dir, "p_dem.csv")
    p_poi_df.to_csv(p_poi_out, index=False)
    p_dem_df.to_csv(p_dem_out, index=False)
    p_poi_written = p_poi_out
    p_dem_written = p_dem_out

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(config_path),
        "npy_root": os.path.abspath(npy_root),
        "vocab_path": os.path.abspath(vocab_path),
        "num_vocab_tokens": int(V),
        "num_special_tokens": int(num_special),
        "effective_poi_tokens": int(V_eff),
        "num_groups": int(len(sum_by_cbg)),
        "selected_indices_mode": True,
        "outputs": {
            "p_poi": os.path.abspath(p_poi_written),
            "p_dem": os.path.abspath(p_dem_written),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Wrote aggregates to: {os.path.abspath(out_dir)}")
    print(f"       p_poi: {p_poi_written}")
    print(f"       p_dem: {p_dem_written}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CBG -> POI marginals.")
    ap.add_argument("--config", type=str, required=True, help="Path to poi_marginals.yaml")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_marginals(args.config)


