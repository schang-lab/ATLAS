from __future__ import annotations

import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .eval_by_demo_deps import _as_list

# Match training defaults for category-based objectives.
CATEGORY_COLUMN_DEFAULT = "top_category"
CATEGORY_DROP_POI_TOKENS_DEFAULT = ("POI_HOME", "POI_WORK", "POI_OTHER")


def _load_demo_from_attrs(attrs: np.ndarray, *, name: str) -> np.ndarray:
    """
    Extract (age_bin, gender_id) from an attrs array.

    Notes:
    - Many pipelines use 6-D attrs: [work_lat, work_lon, home_lat, home_lon, age_bin, gender_id]
    - Some pipelines insert a length id before demo, producing 7-D attrs:
        [work_lat, work_lon, home_lat, home_lon, length_id, age_bin, gender_id]

    To be robust across both, we read demo ids from the last 2 columns.
    """
    if attrs.ndim != 2 or attrs.shape[1] < 6:
        raise ValueError(f"{name} must have shape [N,6+] (coords + [...], age + gender), got {attrs.shape}")
    demo = attrs[:, -2:].astype(np.int64, copy=False)
    return demo


def _summarize_demo_counts(keys: np.ndarray, *, group_by: str) -> Dict[int, int]:
    del group_by
    out: Dict[int, int] = {}
    if keys.size == 0:
        return out
    unique, counts = np.unique(keys.astype(np.int64, copy=False), return_counts=True)
    for k, c in zip(unique.tolist(), counts.tolist()):
        out[int(k)] = int(c)
    return out


def _load_attrs_with_demo(path: str, *, name: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"{name} must have shape [N,6+] (coords + age + gender), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _group_keys(demo: np.ndarray, group_by: str) -> np.ndarray:
    if demo.ndim != 2 or demo.shape[1] < 2:
        raise ValueError(f"demo must be [N,2], got {demo.shape}")
    if group_by == "age":
        return demo[:, 0].astype(np.int64)
    if group_by == "age_gender":
        return (demo[:, 0].astype(np.int64) * 1000 + demo[:, 1].astype(np.int64))
    raise ValueError(f"group_by must be 'age' or 'age_gender' (got {group_by!r})")


def _unpack_key(key: int, group_by: str) -> Tuple[Optional[int], Optional[int]]:
    if group_by == "age":
        return int(key), None
    age = int(key // 1000)
    gender = int(key % 1000)
    return age, gender


def _load_poi_sequences_from_pkl(path: str, *, kind: str) -> List[List[str]]:
    if kind == "real":
        df = pd.read_pickle(path)
        seqs = df["unique_id_seq"].tolist()
        return [_as_list(s) for s in seqs]
    with open(path, "rb") as f:
        seqs = pickle.load(f)
    return [_as_list(s) for s in seqs]


def _load_poi_to_category(
    poi_map_csv: str,
    *,
    category_column: str = CATEGORY_COLUMN_DEFAULT,
) -> Tuple[Dict[str, str], List[str], Dict[str, int]]:
    """Load POI->category mapping from poi_map_feature.csv."""
    df = pd.read_csv(poi_map_csv)
    if df.empty:
        raise ValueError(f"Empty POI map CSV: {poi_map_csv}")
    if "poi_id" not in df.columns:
        raise ValueError(f"POI map CSV must contain 'poi_id' column (got {list(df.columns)})")
    if category_column not in df.columns:
        raise ValueError(f"POI map CSV missing category column {category_column!r} (got {list(df.columns)})")

    poi = df["poi_id"].astype(str).to_numpy()
    cat_raw = df[category_column].astype(str).to_numpy()
    poi_to_category: Dict[str, str] = {}
    for p, c in zip(poi.tolist(), cat_raw.tolist()):
        cc = str(c).strip()
        if not cc or cc.lower() in {"nan", "none", "<na>"}:
            continue
        poi_to_category[str(p)] = cc

    categories = sorted(set(poi_to_category.values()))
    category_to_index = {c: i for i, c in enumerate(categories)}
    return poi_to_category, categories, category_to_index


def _format_group_keys(keys: Iterable[int], *, group_by: str) -> str:
    parts = []
    for k in keys:
        age_bin, gender_id = _unpack_key(int(k), group_by)
        parts.append(f"({age_bin},{gender_id})")
    return ", ".join(parts)
