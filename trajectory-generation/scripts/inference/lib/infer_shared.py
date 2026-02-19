from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import sys

import numpy as np

# Ensure trajectory-generation root is on sys.path when imported standalone.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.utils import batch_count_lengths


def normalize_sequence(seq) -> List[str]:
    if isinstance(seq, str):
        return seq.strip().split()
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [str(token) for token in seq]
    raise TypeError(f"Unsupported trajectory format: {type(seq)}")


def load_length_ids(split_dir: Union[str, Path]) -> Optional[np.ndarray]:
    split_dir = Path(split_dir)
    length_path = split_dir / "trajectory_length_ids.npy"
    if not length_path.exists():
        return None
    try:
        return np.load(length_path, allow_pickle=True).astype(np.int64)
    except Exception as exc:
        print(f"Warning: failed to load {length_path} ({exc}); ignoring cached ids")
        return None


def load_or_compute_length_ids(split_dir: Union[str, Path], max_length: int) -> Optional[np.ndarray]:
    cached = load_length_ids(split_dir)
    if cached is not None:
        return cached
    split_dir = Path(split_dir)
    trajectory_file = split_dir / "final_segments_all_train_data.pkl"
    if not trajectory_file.exists():
        return None
    try:
        with open(trajectory_file, "rb") as f:
            trajectory_df = pickle.load(f)
        sequences = [normalize_sequence(row["unique_id_seq"]) for _, row in trajectory_df.iterrows()]
        return np.asarray(batch_count_lengths(sequences, max_length=max_length), dtype=np.int64)
    except Exception as exc:
        print(f"Warning: failed to compute length ids from {trajectory_file} ({exc})")
        return None


def select_length_subset(length_cache: Optional[np.ndarray], count: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    if length_cache is None or len(length_cache) == 0:
        print("Warning: missing cached trajectory lengths; defaulting to zeros")
        return np.zeros(count, dtype=np.int64)

    if indices is not None:
        if len(length_cache) < indices.max() + 1:
            raise ValueError(
                f"Length cache too small for requested indices (cache={len(length_cache)}, max_index={indices.max()})"
            )
        return length_cache[indices]

    if count <= len(length_cache):
        return length_cache[:count]

    sampled_idx = np.random.choice(len(length_cache), count, replace=True)
    return length_cache[sampled_idx]


def append_length_condition(attrs: np.ndarray, length_ids: Optional[np.ndarray]) -> np.ndarray:
    if length_ids is None:
        print("Warning: length conditioning enabled but no length ids provided; skipping append")
        return attrs
    if attrs.shape[0] != len(length_ids):
        raise ValueError(f"Attribute count mismatch: attrs={attrs.shape[0]} vs length_ids={len(length_ids)}")
    return np.column_stack([attrs, length_ids.reshape(-1, 1)])


def load_demo_pairs_from_attrs_with_demo(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"attrs_with_demo_npy not found: {path}")
    arr = np.load(str(path), allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"attrs_with_demo_npy must have shape [N,6+], got {arr.shape} at {path}")
    demo = arr[:, -2:].astype(np.int64, copy=False)
    valid = (demo[:, 0] >= 0) & (demo[:, 1] >= 0)
    demo = demo[valid]
    if demo.shape[0] == 0:
        raise ValueError(f"No valid demo rows (age/gender >=0) found in {path}")
    return demo


def sample_demo_pairs_from_real_demo(*, n: int, real_demo_pairs: np.ndarray, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if real_demo_pairs.ndim != 2 or real_demo_pairs.shape[1] < 2:
        raise ValueError(f"real_demo_pairs must be [N,2], got {real_demo_pairs.shape}")
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(real_demo_pairs.shape[0], size=int(n), replace=True)
    return real_demo_pairs[idx].astype(np.int64, copy=False)


def load_poi_mapping(tokenizer_path: str, poi_coords_path: str) -> Tuple[Dict[str, int], Dict[str, Tuple[float, float]]]:
    vocab_file = Path(tokenizer_path) / "vocab.txt"
    vocab: Dict[str, int] = {}
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        first_line = lines[0].strip() if lines else ""
        if "\t" in first_line or " " in first_line:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vocab[parts[0]] = int(parts[1])
        else:
            for token_id, line in enumerate(lines):
                token = line.strip()
                if token:
                    vocab[token] = token_id

    poi_coords: Dict[str, Tuple[float, float]] = {}
    poi_path = Path(poi_coords_path)
    if poi_path.exists():
        import pandas as pd

        poi_df = pd.read_csv(poi_path)
        for _, row in poi_df.iterrows():
            poi_coords[str(row["poi_id"])] = (float(row["lat"]), float(row["lon"]))

    return vocab, poi_coords
