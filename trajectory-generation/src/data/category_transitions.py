"""Loader for precomputed CBG -> category-transition distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class CategoryTransitionStore:
    """
    Load an .npz file produced by `scripts/build_category_transition_marginals.py`.

    Expected arrays:
      - cbgs: [G] strings
      - categories: [C] strings
      - probs: [G, C*C] float32 (each row normalized)
      - traj_count: [G] int64 (optional)
    """

    def __init__(self, npz_path: str):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Category transition npz not found: {npz_path}")

        with np.load(self.npz_path, allow_pickle=True) as data:
            if "cbgs" not in data or "categories" not in data or "probs" not in data:
                raise ValueError("Transition npz must contain arrays: cbgs, categories, probs")
            cbgs = data["cbgs"]
            categories = data["categories"]
            probs = data["probs"]
            traj_count = data["traj_count"] if "traj_count" in data else None

        self.cbgs: List[str] = [str(x) for x in cbgs.tolist()]
        self.categories: List[str] = [str(x) for x in categories.tolist()]

        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f"probs must be 2D [G, K], got {probs.shape}")
        if probs.shape[0] != len(self.cbgs):
            raise ValueError(f"Row mismatch: probs G={probs.shape[0]} vs cbgs={len(self.cbgs)}")

        C = len(self.categories)
        K = C * C
        if probs.shape[1] != K:
            raise ValueError(f"probs second dim must be C*C={K}, got {probs.shape[1]}")

        # Normalize defensively
        s = probs.sum(axis=1, keepdims=True)
        s = np.clip(s, 1e-12, None)
        probs = probs / s

        self._probs = torch.from_numpy(probs)  # [G, K]
        self._cbg_to_idx: Dict[str, int] = {str(c): i for i, c in enumerate(self.cbgs)}

        self._traj_count: Optional[np.ndarray] = None
        if traj_count is not None:
            tc = np.asarray(traj_count)
            if tc.shape[0] == len(self.cbgs):
                self._traj_count = tc.astype(np.int64, copy=False)

    def available_cbgs(self) -> List[str]:
        return sorted(self._cbg_to_idx.keys())

    def num_categories(self) -> int:
        return len(self.categories)

    def feature_size(self) -> int:
        C = self.num_categories()
        return C * C

    def get_distribution(self, cbg: str, *, device: Optional[torch.device] = None) -> torch.Tensor:
        key = str(cbg)
        if key not in self._cbg_to_idx:
            raise KeyError(f"CBG {cbg} not found in category transition store.")
        out = self._probs[self._cbg_to_idx[key]]
        if device is not None:
            out = out.to(device)
        return out

    def traj_count(self, cbg: str) -> int:
        if self._traj_count is None:
            return 0
        key = str(cbg)
        if key not in self._cbg_to_idx:
            raise KeyError(f"CBG {cbg} not found in transition store.")
        return int(self._traj_count[self._cbg_to_idx[key]])


