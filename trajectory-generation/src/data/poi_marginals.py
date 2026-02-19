"""Loader for precomputed CBG -> POI marginal distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


class POIMarginalStore:
    """Load `p_poi.csv` (produced by `build_poi_marginals.py`) into tensor form."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"p_poi.csv not found: {csv_path}")
        self._cbg_probs: Dict[str, torch.Tensor] = {}
        self._traj_counts: Dict[str, int] = {}
        self._vocab_size: Optional[int] = None
        self._load()

    def _load(self) -> None:
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"p_poi.csv at {self.csv_path} is empty.")
        if "cbg" not in df.columns or "poi_index" not in df.columns:
            raise ValueError("p_poi.csv must contain columns: cbg, poi_index, and either prob or weight_sum.")

        # Normalize CBG identifiers to strings so they are consistent across cache and marginals.
        df["cbg"] = df["cbg"].astype(str)
        df = df.sort_values(["cbg", "poi_index"]).reset_index(drop=True)
        self._vocab_size = int(df["poi_index"].max() + 1)

        if "demo" in df.columns and "weight_sum" in df.columns:
            # New schema: aggregate across demo using weight_sum, then normalize per CBG
            agg = df.groupby(["cbg", "poi_index"], as_index=False)["weight_sum"].sum()
            # Derive traj_count per CBG by summing per-demo counts (take max within (cbg,demo) then sum over demo)
            if "traj_count" in df.columns:
                per_demo_counts = df.groupby(["cbg", "demo"])["traj_count"].max().reset_index()
                traj_by_cbg = per_demo_counts.groupby("cbg")["traj_count"].sum().to_dict()
            else:
                traj_by_cbg = {}
            grouped = agg.groupby("cbg")
            for cbg, sub in grouped:
                cbg = str(cbg)
                vec = np.zeros(self._vocab_size, dtype=np.float32)
                idx = sub["poi_index"].to_numpy(dtype=int)
                ws = sub["weight_sum"].to_numpy(dtype=np.float32)
                vec[idx] = ws
                probs = _normalize_vector(vec)
                self._cbg_probs[cbg] = torch.from_numpy(probs)
                self._traj_counts[cbg] = int(traj_by_cbg.get(cbg, 0))
        else:
            # Back-compat: expect one row per (cbg, poi_index) with 'prob'
            if "prob" not in df.columns:
                raise ValueError("Legacy p_poi.csv requires 'prob' column when 'demo'/'weight_sum' are absent.")
            grouped = df.groupby("cbg")
            for cbg, sub in grouped:
                cbg = str(cbg)
                probs = sub["prob"].to_numpy(dtype=np.float32)
                if probs.shape[0] != self._vocab_size:
                    padded = np.zeros(self._vocab_size, dtype=np.float32)
                    padded[sub["poi_index"].to_numpy(dtype=int)] = probs
                    probs = padded
                probs = _normalize_vector(probs)
                self._cbg_probs[cbg] = torch.from_numpy(probs)
                traj_counts = sub["traj_count"].to_numpy(dtype=np.int64) if "traj_count" in sub else None
                self._traj_counts[cbg] = int(traj_counts.mean()) if traj_counts is not None else 0

    # ------------------------------------------------------------------
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("Vocab size not initialized.")
        return self._vocab_size

    def available_cbgs(self) -> List[str]:
        return sorted(self._cbg_probs.keys())

    def has(self, cbg: str) -> bool:
        return str(cbg) in self._cbg_probs

    def get_distribution(self, cbg: str, *, device: Optional[torch.device] = None) -> torch.Tensor:
        cbg = str(cbg)
        if cbg not in self._cbg_probs:
            raise KeyError(f"CBG {cbg} not found in POI marginals.")
        tensor = self._cbg_probs[cbg]
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def get_batch(self, cbgs: Iterable[str], *, device: Optional[torch.device] = None) -> torch.Tensor:
        tensors = [self.get_distribution(cbg, device=device) for cbg in cbgs]
        return torch.stack(tensors, dim=0)

    def traj_count(self, cbg: str) -> int:
        if cbg not in self._traj_counts:
            raise KeyError(f"CBG {cbg} missing traj count.")
        return self._traj_counts[cbg]


def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = float(vec.sum())
    if total <= 0:
        return np.full_like(vec, 1.0 / max(vec.size, 1))
    return (vec + eps) / (total + eps * vec.size)

