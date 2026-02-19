"""Load and sample cached per-CBG conditioning tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ConditionBatch:
    """Typed container returned by :meth:`CBGConditionCache.sample`."""

    cbg: str
    home: torch.Tensor
    work: torch.Tensor
    age_bin: torch.Tensor
    gender_id: torch.Tensor
    indices: torch.Tensor


class CBGConditionCache:
    """Memory-light loader for `.npz` files produced by `cache_cbg_conditionals.py`.

    The loader keeps data in numpy format and only converts to torch on demand.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        self._arrays: Dict[str, Dict[str, np.ndarray]] = {}
        self._lengths: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def available_cbgs(self) -> List[str]:
        return sorted({p.stem for p in self.cache_dir.glob("*.npz")})

    def __contains__(self, cbg: str) -> bool:
        return (self.cache_dir / f"{cbg}.npz").exists()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_arrays(self, cbg: str) -> Dict[str, np.ndarray]:
        if cbg not in self._arrays:
            path = self.cache_dir / f"{cbg}.npz"
            if not path.exists():
                raise FileNotFoundError(f"CBG cache not found: {path}")
            with np.load(path) as data:
                arrays = {
                    "home": data["home"].astype(np.float32, copy=True),
                    "work": data["work"].astype(np.float32, copy=True),
                    "age_bin": data["age_bin"].astype(np.int64, copy=True),
                    "gender_id": data["gender_id"].astype(np.int64, copy=True),
                    "indices": data["indices"].astype(np.int64, copy=True),
                }
            length = arrays["home"].shape[0]
            self._arrays[cbg] = arrays
            self._lengths[cbg] = length
        return self._arrays[cbg]

    def population(self, cbg: str) -> int:
        self._load_arrays(cbg)
        return self._lengths[cbg]

    # ------------------------------------------------------------------
    # Sampling + statistics
    # ------------------------------------------------------------------
    def sample(
        self,
        cbg: str,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        replacement: bool = True,
    ) -> ConditionBatch:
        """Sample a batch of conditioning tensors for a given CBG."""

        arrays = self._load_arrays(cbg)
        population = self._lengths[cbg]
        if population == 0:
            raise ValueError(f"CBG {cbg} has zero cached samples.")
        if not replacement and batch_size > population:
            raise ValueError(f"Batch size {batch_size} exceeds available samples ({population}) without replacement.")

        if generator is None:
            indices = torch.randint(
                population,
                size=(batch_size,),
                device="cpu",
                dtype=torch.int64,
                requires_grad=False,
            )
        else:
            indices = torch.randint(
                population,
                size=(batch_size,),
                device="cpu",
                dtype=torch.int64,
                generator=generator,
            )

        if not replacement:
            indices = torch.unique(indices, sorted=False)
            if indices.numel() < batch_size:
                needed = batch_size - indices.numel()
                extra = torch.randint(population, (needed,), dtype=torch.int64)
                indices = torch.cat([indices, extra], dim=0)
        np_idx = indices.numpy()

        device = device or torch.device("cpu")
        home = torch.from_numpy(arrays["home"][np_idx]).to(device=device)
        work = torch.from_numpy(arrays["work"][np_idx]).to(device=device)
        age = torch.from_numpy(arrays["age_bin"][np_idx]).to(device=device)
        gender = torch.from_numpy(arrays["gender_id"][np_idx]).to(device=device)
        sel_indices = torch.from_numpy(arrays["indices"][np_idx]).to(device=device)

        return ConditionBatch(
            cbg=cbg,
            home=home,
            work=work,
            age_bin=age,
            gender_id=gender,
            indices=sel_indices,
        )

    def demo_distribution(
        self,
        cbg: str,
        *,
        num_age_bins: Optional[int] = None,
        num_genders: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return empirical age/gender histogram + normalized distribution."""

        arrays = self._load_arrays(cbg)
        age = arrays["age_bin"]
        gender = arrays["gender_id"]
        if num_genders is None:
            num_genders = int(gender.max() + 1) if gender.size > 0 else 1
        if num_age_bins is None:
            num_age_bins = int(age.max() + 1) if age.size > 0 else 1
        joint_index = age * num_genders + gender
        hist = np.bincount(joint_index, minlength=num_age_bins * num_genders).astype(np.float64)
        if hist.sum() > 0:
            dist = hist / hist.sum()
        else:
            dist = np.full_like(hist, 1.0 / hist.size)
        return hist, dist

    def as_torch_distribution(
        self,
        cbg: str,
        *,
        num_age_bins: Optional[int] = None,
        num_genders: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the empirical demo distribution as a torch tensor."""

        _, dist = self.demo_distribution(cbg, num_age_bins=num_age_bins, num_genders=num_genders)
        tensor = torch.from_numpy(dist.astype(np.float32))
        if device is not None:
            tensor = tensor.to(device)
        return tensor

