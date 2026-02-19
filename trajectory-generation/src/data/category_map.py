"""POI -> category mapping utilities ( `poi_map_feature.csv`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def _read_vocab_tokens(vocab_path: str) -> List[str]:
    """
    Read a vocab file that is either:
      - one token per line, or
      - "token id" pairs per line.
    Returns a list where index == vocab id.
    """
    lines = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []

    # Detect "token id" pairs
    as_pairs = True
    for ln in lines[:200]:
        parts = ln.split()
        if len(parts) != 2:
            as_pairs = False
            break
        try:
            _ = int(parts[1])
        except Exception:
            as_pairs = False
            break

    if not as_pairs:
        return [ln.split()[0] for ln in lines]

    tok_to_id: Dict[str, int] = {}
    max_id = -1
    for ln in lines:
        tok, idx_s = ln.split()[0], ln.split()[1]
        idx = int(idx_s)
        tok_to_id[tok] = idx
        max_id = max(max_id, idx)
    id_to_tok = [""] * (max_id + 1)
    for tok, idx in tok_to_id.items():
        if 0 <= idx < len(id_to_tok):
            id_to_tok[idx] = tok
    for i, tok in enumerate(id_to_tok):
        if tok == "":
            id_to_tok[i] = f"<id_{i}>"
    return id_to_tok


def _load_poi_to_category(
    poi_map_csv: str,
    *,
    category_column: str,
) -> Dict[str, str]:
    df = pd.read_csv(poi_map_csv)
    if df.empty:
        raise ValueError(f"Empty POI map CSV: {poi_map_csv}")
    if "poi_id" not in df.columns:
        raise ValueError(f"POI map CSV must contain 'poi_id' column (got {list(df.columns)})")
    if category_column not in df.columns:
        raise ValueError(
            f"POI map CSV missing category column {category_column!r} (got {list(df.columns)})"
        )

    poi = df["poi_id"].astype(str).to_numpy()
    cat_raw = df[category_column].astype(str).to_numpy()

    out: Dict[str, str] = {}
    for p, c in zip(poi.tolist(), cat_raw.tolist()):
        cc = str(c).strip()
        if not cc or cc.lower() in {"nan", "none", "<na>"}:
            continue
        out[str(p)] = cc
    return out


@dataclass(frozen=True)
class CategoryMapSpec:
    poi_map_csv: str
    vocab_path: str
    num_special_tokens: int
    category_column: str = "top_category"
    # Optional: drop these POI token strings from the category mapping entirely.
    # Use this to exclude POI_HOME/POI_WORK/POI_OTHER from category-based objectives.
    drop_poi_tokens: Tuple[str, ...] = ()


class POICategoryMap:
    """
    Map POI-token probabilities (POI-only vocab indexing, after removing num_special_tokens)
    into a smaller category space.

    Any POI token not found in poi_map_feature.csv (including POI_HOME/WORK/OTHER) is dropped,
    and the resulting category probabilities are renormalized.
    """

    def __init__(
        self,
        spec: CategoryMapSpec,
        *,
        categories: Optional[Sequence[str]] = None,
    ):
        self.spec = spec
        self._poi_to_category = _load_poi_to_category(spec.poi_map_csv, category_column=spec.category_column)
        vocab = _read_vocab_tokens(spec.vocab_path)
        if not vocab:
            raise ValueError(f"Empty vocab at {spec.vocab_path}")
        if spec.num_special_tokens < 0 or spec.num_special_tokens >= len(vocab):
            raise ValueError(
                f"num_special_tokens={spec.num_special_tokens} out of range for vocab size {len(vocab)}"
            )

        # Choose category ordering
        if categories is None:
            cats = sorted(set(self._poi_to_category.values()))
        else:
            cats = [str(x) for x in list(categories)]
        if not cats:
            raise ValueError("No categories resolved (check category_column and mapping CSV).")
        self.categories: List[str] = cats
        self.category_to_index: Dict[str, int] = {c: i for i, c in enumerate(cats)}

        # Build poi_index -> category_index (poi_index is POI-only indexing, 0..V_eff-1)
        poi_tokens = vocab[spec.num_special_tokens :]
        V_eff = len(poi_tokens)
        poi_to_cat = np.full((V_eff,), -1, dtype=np.int64)
        drop_tok = set(str(x) for x in (spec.drop_poi_tokens or ()))
        for i, tok in enumerate(poi_tokens):
            if drop_tok and str(tok) in drop_tok:
                continue
            cat = self._poi_to_category.get(str(tok))
            if cat is None:
                continue
            j = self.category_to_index.get(cat)
            if j is None:
                continue
            poi_to_cat[i] = int(j)

        self._poi_to_cat_np = poi_to_cat
        valid_poi = np.where(poi_to_cat >= 0)[0].astype(np.int64)
        valid_cat = poi_to_cat[valid_poi].astype(np.int64, copy=False)
        self._valid_poi_np = valid_poi
        self._valid_cat_np = valid_cat

    def num_categories(self) -> int:
        return len(self.categories)

    def poi_vocab_size(self) -> int:
        return int(self._poi_to_cat_np.shape[0])

    def _valid_index_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        poi = torch.from_numpy(self._valid_poi_np).to(device=device, dtype=torch.long)
        cat = torch.from_numpy(self._valid_cat_np).to(device=device, dtype=torch.long)
        return poi, cat

    def map_probs(self, poi_probs: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
        """
        Map [..., V_poi] -> [..., C] by summing POI mass within each category, then renormalize.
        """
        if poi_probs.size(-1) != self.poi_vocab_size():
            raise ValueError(
                f"poi_probs last dim {int(poi_probs.size(-1))} != expected {self.poi_vocab_size()} "
                f"(check num_special_tokens/vocab alignment)."
            )
        C = self.num_categories()
        if C <= 0:
            raise ValueError("Category map has zero categories.")

        valid_poi, valid_cat = self._valid_index_tensors(device=poi_probs.device)
        src = poi_probs.index_select(dim=-1, index=valid_poi)  # [..., M]
        out = torch.zeros(*poi_probs.shape[:-1], C, device=poi_probs.device, dtype=poi_probs.dtype)
        out.index_add_(dim=-1, index=valid_cat, source=src)

        denom = out.sum(dim=-1, keepdim=True)
        bad = denom <= float(epsilon)
        out = out / denom.clamp_min(float(epsilon))
        if bad.any():
            # Replace degenerate rows with uniform over categories.
            out[bad.expand_as(out)] = 0.0
            # broadcast-safe uniform assignment
            uniform = float(1.0 / float(C))
            out = out + bad.to(dtype=out.dtype) * uniform
        return out

    def map_dist(self, poi_dist: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
        """Map [V_poi] -> [C] and renormalize."""
        if poi_dist.dim() != 1:
            raise ValueError("poi_dist must be 1D [V_poi]")
        mapped = self.map_probs(poi_dist.view(1, 1, -1), epsilon=epsilon).view(-1)
        return mapped


