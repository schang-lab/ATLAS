#!/usr/bin/env python3
"""
Design a mixture-LLP demographic matrix Π (regions x demos).

Default (Carlos): 8 demos (4 ages x 2 genders) and 16 regions:
  - 8 near-one-hot regions (dominant demo mass = 0.85)
  - 8 pairwise mixture regions (0.5 / 0.5), built from two edge-disjoint perfect matchings

You can reduce the number of regions by lowering:
  - --num-hot (<= 8) and/or
  - --num-pair (<= 8)

Notes:
  - If you want Π to be full-rank for 8 demos, you typically want at least 8 diverse regions.
    Using all 8 near-one-hot rows already gives rank 8 in practice.

Outputs:
  - pi_matrix.json
  - (optional) pi_matrix.csv

No PyYAML required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DEMOS = [
    "a0_g0",
    "a0_g1",
    "a1_g0",
    "a1_g1",
    "a2_g0",
    "a2_g1",
    "a3_g0",
    "a3_g1",
]


def _pairs_from_two_matchings(demo_keys: List[str], seed: int) -> List[Tuple[str, str]]:
    """
    Build 8 pairs by taking 2 random perfect matchings over the 8 demos.
    Each demo appears exactly twice across the 8 pairs.
    """
    if len(demo_keys) != 8:
        raise ValueError("This helper currently expects exactly 8 demos.")
    seed = int(seed)

    def matching(seed_offset: int) -> List[Tuple[str, str]]:
        keys = list(demo_keys)
        rng = np.random.default_rng(seed + int(seed_offset))
        rng.shuffle(keys)
        return [(keys[i], keys[i + 1]) for i in range(0, len(keys), 2)]

    pairs1 = matching(0)
    edges1 = {tuple(sorted(x)) for x in pairs1}

    # Resample the second matching until it is edge-disjoint from the first.
    # This guarantees:
    #   - 8 unique pairs (4 + 4)
    #   - each demo appears exactly twice across the 8 pairs (degree 2 in the union)
    for attempt in range(1, 10_000):
        pairs2 = matching(attempt)
        edges2 = {tuple(sorted(x)) for x in pairs2}
        if edges1.isdisjoint(edges2):
            return pairs1 + pairs2

    raise RuntimeError("Failed to construct two edge-disjoint perfect matchings after many attempts.")


def _matrix_rank_and_cond(P: np.ndarray, tol: float = 1e-10) -> Dict[str, float]:
    if P.ndim != 2:
        raise ValueError("Π must be 2D")
    U, s, Vt = np.linalg.svd(P, full_matrices=False)
    rank = int(np.sum(s > tol))
    cond = float(s[0] / max(s[-1], tol)) if s.size > 0 else float("nan")
    return {"rank": float(rank), "cond": float(cond), "s_min": float(s[-1] if s.size else 0.0), "s_max": float(s[0] if s.size else 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=str, required=True, help="Output pi_matrix.json path")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional pi_matrix.csv path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo-keys", type=str, nargs="*", default=None, help="Demo keys like a0_g0 a0_g1 ... (default: 8 Carlos demos)")
    ap.add_argument("--dominant-mass", type=float, default=0.85, help="Mass on dominant demo for near-one-hot rows")
    ap.add_argument("--pair-mass", type=float, default=0.5, help="Mass on each of two demos for pair rows")
    ap.add_argument("--num-hot", type=int, default=8, help="How many near-one-hot regions to include (<=8)")
    ap.add_argument("--num-pair", type=int, default=8, help="How many pairwise-mixture regions to include (<=8)")
    ap.add_argument("--prefix", type=str, default="mix", help="Region id prefix")
    args = ap.parse_args()

    demo_keys = list(args.demo_keys) if args.demo_keys else list(DEFAULT_DEMOS)
    D = len(demo_keys)
    if D != 8:
        raise ValueError(f"This script is currently tuned for 8 demos; got {D}.")
    num_hot = int(args.num_hot)
    num_pair = int(args.num_pair)
    if not (0 <= num_hot <= 8):
        raise ValueError("--num-hot must be in [0, 8]")
    if not (0 <= num_pair <= 8):
        raise ValueError("--num-pair must be in [0, 8]")
    if num_hot + num_pair <= 0:
        raise ValueError("Need at least one region: num_hot + num_pair must be > 0")
    dom = float(args.dominant_mass)
    if not (0.5 < dom < 1.0):
        raise ValueError("--dominant-mass must be in (0.5, 1.0)")
    pair_mass = float(args.pair_mass)
    if abs(pair_mass * 2.0 - 1.0) > 1e-6:
        raise ValueError("--pair-mass must be 0.5 for 2-way mixtures (so masses sum to 1).")

    regions: List[Dict[str, object]] = []
    rows: List[np.ndarray] = []

    # Near-one-hot rows
    eps = (1.0 - dom) / float(D - 1)
    for d_idx, d in enumerate(demo_keys[:num_hot]):
        pi = np.full((D,), eps, dtype=np.float64)
        pi[d_idx] = dom
        region_id = f"{args.prefix}_hot_{d}"
        regions.append({"region_id": region_id, "pi": {k: float(pi[i]) for i, k in enumerate(demo_keys)}})
        rows.append(pi)

    # Pairwise rows (subset of the 8 disjoint-matching pairs)
    pairs = _pairs_from_two_matchings(demo_keys, seed=int(args.seed))[:num_pair]
    for i, (a, b) in enumerate(pairs):
        pi = np.zeros((D,), dtype=np.float64)
        pi[demo_keys.index(a)] = pair_mass
        pi[demo_keys.index(b)] = pair_mass
        region_id = f"{args.prefix}_pair_{i:02d}_{a}_{b}"
        regions.append({"region_id": region_id, "pi": {k: float(pi[j]) for j, k in enumerate(demo_keys)}})
        rows.append(pi)

    P = np.stack(rows, axis=0)  # [R, D]
    stats = _matrix_rank_and_cond(P)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "demo_keys": demo_keys,
        "regions": regions,
        "design": {
            "type": "near_one_hot_plus_pairs",
            "dominant_mass": dom,
            "pair_mass": pair_mass,
            "seed": int(args.seed),
            "num_hot": int(num_hot),
            "num_pair": int(num_pair),
            "num_regions": int(P.shape[0]),
            "num_demos": int(D),
        },
        "matrix_stats": stats,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_rows = []
        for entry in regions:
            rid = str(entry["region_id"])
            pi_map = entry["pi"]
            row = {"region_id": rid}
            for k in demo_keys:
                row[k] = float(pi_map[k])
            df_rows.append(row)
        pd.DataFrame(df_rows).to_csv(out_csv, index=False)
        print(f"[DONE] Wrote {out_csv}")


if __name__ == "__main__":
    main()
