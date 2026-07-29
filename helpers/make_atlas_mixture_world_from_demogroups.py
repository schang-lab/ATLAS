#!/usr/bin/env python3
"""
Build a *mixture* LLP-world by sampling REAL trajectories from an existing demo-separated
world (e.g. llp_world_embee/world_train_demogroups).

This matches the user's desired "A1" setting:
  - We are NOT synthesizing new POI sequences.
  - Each output region is a mixture of demo groups according to a designed Π (pi_matrix.json).
  - Later, training uses:
      - aggregate/LLP branch with training.llp.demo_source = "pi" (demo is sampled from π, not from per-trajectory labels)
      - MSE branch as an anchor (diffusion_mse.enabled=true) but with diffusion_mse.demo_source="null" to avoid cheating.

The output world layout is compatible with:
  - trajectory-generation/scripts/build_poi_marginals.py
  - trajectory-generation/scripts/precompute/cache_cbg_conditionals.py

No PyYAML required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_demo_key(d: str) -> Tuple[int, int]:
    # "a2_g1" -> (2,1)
    s = str(d).strip()
    if not (s.startswith("a") and "_g" in s):
        raise ValueError(f"Invalid demo key: {d}")
    a_str, g_str = s[1:].split("_g", 1)
    return int(a_str), int(g_str)


def _load_pi_matrix(pi_json: Path) -> Tuple[List[str], List[Tuple[str, np.ndarray]]]:
    obj = json.loads(pi_json.read_text(encoding="utf-8"))
    demo_keys = [str(x) for x in obj["demo_keys"]]
    regions: List[Tuple[str, np.ndarray]] = []
    for r in obj["regions"]:
        rid = str(r["region_id"])
        pi_map = r["pi"]
        vec = np.array([float(pi_map[k]) for k in demo_keys], dtype=np.float64)
        s = float(vec.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
            raise ValueError(f"Region {rid}: π does not sum to 1 (sum={s})")
        if (vec < -1e-9).any():
            raise ValueError(f"Region {rid}: π has negative entries")
        regions.append((rid, vec))
    return demo_keys, regions


def _allocate_counts_for_region(pi_vec: np.ndarray, N: int) -> np.ndarray:
    # Deterministic rounding: floor + distribute remainder by largest fractional parts.
    expected = pi_vec * float(N)
    base = np.floor(expected).astype(np.int64)
    rem = int(N - int(base.sum()))
    if rem > 0:
        frac = expected - base
        order = np.argsort(-frac)  # descending
        for j in order[:rem]:
            base[int(j)] += 1
    elif rem < 0:
        # Shouldn't happen, but guard: remove from smallest fractional parts where base>0.
        frac = expected - base
        order = np.argsort(frac)  # ascending
        need = -rem
        for j in order:
            jj = int(j)
            if base[jj] > 0:
                base[jj] -= 1
                need -= 1
                if need <= 0:
                    break
        if need != 0:
            raise RuntimeError("Failed to fix negative remainder in count allocation.")
    if int(base.sum()) != int(N):
        raise RuntimeError("Count allocation bug: region counts do not sum to N.")
    return base


def _choose_default_N(
    demo_counts: Dict[str, int],
    demo_keys: List[str],
    regions: List[Tuple[str, np.ndarray]],
    *,
    max_N: Optional[int],
) -> int:
    # For each demo d, total required across all regions is approximately N * sum_r pi_r(d).
    P = np.stack([pi for _rid, pi in regions], axis=0)  # [R, D]
    mass = P.sum(axis=0)  # [D]
    # Avoid division by zero (shouldn't happen with valid Π).
    mass = np.maximum(mass, 1e-12)
    N_max = None
    for j, d in enumerate(demo_keys):
        avail = int(demo_counts[d])
        cap = int(np.floor(avail / float(mass[j])))
        N_max = cap if N_max is None else min(N_max, cap)
    if N_max is None:
        raise RuntimeError("Failed to derive N.")
    if N_max <= 0:
        raise ValueError(
            f"Not enough trajectories to build any mixture world without reuse (derived N_max={N_max}). "
            "Either enable --allow-reuse, reduce regions, or use a different source world."
        )
    if max_N is not None:
        N_max = min(N_max, int(max_N))
    return int(N_max)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-world-root", type=str, required=True, help="Demo-separated world root (e.g. world_train_demogroups)")
    ap.add_argument("--pi-json", type=str, required=True, help="pi_matrix.json (regions x demos)")
    ap.add_argument("--out-root", type=str, required=True, help="Output mixture world root")
    ap.add_argument("--num-traj-per-region", type=int, default=None, help="If omitted, auto-derive the max feasible N without reuse.")
    ap.add_argument("--max-num-traj-per-region", type=int, default=None, help="Optional cap when auto-deriving N.")
    ap.add_argument("--allow-reuse", action="store_true", help="Allow reusing the same source trajectory across regions (samples with replacement).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source_root = Path(args.source_world_root)
    out_root = Path(args.out_root)
    pi_json = Path(args.pi_json)

    demo_keys, regions = _load_pi_matrix(pi_json)

    # Load source demo buckets
    source_seq: Dict[str, np.ndarray] = {}
    source_attr: Dict[str, np.ndarray] = {}
    demo_counts: Dict[str, int] = {}
    for d in demo_keys:
        src_dir = source_root / f"demo_{d}"
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source demo directory: {src_dir}")
        seq_path = src_dir / "generated_sequences.npy"
        attr_path = src_dir / "all_attr_results.demographics.npy"
        if not seq_path.exists() or not attr_path.exists():
            raise FileNotFoundError(f"Missing required files under {src_dir}")
        seq_arr = np.load(seq_path, allow_pickle=True)
        attr_arr = np.load(attr_path, allow_pickle=False).astype(np.float32, copy=False)
        if attr_arr.ndim != 2 or attr_arr.shape[1] < 6:
            raise ValueError(f"Bad demographics array at {attr_path}: {attr_arr.shape}")
        if seq_arr.shape[0] != attr_arr.shape[0]:
            raise ValueError(f"Seq/attr length mismatch in {src_dir}: {seq_arr.shape[0]} vs {attr_arr.shape[0]}")
        source_seq[d] = seq_arr
        source_attr[d] = attr_arr
        demo_counts[d] = int(attr_arr.shape[0])

    # Choose N
    if args.num_traj_per_region is None:
        N = _choose_default_N(demo_counts, demo_keys, regions, max_N=args.max_num_traj_per_region)
        print(f"[INFO] Auto-derived num_traj_per_region={N} (allow_reuse={bool(args.allow_reuse)})")
    else:
        N = int(args.num_traj_per_region)
        if N <= 0:
            raise ValueError("--num-traj-per-region must be > 0")

    # Compute counts per region; if no reuse, ensure feasibility under rounding.
    while True:
        counts_by_region: Dict[str, np.ndarray] = {}
        total_per_demo = np.zeros((len(demo_keys),), dtype=np.int64)
        for rid, pi in regions:
            c = _allocate_counts_for_region(pi, N)
            counts_by_region[rid] = c
            total_per_demo += c

        if args.allow_reuse:
            break

        ok = True
        for j, d in enumerate(demo_keys):
            if int(total_per_demo[j]) > int(demo_counts[d]):
                ok = False
                break
        if ok:
            break
        N -= 1
        if N <= 0:
            raise ValueError("Failed to find a feasible num_traj_per_region without reuse. Use --allow-reuse.")
        print(f"[WARN] Rounding caused demo overuse; retrying with num_traj_per_region={N}")

    rng = np.random.default_rng(int(args.seed))
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare index pools per demo
    demo_perm: Dict[str, np.ndarray] = {}
    demo_ptr: Dict[str, int] = {}
    for d in demo_keys:
        if args.allow_reuse:
            demo_perm[d] = np.arange(demo_counts[d], dtype=np.int64)  # not used sequentially
        else:
            demo_perm[d] = rng.permutation(demo_counts[d]).astype(np.int64, copy=False)
        demo_ptr[d] = 0

    summaries: List[Dict[str, object]] = []
    cbg_ids: List[str] = []

    for rid, _pi in regions:
        region_dir = out_root / rid
        region_dir.mkdir(parents=True, exist_ok=True)
        c = counts_by_region[rid]

        seq_list: List[np.ndarray] = []
        attr_list: List[np.ndarray] = []

        for j, d in enumerate(demo_keys):
            k = int(c[j])
            if k <= 0:
                continue
            if args.allow_reuse:
                idx = rng.integers(0, demo_counts[d], size=k, dtype=np.int64)
            else:
                start = int(demo_ptr[d])
                end = start + k
                if end > demo_perm[d].shape[0]:
                    raise RuntimeError(
                        f"Internal allocation bug: demo {d} needs {k} more but only {demo_perm[d].shape[0] - start} left."
                    )
                idx = demo_perm[d][start:end]
                demo_ptr[d] = end

            seq_chunk = source_seq[d][idx]
            attr_chunk = source_attr[d][idx]
            seq_list.extend(seq_chunk.tolist() if isinstance(seq_chunk, np.ndarray) and seq_chunk.dtype == object else list(seq_chunk))
            attr_list.append(attr_chunk)

        if not seq_list:
            raise RuntimeError(f"Region {rid}: empty after sampling")

        attrs = np.concatenate(attr_list, axis=0).astype(np.float32, copy=False)
        # Shuffle within region so demos are not in contiguous blocks.
        perm = rng.permutation(attrs.shape[0])
        attrs = attrs[perm]
        seq_list = [seq_list[i] for i in perm.tolist()]

        np.save(region_dir / "generated_sequences.npy", np.array(seq_list, dtype=object), allow_pickle=True)
        np.save(region_dir / "all_attr_results.demographics.npy", attrs, allow_pickle=False)
        np.save(region_dir / "selected_indices.npy", np.arange(attrs.shape[0], dtype=np.int64), allow_pickle=False)

        # Summary
        age = attrs[:, 4].astype(np.int64)
        gender = attrs[:, 5].astype(np.int64)
        demo_pairs = [f"a{a}_g{g}" for a, g in zip(age.tolist(), gender.tolist())]
        counts = pd.Series(demo_pairs).value_counts().to_dict()
        summaries.append(
            {
                "region_id": rid,
                "num_traj": int(attrs.shape[0]),
                "home_lat_min": float(np.min(attrs[:, 2])),
                "home_lat_max": float(np.max(attrs[:, 2])),
                "home_lon_min": float(np.min(attrs[:, 3])),
                "home_lon_max": float(np.max(attrs[:, 3])),
                "demo_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
        cbg_ids.append(rid)

    pd.DataFrame(summaries).to_csv(out_root / "region_summary.csv", index=False)
    (out_root / "cbgs.txt").write_text("\n".join(cbg_ids) + "\n", encoding="utf-8")

    meta = {
        "source_world_root": str(source_root),
        "pi_json": str(pi_json),
        "out_root": str(out_root),
        "allow_reuse": bool(args.allow_reuse),
        "num_traj_per_region": int(N),
        "seed": int(args.seed),
        "regions_written": int(len(cbg_ids)),
        "demo_counts_source": {k: int(v) for k, v in demo_counts.items()},
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote real-sample mixture world to {out_root} ({len(cbg_ids)} regions, N={N})")


if __name__ == "__main__":
    main()

