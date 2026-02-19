#!/usr/bin/env python3
"""
Generate a "perfect" mixture LLP world.

Given:
  - a region-by-demo mixture design Π (pi_matrix.json), and
  - demo-specific POI distributions q_d (from p_poi.csv or a demogroups world),

this script creates a world directory where each region contains a mixture of demos.

The intention is to satisfy LLP mixture assumptions as closely as possible:
  p_region ≈ Σ_d π_region(d) q_d

No PyYAML required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _read_vocab(vocab_path: Path) -> int:
    lines = vocab_path.read_text(encoding="utf-8").splitlines()
    toks = [ln.strip() for ln in lines if ln.strip()]
    return int(len(toks))


def _parse_demo_key(d: str) -> Tuple[int, int]:
    # "a2_g1" -> (2,1)
    s = d.strip()
    if not (s.startswith("a") and "_g" in s):
        raise ValueError(f"Invalid demo key: {d}")
    a_str, g_str = s[1:].split("_g", 1)
    return int(a_str), int(g_str)


def _load_pi_matrix(pi_json: Path) -> Tuple[List[str], List[Tuple[str, np.ndarray]]]:
    obj = json.loads(pi_json.read_text(encoding="utf-8"))
    demo_keys = [str(x) for x in obj["demo_keys"]]
    regions = []
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


def _load_q_from_p_poi(
    p_poi_csv: Path,
    demo_keys: List[str],
    *,
    cbg_name_for_demo: Dict[str, str],
    V_eff: int,
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(p_poi_csv)
    needed = {"cbg", "demo", "poi_index", "prob"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"{p_poi_csv} missing required columns {sorted(needed)} (has {sorted(df.columns)})")
    out: Dict[str, np.ndarray] = {}
    for d in demo_keys:
        cbg = cbg_name_for_demo.get(d)
        if cbg is None:
            raise ValueError(f"Missing cbg mapping for demo {d}")
        sub = df[(df["cbg"] == cbg) & (df["demo"] == d)]
        if sub.empty:
            raise ValueError(f"No rows found in p_poi.csv for cbg={cbg} demo={d}")
        vec = np.zeros((V_eff,), dtype=np.float64)
        idx = sub["poi_index"].to_numpy(dtype=np.int64)
        prob = sub["prob"].to_numpy(dtype=np.float64)
        if idx.min() < 0 or idx.max() >= V_eff:
            raise ValueError(f"poi_index out of range for demo={d}: min={idx.min()} max={idx.max()} V_eff={V_eff}")
        vec[idx] = prob
        s = vec.sum()
        if s <= 0:
            raise ValueError(f"Demo {d}: q_d sums to 0")
        vec = vec / s
        out[d] = vec
    return out


def _compute_q_from_world(
    world_root: Path,
    demo_keys: List[str],
    *,
    vocab_size: int,
    num_special_tokens: int,
) -> Dict[str, np.ndarray]:
    V_eff = int(vocab_size - num_special_tokens)
    out: Dict[str, np.ndarray] = {}
    for d in demo_keys:
        region_id = f"demo_{d}"
        region_dir = world_root / region_id
        seq_path = region_dir / "generated_sequences.npy"
        if not seq_path.exists():
            raise FileNotFoundError(f"Missing {seq_path} (expected demogroups world root)")
        arr = np.load(seq_path, allow_pickle=True)
        seqs = arr.tolist() if arr.dtype == object else [arr[i] for i in range(arr.shape[0])]
        sum_vec = np.zeros((V_eff,), dtype=np.float64)
        used = 0
        for s in seqs:
            tok = np.asarray(s, dtype=np.int64)
            mask = (tok >= num_special_tokens) & (tok < vocab_size)
            if not np.any(mask):
                continue
            poi = tok[mask] - num_special_tokens
            counts = np.bincount(poi, minlength=V_eff).astype(np.float64)
            if counts.sum() <= 0:
                continue
            sum_vec += counts / counts.sum()
            used += 1
        if used <= 0:
            raise RuntimeError(f"No usable sequences found for demo {d} under {region_dir}")
        vec = sum_vec / float(used)
        vec = (vec + 1e-12) / (vec.sum() + 1e-12 * vec.size)
        out[d] = vec
    return out


def _load_coords_pool_from_attrs(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected coords pool array [N,>=4], got {arr.shape} at {npy_path}")
    return arr[:, :4].astype(np.float32, copy=False)


def _load_demo_coords_pools_from_demogroups_world(world_root: Path, demo_keys: List[str]) -> Dict[str, np.ndarray]:
    pools: Dict[str, np.ndarray] = {}
    for d in demo_keys:
        region_dir = world_root / f"demo_{d}"
        p = region_dir / "all_attr_results.demographics.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        arr = np.load(p, allow_pickle=True).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Bad demographics array at {p}: {arr.shape}")
        pools[d] = arr[:, :4].astype(np.float32, copy=False)
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-json", type=str, required=True, help="pi_matrix.json from design_mixture_pi.py")
    ap.add_argument("--out-root", type=str, required=True, help="Output world root")
    ap.add_argument("--vocab-path", type=str, required=True, help="tokenizer/vocab.txt")
    ap.add_argument("--num-special-tokens", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-traj-per-region", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--q-p-poi-csv", type=str, default=None, help="Optional p_poi.csv to define q_d")
    ap.add_argument("--q-world-root", type=str, default="atlas_world/world_train_demogroups", help="Demogroups world root (fallback q_d source). Change to your own folder path.")

    ap.add_argument("--coords-mode", type=str, default="global_pool", choices=["global_pool", "constant", "per_demo_pool"])
    ap.add_argument("--coords-pool-npy", type=str, default=None, help="Path to all_attr_results_with_demo.npy (or any [N,>=4])")
    ap.add_argument("--coords-pool-world-root", type=str, default="atlas_world/world_train_demogroups", help="Demogroups world root to build per-demo pools. Change to your own folder path.")
    ap.add_argument("--constant-coords", type=float, nargs=4, default=None, metavar=("WORK_LAT", "WORK_LON", "HOME_LAT", "HOME_LON"))
    args = ap.parse_args()

    pi_json = Path(args.pi_json)
    out_root = Path(args.out_root)
    vocab_path = Path(args.vocab_path)
    out_root.mkdir(parents=True, exist_ok=True)

    demo_keys, regions = _load_pi_matrix(pi_json)
    if set(demo_keys) != set(DEFAULT_DEMOS):
        # Still allow custom ordering; just warn by printing.
        print(f"[WARN] demo_keys differ from default Carlos set: {demo_keys}")

    V = _read_vocab(vocab_path)
    num_special = int(args.num_special_tokens)
    if V <= num_special:
        raise ValueError(f"vocab too small: V={V} <= num_special_tokens={num_special}")
    V_eff = V - num_special
    print(f"[INFO] vocab size={V}, num_special_tokens={num_special} => POI-only dim={V_eff}")

    # Determine mapping demo -> cbg name for q-source in p_poi.csv using demogroups region naming.
    cbg_name_for_demo = {d: f"demo_{d}" for d in demo_keys}
    # If user provides region_summary.csv from demogroups world root, keep consistent.
    q_world_root = Path(args.q_world_root)
    rs_path = q_world_root / "region_summary.csv"
    if rs_path.exists():
        try:
            rs = pd.read_csv(rs_path)
            for rid in rs["region_id"].tolist():
                if isinstance(rid, str) and rid.startswith("demo_"):
                    d = rid[len("demo_") :]
                    if d in cbg_name_for_demo:
                        cbg_name_for_demo[d] = rid
        except Exception:
            pass

    # Load q_d
    q: Dict[str, np.ndarray]
    if args.q_p_poi_csv is not None:
        q = _load_q_from_p_poi(Path(args.q_p_poi_csv), demo_keys, cbg_name_for_demo=cbg_name_for_demo, V_eff=V_eff)
        q_source = {"type": "p_poi.csv", "path": str(args.q_p_poi_csv)}
    else:
        q = _compute_q_from_world(q_world_root, demo_keys, vocab_size=V, num_special_tokens=num_special)
        q_source = {"type": "world_sequences", "path": str(q_world_root)}

    # Prepare coords sampling
    rng = np.random.default_rng(int(args.seed))
    coords_mode = str(args.coords_mode)
    coords_pool = None
    per_demo_pools: Optional[Dict[str, np.ndarray]] = None
    if coords_mode == "global_pool":
        if args.coords_pool_npy is None:
            raise ValueError("--coords-pool-npy is required for coords-mode=global_pool")
        coords_pool = _load_coords_pool_from_attrs(Path(args.coords_pool_npy))
        if coords_pool.shape[0] <= 0:
            raise ValueError("coords_pool is empty")
    elif coords_mode == "per_demo_pool":
        per_demo_pools = _load_demo_coords_pools_from_demogroups_world(Path(args.coords_pool_world_root), demo_keys)
    elif coords_mode == "constant":
        if args.constant_coords is None:
            raise ValueError("--constant-coords is required for coords-mode=constant")
    else:
        raise ValueError(f"Unknown coords-mode: {coords_mode}")

    # Build world
    summaries: List[Dict[str, object]] = []
    cbg_ids: List[str] = []

    D = len(demo_keys)
    seq_len = int(args.seq_len)
    N = int(args.num_traj_per_region)
    if N <= 0:
        raise ValueError("--num-traj-per-region must be > 0")
    if seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    # Precompute demo categorical distribution sampler per region for speed.
    demo_indices = np.arange(D, dtype=np.int64)

    for region_id, pi_vec in regions:
        region_dir = out_root / region_id
        region_dir.mkdir(parents=True, exist_ok=True)

        demo_idx = rng.choice(demo_indices, size=N, replace=True, p=pi_vec.astype(np.float64))
        demo_for_row = [demo_keys[i] for i in demo_idx.tolist()]

        # Sample sequences and attrs
        seqs: List[np.ndarray] = []
        attrs = np.zeros((N, 6), dtype=np.float32)

        # coords
        if coords_mode == "global_pool":
            assert coords_pool is not None
            pool_idx = rng.integers(0, coords_pool.shape[0], size=N, dtype=np.int64)
            attrs[:, 0:4] = coords_pool[pool_idx]
        elif coords_mode == "constant":
            wlat, wlon, hlat, hlon = [float(x) for x in args.constant_coords]
            attrs[:, 0] = wlat
            attrs[:, 1] = wlon
            attrs[:, 2] = hlat
            attrs[:, 3] = hlon
        else:  # per_demo_pool
            assert per_demo_pools is not None
            for i, d in enumerate(demo_for_row):
                pool = per_demo_pools[d]
                j = int(rng.integers(0, pool.shape[0]))
                attrs[i, 0:4] = pool[j]

        # demo ids
        for i, d in enumerate(demo_for_row):
            a, g = _parse_demo_key(d)
            attrs[i, 4] = float(a)
            attrs[i, 5] = float(g)

        # sequences
        for d in demo_for_row:
            qd = q[d]
            poi_ids = rng.choice(np.arange(V_eff, dtype=np.int64), size=seq_len, replace=True, p=qd.astype(np.float64))
            tok_ids = poi_ids + num_special
            seqs.append(tok_ids.astype(np.int64, copy=False))

        np.save(region_dir / "generated_sequences.npy", np.array(seqs, dtype=object), allow_pickle=True)
        np.save(region_dir / "all_attr_results.demographics.npy", attrs, allow_pickle=False)
        np.save(region_dir / "selected_indices.npy", np.arange(N, dtype=np.int64), allow_pickle=False)

        # summary
        age = attrs[:, 4].astype(np.int64)
        gender = attrs[:, 5].astype(np.int64)
        pairs = list(zip(age.tolist(), gender.tolist()))
        counts = pd.Series(pairs).value_counts().sort_index()
        demo_json = {f"a{a}_g{g}": int(n) for (a, g), n in counts.items()}

        summaries.append(
            {
                "region_id": region_id,
                "num_traj": int(N),
                "home_lat_min": float(np.min(attrs[:, 2])),
                "home_lat_max": float(np.max(attrs[:, 2])),
                "home_lon_min": float(np.min(attrs[:, 3])),
                "home_lon_max": float(np.max(attrs[:, 3])),
                "demo_counts": demo_json,
            }
        )
        cbg_ids.append(region_id)

    pd.DataFrame(summaries).to_csv(out_root / "region_summary.csv", index=False)
    (out_root / "cbgs.txt").write_text("\n".join(cbg_ids) + "\n", encoding="utf-8")
    meta = {
        "out_root": str(out_root),
        "pi_json": str(pi_json),
        "q_source": q_source,
        "vocab_path": str(vocab_path),
        "num_special_tokens": int(num_special),
        "seq_len": int(seq_len),
        "num_traj_per_region": int(N),
        "coords_mode": coords_mode,
        "coords_pool_npy": None if args.coords_pool_npy is None else str(args.coords_pool_npy),
        "seed": int(args.seed),
        "regions_written": int(len(cbg_ids)),
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote mixture world to {out_root} ({len(cbg_ids)} regions)")


if __name__ == "__main__":
    main()

