#!/usr/bin/env python3
"""
Build a mixture ATLAS-world over the 6 NYC GEOREGIONS (the georegion analogue of
06_build_atlas_world_fullrank.py, which mixes the 8 demo groups).

Source: atlas_world/foursquare_nyc_georegions/world_{split}_georegions/, whose
buckets are the 6 georegions (Manhattan_S/M/N, Brooklyn, Outer, New_Jersey).

Each output region is a mixture of georegions with a designed proportion vector
pi (the "label proportions" for LLP):
  - 6 near-one-hot regions  (dominant georegion mass = 0.85)
  - 6 pairwise regions      (0.5 / 0.5), from two edge-disjoint perfect matchings
The 6 one-hot rows already make pi full column-rank (6), so per-georegion
behaviour is identifiable from the aggregate regional proportions.

Trajectories are REAL (sampled from the georegion buckets, not synthesized);
each keeps its own demographics. Output layout matches build_poi_marginals.py /
cache_cbg_conditionals.py, exactly like the fullrank world.

Usage:
    python foursquare_preprocessing/07_build_atlas_world_georegion_mixture.py \
        --source_georegion_root atlas_world/foursquare_nyc_georegions \
        --out_root atlas_world/foursquare_nyc_georegion_mixture
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAJGEN = REPO_ROOT / "trajectory-generation"
HELPERS = REPO_ROOT / "helpers"
MAKE_MIXTURE = HELPERS / "make_atlas_mixture_world_from_demogroups.py"
WRITE_CONFIGS = HELPERS / "write_atlas_configs.py"
BUILD_LENGTHS = HELPERS / "build_length_dists_from_atlas_world.py"
BUILD_MARGINALS = TRAJGEN / "scripts" / "precompute" / "build_poi_marginals.py"
CACHE_CONDITIONALS = TRAJGEN / "scripts" / "precompute" / "cache_cbg_conditionals.py"

GEOREGIONS = ["Manhattan_S", "Manhattan_M", "Manhattan_N", "Brooklyn", "Outer", "New_Jersey"]


def _run(cmd: list, desc: str) -> None:
    print(f"\n{'=' * 60}\n  {desc}\n{'=' * 60}")
    if subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode != 0:
        raise SystemExit(f"[ERROR] {desc} failed")


def _two_edge_disjoint_matchings(n: int) -> List[Tuple[int, int]]:
    """Two edge-disjoint perfect matchings over n (even) nodes -> n pairs,
    each node used exactly twice. Deterministic (circle method)."""
    assert n % 2 == 0
    m1 = [(i, i + 1) for i in range(0, n, 2)]                     # (0,1)(2,3)(4,5)
    m2 = [(i, (i + 1) % n) for i in range(1, n, 2)]              # (1,2)(3,4)(5,0)
    return m1 + m2


def _design_pi(groups: List[str], dominant: float = 0.85) -> dict:
    G = len(groups)
    off = (1.0 - dominant) / (G - 1)
    regions = []
    # near-one-hot rows
    for g in groups:
        regions.append({
            "region_id": f"mix_hot_{g}",
            "pi": {x: (dominant if x == g else off) for x in groups},
        })
    # pairwise 50/50 rows
    for k, (i, j) in enumerate(_two_edge_disjoint_matchings(G)):
        gi, gj = groups[i], groups[j]
        regions.append({
            "region_id": f"mix_pair_{k:02d}_{gi}_{gj}",
            "pi": {x: (0.5 if x in (gi, gj) else 0.0) for x in groups},
        })
    return {"demo_keys": groups, "regions": regions}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NYC georegion mixture ATLAS world")
    ap.add_argument("--source_georegion_root", default="atlas_world/foursquare_nyc_georegions")
    ap.add_argument("--out_root", default="atlas_world/foursquare_nyc_georegion_mixture")
    ap.add_argument("--split_data_root", default="data/foursquare_nyc/controlled")
    ap.add_argument("--num_traj_per_region", type=int, default=None,
                    help="If omitted, auto-derive max feasible N (no reuse) per split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    source_root = Path(args.source_georegion_root).resolve()
    out_root = Path(args.out_root).resolve()
    vocab_path = Path(args.split_data_root).resolve() / "tokenizer" / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: design + write pi over the 6 georegions
    pi = _design_pi(GEOREGIONS)
    pi_json = out_root / "pi_matrix_georegion_6.json"
    pi_json.write_text(json.dumps(pi, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote pi matrix: {pi_json}  ({len(pi['regions'])} regions over {len(GEOREGIONS)} georegions)")

    # Step 2: sample mixture worlds from the georegion buckets (real trajectories)
    for split in ["train", "val", "test"]:
        source_world = source_root / f"world_{split}_georegions"
        out_world = out_root / f"world_{split}_mixture"
        cmd = [
            sys.executable, str(MAKE_MIXTURE),
            "--source-world-root", str(source_world),
            "--pi-json", str(pi_json),
            "--out-root", str(out_world),
            "--bucket-prefix", "",  # georegion dirs have no 'demo_' prefix
            "--seed", str(args.seed),
        ]
        if args.num_traj_per_region is not None:
            cmd += ["--num-traj-per-region", str(args.num_traj_per_region)]
        _run(cmd, desc=f"Build georegion mixture world ({split})")

    # Step 3: configs + marginals + caches per split
    for split in ["train", "val", "test"]:
        world_root = out_root / f"world_{split}_mixture"
        cfg_dir = out_root / "configs" / split
        agg_dir = out_root / "aggregates" / split
        cache_dir = out_root / "cache" / split
        for d in (cfg_dir, agg_dir, cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, str(WRITE_CONFIGS),
              "--world-root", str(world_root), "--vocab-path", str(vocab_path),
              "--out-config-dir", str(cfg_dir), "--out-aggregates-dir", str(agg_dir),
              "--out-cache-dir", str(cache_dir)],
             desc=f"Write georegion mixture configs ({split})")
        _run([sys.executable, str(BUILD_MARGINALS), "--config", str(cfg_dir / "poi_marginals_atlas_world.yaml")],
             desc=f"Build POI marginals ({split})")
        _run([sys.executable, str(CACHE_CONDITIONALS), "--config", str(cfg_dir / "cache_cbg_conditionals_atlas_world.yaml")],
             desc=f"Cache CBG conditionals ({split})")

    # Step 4: train length distributions
    _run([sys.executable, str(BUILD_LENGTHS),
          "--world-root", str(out_root / "world_train_mixture"),
          "--out-json", str(out_root / "length_dists_train_mixture.json"),
          "--max-length", str(args.max_length)],
         desc="Build length distributions (train)")

    print(f"\n{'=' * 60}\n  Georegion mixture world done -> {out_root}\n{'=' * 60}")


if __name__ == "__main__":
    main()
