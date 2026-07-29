#!/usr/bin/env python3
"""
Build a full-rank NYC LLP mixture world from the NYC demo-group world.

Pipeline:
  1) Design a full-rank pi matrix over the 8 NYC demo groups.
  2) Build train/val/test mixture worlds from the demo-group source worlds.
  3) Write aggregate/cache configs.
  4) Build POI marginals, conditioning caches, and train length distributions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAJGEN = REPO_ROOT / "trajectory-generation"
HELPERS = REPO_ROOT / "helpers"
DESIGN_PI = HELPERS / "design_mixture_pi.py"
MAKE_MIXTURE = HELPERS / "make_atlas_mixture_world_from_demogroups.py"
WRITE_CONFIGS = HELPERS / "write_atlas_configs.py"
BUILD_LENGTHS = HELPERS / "build_length_dists_from_atlas_world.py"
BUILD_MARGINALS = TRAJGEN / "scripts" / "precompute" / "build_poi_marginals.py"
CACHE_CONDITIONALS = TRAJGEN / "scripts" / "precompute" / "cache_cbg_conditionals.py"


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-rank NYC ATLAS world")
    parser.add_argument(
        "--source_demo_root",
        default="atlas_world/foursquare_nyc_demogroups",
        help="Root containing world_{train,val,test}_demogroups",
    )
    parser.add_argument(
        "--out_root",
        default="atlas_world/foursquare_nyc_fullrank",
        help="Output root for full-rank mixture world",
    )
    parser.add_argument(
        "--split_data_root",
        default="data/foursquare_nyc/controlled",
        help="Used only for tokenizer vocab path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_hot", type=int, default=8)
    parser.add_argument("--num_pair", type=int, default=8)
    args = parser.parse_args()

    source_demo_root = Path(args.source_demo_root).resolve()
    out_root = Path(args.out_root).resolve()
    split_data_root = Path(args.split_data_root).resolve()
    vocab_path = split_data_root / "tokenizer" / "vocab.txt"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    pi_json = out_root / "pi_matrix_full_rank_8.json"
    pi_csv = out_root / "pi_matrix_full_rank_8.csv"
    _run(
        [
            sys.executable,
            str(DESIGN_PI),
            "--out-json",
            str(pi_json),
            "--out-csv",
            str(pi_csv),
            "--seed",
            str(args.seed),
            "--num-hot",
            str(args.num_hot),
            "--num-pair",
            str(args.num_pair),
            "--prefix",
            "mix",
        ],
        desc="Design NYC full-rank pi matrix",
    )

    for split in ["train", "val", "test"]:
        source_world = source_demo_root / f"world_{split}_demogroups"
        out_world = out_root / f"world_{split}_mixture"
        _run(
            [
                sys.executable,
                str(MAKE_MIXTURE),
                "--source-world-root",
                str(source_world),
                "--pi-json",
                str(pi_json),
                "--out-root",
                str(out_world),
                "--seed",
                str(args.seed),
            ],
            desc=f"Build NYC full-rank mixture world ({split})",
        )

    for split in ["train", "val", "test"]:
        world_root = out_root / f"world_{split}_mixture"
        cfg_dir = out_root / "configs" / split
        agg_dir = out_root / "aggregates" / split
        cache_dir = out_root / "cache" / split
        cfg_dir.mkdir(parents=True, exist_ok=True)
        agg_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                sys.executable,
                str(WRITE_CONFIGS),
                "--world-root",
                str(world_root),
                "--vocab-path",
                str(vocab_path),
                "--out-config-dir",
                str(cfg_dir),
                "--out-aggregates-dir",
                str(agg_dir),
                "--out-cache-dir",
                str(cache_dir),
            ],
            desc=f"Write NYC full-rank configs ({split})",
        )

        _run(
            [sys.executable, str(BUILD_MARGINALS), "--config", str(cfg_dir / "poi_marginals_atlas_world.yaml")],
            desc=f"Build NYC full-rank POI marginals ({split})",
        )
        _run(
            [sys.executable, str(CACHE_CONDITIONALS), "--config", str(cfg_dir / "cache_cbg_conditionals_atlas_world.yaml")],
            desc=f"Build NYC full-rank conditioning cache ({split})",
        )

    _run(
        [
            sys.executable,
            str(BUILD_LENGTHS),
            "--world-root",
            str(out_root / "world_train_mixture"),
            "--out-json",
            str(out_root / "length_dists_train_demogroups.json"),
            "--max-length",
            str(args.max_length),
        ],
        desc="Build NYC full-rank length distributions",
    )


if __name__ == "__main__":
    main()
