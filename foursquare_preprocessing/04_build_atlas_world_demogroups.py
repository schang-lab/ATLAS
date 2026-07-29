#!/usr/bin/env python3
"""
Build an ATLAS-world for Foursquare NYC split data using demographic groups as regions.

This is the NYC analogue of `04_build_atlas_world_demogroups.py`, but defaults to the
`split_data_Foursquare_NYC` layout and 4-age-bin x 2-gender demo groups.

Outputs:
  - world_{train,val,test}_demogroups/
  - aggregates/{train,val,test}/p_poi.csv
  - cache/{train,val,test}/*.npz
  - configs/{train,val,test}/{poi_marginals,cache_cbg_conditionals}_atlas_world.yaml
  - length_dists_train_demogroups.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAJGEN = REPO_ROOT / "trajectory-generation"
HELPERS = REPO_ROOT / "helpers"
MAKE_WORLD = HELPERS / "make_atlas_world_from_split.py"
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


def _write_poi_marginals_yaml(path: Path, *, world_root: str, vocab_path: str, output_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Auto-generated for Foursquare NYC ATLAS-world POI marginals
llm_world:
  npy_root: {world_root}
  files:
    poi_sequences: generated_sequences.npy
    demographics: all_attr_results.demographics.npy
    selected_indices: selected_indices.npy
  demo_source: demographics
  vocab_path: {vocab_path}
  num_special_tokens: 5
  attr_keys:
    age_key: age_id
    gender_key: gender_id
    num_genders: 2

groups:
  cbgs: []
  demos: []

stats:
  epsilon: 1.0e-6
  min_traj_per_group: 1

output:
  dir: {output_dir}
  overwrite: true

runtime:
  verbose: true
  seed: 42
""",
        encoding="utf-8",
    )


def _write_cache_conditionals_yaml(path: Path, *, world_root: str, output_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Auto-generated for Foursquare NYC ATLAS-world conditioning cache
llm_world:
  npy_root: {world_root}
  files:
    selected_indices: selected_indices.npy
    demographics: all_attr_results.demographics.npy

groups:
  cbgs: []

output:
  dir: {output_dir}
  overwrite: true

runtime:
  verbose: true
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ATLAS world for Foursquare NYC data")
    parser.add_argument(
        "--split_data_root",
        default="data/foursquare_nyc/controlled",
        help="Root containing train/val/test and tokenizer/",
    )
    parser.add_argument(
        "--out_root",
        default="atlas_world/foursquare_nyc_demogroups",
        help="Output ATLAS-world root",
    )
    parser.add_argument(
        "--min_region_size",
        type=int,
        default=25,
        help="Minimum trajectories per demographic region",
    )
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_root = Path(args.split_data_root).resolve()
    out_root = Path(args.out_root).resolve()
    vocab_path = split_root / "tokenizer" / "vocab.txt"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    if not MAKE_WORLD.exists():
        raise FileNotFoundError(f"Missing builder script: {MAKE_WORLD}")

    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} not found")
            continue

        world_dir = out_root / f"world_{split}_demogroups"
        _run(
            [
                sys.executable,
                str(MAKE_WORLD),
                "--split-root",
                str(split_dir),
                "--tokenizer-vocab",
                str(vocab_path),
                "--out-root",
                str(world_dir),
                "--region-mode",
                "demo_group",
                "--min-region-size",
                str(args.min_region_size),
                "--seed",
                str(args.seed),
            ],
            desc=f"Build NYC ATLAS world ({split})",
        )

    train_world = out_root / "world_train_demogroups"
    length_json = out_root / "length_dists_train_demogroups.json"
    if train_world.exists():
        _run(
            [
                sys.executable,
                str(BUILD_LENGTHS),
                "--world-root",
                str(train_world),
                "--out-json",
                str(length_json),
                "--max-length",
                str(args.max_length),
                "--num-special-tokens",
                "5",
                "--min-length",
                "1",
                "--clip",
            ],
            desc="Build NYC length distributions (train)",
        )

    configs_dir = out_root / "configs"
    for split in ["train", "val", "test"]:
        world_dir = out_root / f"world_{split}_demogroups"
        if not world_dir.exists():
            continue
        _write_poi_marginals_yaml(
            configs_dir / split / "poi_marginals_atlas_world.yaml",
            world_root=str(world_dir),
            vocab_path=str(vocab_path),
            output_dir=str(out_root / "aggregates" / split),
        )
        _write_cache_conditionals_yaml(
            configs_dir / split / "cache_cbg_conditionals_atlas_world.yaml",
            world_root=str(world_dir),
            output_dir=str(out_root / "cache" / split),
        )

    for split in ["train", "val", "test"]:
        cfg_dir = configs_dir / split
        marginals_cfg = cfg_dir / "poi_marginals_atlas_world.yaml"
        cache_cfg = cfg_dir / "cache_cbg_conditionals_atlas_world.yaml"
        if marginals_cfg.exists():
            _run(
                [sys.executable, str(BUILD_MARGINALS), "--config", str(marginals_cfg)],
                desc=f"Build NYC POI marginals ({split})",
            )
        if cache_cfg.exists():
            _run(
                [sys.executable, str(CACHE_CONDITIONALS), "--config", str(cache_cfg)],
                desc=f"Build NYC conditioning cache ({split})",
            )

    print(f"\n{'=' * 60}")
    print("  NYC LLP World Summary")
    print(f"{'=' * 60}")
    print(f"Output root: {out_root}")
    if length_json.exists():
        with open(length_json, "r", encoding="utf-8") as f:
            length_info = json.load(f)
        print(f"Length regions: {list(length_info.keys())}")


if __name__ == "__main__":
    main()
