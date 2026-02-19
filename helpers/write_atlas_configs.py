#!/usr/bin/env python3
"""
Write template YAML configs for:
  - trajectory-generation/scripts/build_poi_marginals.py
  - trajectory-generation/scripts/precompute/cache_cbg_conditionals.py

This script only *writes text*; it does not depend on PyYAML.
You still need PyYAML in the environment where you run the above scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-root", type=str, required=True, help="Root produced by make_atlas_world_from_split.py")
    ap.add_argument("--vocab-path", type=str, required=True, help="split_data.../controlled/tokenizer/vocab.txt")
    ap.add_argument("--out-config-dir", type=str, required=True, help="Where to write YAML templates")
    ap.add_argument("--out-aggregates-dir", type=str, required=True, help="Where build_poi_marginals writes p_poi.csv")
    ap.add_argument("--out-cache-dir", type=str, default=None, help="Where cache_cbg_conditionals writes <cbg>.npz")
    ap.add_argument("--num-special-tokens", type=int, default=5)
    ap.add_argument("--num-genders", type=int, default=2)
    args = ap.parse_args()

    world_root = Path(args.world_root).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    out_cfg = Path(args.out_config_dir).resolve()
    out_aggs = Path(args.out_aggregates_dir).resolve()
    out_cache = Path(args.out_cache_dir).resolve() if args.out_cache_dir else (out_aggs / "cbg_condition_cache")

    out_cfg.mkdir(parents=True, exist_ok=True)
    out_aggs.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    poi_yaml = f"""# Auto-generated template for ATLAS-world POI marginals
llm_world:
  npy_root: {world_root}
  files:
    poi_sequences: generated_sequences.npy
    demographics: all_attr_results.demographics.npy
    selected_indices: selected_indices.npy
  demo_source: demographics
  vocab_path: {vocab_path}
  num_special_tokens: {int(args.num_special_tokens)}
  attr_keys:
    age_key: age_id
    gender_key: gender_id
    num_genders: {int(args.num_genders)}

groups:
  cbgs: []   # leave empty to use all subfolders under npy_root
  demos: []  # leave empty to infer from data

stats:
  epsilon: 1.0e-6
  min_traj_per_group: 1

output:
  dir: {out_aggs}
  overwrite: true

runtime:
  verbose: true
  seed: 42
"""

    cache_yaml = f"""# Auto-generated template for ATLAS-world conditioning cache
llm_world:
  npy_root: {world_root}
  files:
    selected_indices: selected_indices.npy
    demographics: all_attr_results.demographics.npy

groups:
  cbgs: []   # leave empty to use all subfolders under npy_root

output:
  dir: {out_cache}
  overwrite: true

runtime:
  verbose: true
"""

    poi_path = out_cfg / "poi_marginals_atlas_world.yaml"
    cache_path = out_cfg / "cache_cbg_conditionals_atlas_world.yaml"
    poi_path.write_text(poi_yaml, encoding="utf-8")
    cache_path.write_text(cache_yaml, encoding="utf-8")

    print(f"[DONE] Wrote: {poi_path}")
    print(f"[DONE] Wrote: {cache_path}")
    print("Next:")
    print(f"  python trajectory-generation/scripts/build_poi_marginals.py --config {poi_path}")
    print(f"  python trajectory-generation/scripts/precompute/cache_cbg_conditionals.py --config {cache_path}")


if __name__ == "__main__":
    main()
