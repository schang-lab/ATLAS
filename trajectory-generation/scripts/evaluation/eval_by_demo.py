#!/usr/bin/env python3
"""
Evaluation stratified by demographic group (age_bin, gender_id).
"""

from __future__ import annotations

import argparse
import os
import sys

# Make repo modules importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from lib.eval_by_demo_run import run_eval_by_demo  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation grouped by demo (age_bin, gender_id)")
    parser.add_argument("--real_poi_pkl", type=str, required=True, help="final_segments_all_train_data.pkl")
    parser.add_argument("--real_attr_with_demo_npy", type=str, required=True, help="all_attr_results_with_demo.npy aligned with real_poi_pkl")
    parser.add_argument("--synthetic_poi_pkl", type=str, required=True, help="generated_poi_sequences.pkl")
    parser.add_argument("--synthetic_attr_npy", type=str, required=True, help="sampled_attributes.npy aligned with synthetic_poi_pkl")
    parser.add_argument("--synthetic_name", type=str, default="model", help="Label for synthetic model in outputs")

    parser.add_argument("--synthetic2_poi_pkl", type=str, default=None)
    parser.add_argument("--synthetic2_attr_npy", type=str, default=None)
    parser.add_argument("--synthetic2_name", type=str, default="model2")

    parser.add_argument("--poi_map_csv", type=str, required=True, help="poi_map_feature.csv for POI->(lat,lon) mapping")
    parser.add_argument("--save_dir", type=str, default="demo_group_eval")

    parser.add_argument("--group_by", type=str, choices=["age_gender", "age"], default="age_gender")
    parser.add_argument("--min_count", type=int, default=200)
    parser.add_argument(
        "--drop_missing_demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), filters rows with age_bin<0 or gender_id<0 before grouping.",
    )

    parser.add_argument("--poi_other_token", type=str, default="POI_OTHER")
    parser.add_argument("--poi_home_token", type=str, default="POI_HOME")
    parser.add_argument("--poi_work_token", type=str, default="POI_WORK")
    parser.add_argument("--min_mapped_pois", type=int, default=2)

    parser.add_argument("--histogram_bins", type=int, default=10000)
    parser.add_argument("--spatial_bins", type=int, default=10000)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--enable_wasserstein", action="store_true", default=False)
    parser.add_argument("--wasserstein_subsample", type=int, default=10000)

    parser.add_argument(
        "--normalize_home_work",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, evaluate in a home-centered, work-aligned local frame (x/y in km).",
    )
    parser.add_argument(
        "--normalize_scale_by_commute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set with --normalize_home_work, also divide by ||home->work|| so commute length is ~1.",
    )
    parser.add_argument(
        "--drop_home_work_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, drop POI_HOME/POI_WORK points from trajectories before metric computation.",
    )
    parser.add_argument(
        "--poi_frequency_exclude_home_work_other",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, exclude POI_HOME, POI_WORK, and POI_OTHER from poi_frequency_jsd calculation (matching category-based metrics).",
    )
    parser.add_argument(
        "--include_poi_other_in_length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, length_jsd is computed from raw sequence lengths (including POI_OTHER) instead of mapped coordinate counts.",
    )
    parser.add_argument(
        "--category_include_home_work_other",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, include POI_HOME, POI_WORK, and POI_OTHER in poi_category_jsd and poi_category_transition_jsd calculations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval_by_demo(args)


if __name__ == "__main__":
    main()
