#!/usr/bin/env python3
"""
Inference for DiT-only generation with demographic conditioning.

Outputs mirror `inference.py`, but `sampled_attributes.npy` contains
the full attribute vectors (6-D or 7-D when length conditioning is enabled).
"""

from __future__ import annotations

import argparse
import os
import sys

# Make repo modules importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PARENT_DIR)

from src.helpers import normalize_prediction_type  
from lib.infer_demo_run import run_inference_demo  


def parse_args():
    parser = argparse.ArgumentParser(
        description="inference conditioned on (home, work, age, gender)"
    )
    parser.add_argument("-d", "--model_dir", type=str, required=True, help="Directory containing trained DiT model")
    parser.add_argument("--model_file", type=str, default=None, help="Specific model file name (e.g., 'dit_final.pt')")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path (YAML) for DiT")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("-n", "--num_samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("-s", "--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("-g", "--guidance_scale", type=float, default=2.0, help="Guidance scale for sampling")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")

    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "logsnr", "logsnr_linear", "log-snr"])
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--logsnr_max", type=float, default=30.0)
    parser.add_argument("--logsnr_min", type=float, default=-2.0)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--latent_pca_path", type=str, default=None)
    parser.add_argument("--latent_mapper_path", type=str, default=None)
    parser.add_argument("--mapper_config", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default="epsilon")

    parser.add_argument("--max_traj_length", type=int, default=64, help="Maximum number of coordinate points to output")
    parser.add_argument("--min_traj_length", type=int, default=2)
    parser.add_argument("--autoencoder_path", type=str, required=True, help="Path to phase1 autoencoder checkpoint directory")

    # Data source for attributes
    parser.add_argument("--test_data_dir", type=str, required=True,
                        help="Root split_data directory (contains controlled/{split}/...) used to sample attrs and load mapping.")
    parser.add_argument("--data_split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--data_type", type=str, choices=["controlled", "uncontrolled", "unified"], default="controlled")
    parser.add_argument("--controlled_ratio", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=42)

    # Demo handling
    parser.add_argument("--keep_missing_demo", action="store_true", default=False,
                        help="Keep rows with age/gender == -1 by mapping them to null demo (unconditional demo).")
    parser.add_argument("--manual_demo", type=int, nargs=2, metavar=("AGE_BIN", "GENDER_ID"), default=None,
                        help="Override (age_bin, gender_id) for all samples (raw 0-based ids).")
    parser.add_argument("--zero_coords", action="store_true", default=False,
                        help="Zero out work/home coordinates (indices 0-3), keeping only demo conditioning. Useful for testing demo-only generation after unconditional pretrain + LLP finetune.")

    # Home/work token handling
    parser.add_argument("--poi_home_token", type=str, default="POI_HOME")
    parser.add_argument("--poi_work_token", type=str, default="POI_WORK")
    parser.add_argument("--poi_other_token", type=str, default="POI_OTHER")

    # Manual override for coords used for generation (also used to inject home/work token coords)
    parser.add_argument("--manual_home_work", type=float, nargs=4, metavar=("HOME_LAT", "HOME_LON", "WORK_LAT", "WORK_LON"),
                        default=None, help="Override attribute sampling with fixed home/work coords.")

    # Length control (optional)
    parser.add_argument("--enable_length_condition", action="store_true", default=False)
    parser.add_argument("--force_empirical_length", action="store_true", default=False)
    parser.add_argument("--length_vocab_size", type=int, default=513)
    parser.add_argument("--manual_length_id", type=int, default=None)

    # Generation knobs (same as inference script)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument("--unk_logit_penalty", type=float, default=0.0)
    parser.add_argument("--forbid_token_ids", type=str, nargs="?", default="")
    parser.add_argument("--forbid_poi_special_tokens", action="store_true", default=False,
                        help="Automatically forbid POI_HOME, POI_WORK, and POI_OTHER tokens during generation")
    parser.add_argument("--penalize_poi_special_tokens", type=float, default=0.0,
                        help="Penalty (logit subtraction) for POI_HOME, POI_WORK, and POI_OTHER tokens")
    parser.add_argument("--penalize_poi_home", type=float, default=0.0)
    parser.add_argument("--penalize_poi_work", type=float, default=0.0)
    parser.add_argument("--penalize_poi_other", type=float, default=0.0)
    parser.add_argument("--forbid_poi_other_only", action="store_true", default=False)
    parser.add_argument("--penalize_poi_home_work", type=float, default=0.0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="dit_infer_demo_results")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.prediction_type = normalize_prediction_type(getattr(args, "prediction_type", "epsilon"))
    run_inference_demo(args)


if __name__ == "__main__":
    main()
