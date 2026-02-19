#!/usr/bin/env python3
"""
Run inference + evaluation for a sequence of DiT checkpoints (dit_step_XXX.pt).

For each checkpoint:
  1) Runs inference with inference_dit_with_demo_carlos.py
  2) Runs evaluate_carlos_by_demo_group.py on the outputs
  3) Aggregates JSD metrics into a summary CSV/JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


CKPT_PATTERN = re.compile(r"^cbg_finetune_step_(\d+)\.pt$")


def _find_checkpoints(ckpt_dir: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = CKPT_PATTERN.match(p.name)
        if m:
            step = int(m.group(1))
            found.append((step, p))
    found.sort(key=lambda x: x[0])
    return found


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _output_ready(infer_dir: Path) -> bool:
    return (infer_dir / "generated_poi_sequences.pkl").exists() and (infer_dir / "sampled_attributes.npy").exists()


def _aggregate_jsd(metrics_csv: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(metrics_csv)
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return []
    metric_series = df["metric"].astype(str)
    jsd_df = df[
        metric_series.str.contains("jsd", case=False, na=False)
        | metric_series.str.contains(r"^pattern_score$", case=False, na=False, regex=True)
    ]
    if jsd_df.empty:
        return []
    grouped = jsd_df.groupby("metric", as_index=False)["value"].agg(["mean", "count"]).reset_index()
    rows: List[Dict[str, object]] = []
    for _, r in grouped.iterrows():
        rows.append(
            {
                "metric": str(r["metric"]),
                "value_mean": float(r["mean"]),
                "n_groups": int(r["count"]),
            }
        )
    return rows


def _collect_jsd_by_group(metrics_csv: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(metrics_csv)
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return []
    metric_series = df["metric"].astype(str)
    jsd_df = df[
        metric_series.str.contains("jsd", case=False, na=False)
        | metric_series.str.contains(r"^pattern_score$", case=False, na=False, regex=True)
    ].copy()
    if jsd_df.empty:
        return []
    cols = ["metric", "value", "key", "age_bin", "gender_id", "n_real_raw", "n_model_raw", "n_real_mapped", "n_model_mapped"]
    available = [c for c in cols if c in jsd_df.columns]
    return jsd_df[available].to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dit_step_*.pt checkpoints using Carlos demo evaluator")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing cbg_finetune_step_*.pt")
    parser.add_argument("--config", type=str, required=True, help="DiT config YAML")
    parser.add_argument("--autoencoder_path", type=str, required=True, help="Phase-1 autoencoder checkpoint dir")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Root split_data dir")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_type", type=str, default="controlled", choices=["controlled", "uncontrolled", "unified"])

    parser.add_argument("--real_poi_pkl", type=str, required=True, help="final_segments_all_train_data.pkl")
    parser.add_argument("--real_attr_with_demo_npy", type=str, required=True, help="all_attr_results_with_demo.npy")
    parser.add_argument("--poi_map_csv", type=str, required=True, help="poi_map_feature.csv")
    parser.add_argument("--group_by", type=str, default="age_gender", choices=["age_gender", "age"])
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--inference_script", type=str, default=None, help="Path to inference script (optional)")
    parser.add_argument("--eval_script", type=str, default=None, help="Path to eval script (optional)")

    # Evaluation histogram/grid parameters (passed to evaluate_carlos_by_demo_group.py)
    parser.add_argument("--histogram_bins", type=int, default=40, help="Histogram bins for evaluation (default: 10000)")
    parser.add_argument("--spatial_bins", type=int, default=40, help="Spatial bins for evaluation (default: 10000)")
    parser.add_argument("--grid_size", type=int, default=40, help="Grid size for evaluation (default: 20)")

    parser.add_argument("--num_samples", type=int, default=9000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--force_cpu", action="store_true")

    parser.add_argument("--output_root", type=str, default="ckpt_eval_outputs")
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if outputs already exist")
    parser.add_argument("--continue_on_error", action="store_true")

    parser.add_argument(
        "--inference_extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to inference_dit_with_demo_carlos.py (use after --)",
    )
    parser.add_argument(
        "--eval_extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to evaluate_carlos_by_demo_group.py (use after --)",
    )

    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = _find_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching dit_step_*.pt in {ckpt_dir}")

    script_dir = Path(__file__).resolve().parent
    default_inference = script_dir.parent / "inference" / "inference_dit_with_demo_carlos.py"
    default_eval = script_dir / "evaluate_carlos_by_demo_group.py"

    inference_script = Path(args.inference_script) if args.inference_script else default_inference
    eval_script = Path(args.eval_script) if args.eval_script else default_eval

    summary_rows: List[Dict[str, object]] = []
    per_group_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for step, ckpt_path in checkpoints:
        step_dir = output_root / f"step_{step}"
        infer_dir = step_dir / "inference"
        eval_dir = step_dir / "evaluation"
        infer_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not (args.skip_existing and _output_ready(infer_dir)):
                cmd = [
                    sys.executable,
                    str(inference_script),
                    "--model_dir",
                    str(ckpt_dir),
                    "--model_file",
                    str(ckpt_path.name),
                    "--config",
                    str(args.config),
                    "--autoencoder_path",
                    str(args.autoencoder_path),
                    "--test_data_dir",
                    str(args.test_data_dir),
                    "--data_split",
                    str(args.data_split),
                    "--data_type",
                    str(args.data_type),
                    "--output_dir",
                    str(infer_dir),
                    "--num_samples",
                    str(args.num_samples),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_steps",
                    str(args.num_steps),
                    "--guidance_scale",
                    str(args.guidance_scale),
                ]
                if args.force_cpu:
                    cmd.append("--force_cpu")
                if args.inference_extra:
                    cmd.extend(args.inference_extra)
                _run(cmd)

            eval_cmd = [
                sys.executable,
                str(eval_script),
                "--real_poi_pkl",
                str(args.real_poi_pkl),
                "--real_attr_with_demo_npy",
                str(args.real_attr_with_demo_npy),
                "--synthetic_poi_pkl",
                str(infer_dir / "generated_poi_sequences.pkl"),
                "--synthetic_attr_npy",
                str(infer_dir / "sampled_attributes.npy"),
                "--synthetic_name",
                f"step_{step}",
                "--poi_map_csv",
                str(args.poi_map_csv),
                "--save_dir",
                str(eval_dir),
                "--group_by",
                str(args.group_by),
                "--min_count",
                str(args.min_count),
            ]
            if args.histogram_bins is not None:
                eval_cmd.extend(["--histogram_bins", str(args.histogram_bins)])
            if args.spatial_bins is not None:
                eval_cmd.extend(["--spatial_bins", str(args.spatial_bins)])
            if args.grid_size is not None:
                eval_cmd.extend(["--grid_size", str(args.grid_size)])
            if args.eval_extra:
                eval_cmd.extend(args.eval_extra)
            _run(eval_cmd)

            metrics_csv = eval_dir / "demo_group_metrics.csv"
            agg_rows = _aggregate_jsd(metrics_csv)
            for row in agg_rows:
                row.update(
                    {
                        "step": int(step),
                        "checkpoint": str(ckpt_path),
                        "eval_metrics_csv": str(metrics_csv),
                    }
                )
                summary_rows.append(row)

            group_rows = _collect_jsd_by_group(metrics_csv)
            for row in group_rows:
                row.update(
                    {
                        "step": int(step),
                        "checkpoint": str(ckpt_path),
                        "eval_metrics_csv": str(metrics_csv),
                    }
                )
                per_group_rows.append(row)
        except Exception as exc:
            err = {"step": int(step), "checkpoint": str(ckpt_path), "error": str(exc)}
            error_rows.append(err)
            if not args.continue_on_error:
                raise

    summary_csv = output_root / "ckpt_jsd_summary.csv"
    summary_json = output_root / "ckpt_jsd_summary.json"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    per_group_csv = output_root / "ckpt_jsd_by_group.csv"
    per_group_json = output_root / "ckpt_jsd_by_group.json"
    pd.DataFrame(per_group_rows).to_csv(per_group_csv, index=False)
    with open(per_group_json, "w", encoding="utf-8") as f:
        json.dump(per_group_rows, f, indent=2)

    if error_rows:
        error_json = output_root / "ckpt_eval_errors.json"
        with open(error_json, "w", encoding="utf-8") as f:
            json.dump(error_rows, f, indent=2)
        print(f"[WARN] Some steps failed; errors saved to {error_json}")

    print(
        "Done. Summary saved to "
        f"{summary_csv} and {summary_json}. "
        f"Per-group saved to {per_group_csv} and {per_group_json}"
    )


if __name__ == "__main__":
    main()

