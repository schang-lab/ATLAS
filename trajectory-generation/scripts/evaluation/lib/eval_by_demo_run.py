"""
Execution pipeline for evaluation stratified by demographic group.
This module orchestrates evaluation and delegates IO/mapping/metrics to submodules.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .eval_by_demo_deps import MappingConfig, load_poi_coords
from .eval_by_demo_io import (
    CATEGORY_COLUMN_DEFAULT,
    _format_group_keys,
    _group_keys,
    _load_attrs_with_demo,
    _load_demo_from_attrs,
    _load_poi_sequences_from_pkl,
    _load_poi_to_category,
    _summarize_demo_counts,
    _unpack_key,
)
from .eval_by_demo_mapping import _map_sequences_to_coords_embee, _normalize_traj_home_work_frame
from .eval_by_demo_metrics import (
    _compute_metrics_for_group,
    _compute_poi_category_jsd,
    _compute_poi_category_transition_jsd,
    _compute_poi_frequency_jsd,
)


def _eval_one_model(
    *,
    model_name: str,
    real_seqs: List[List[str]],
    real_attrs: np.ndarray,
    real_demo: np.ndarray,
    syn_seqs: List[List[str]],
    syn_attrs: np.ndarray,
    syn_demo: np.ndarray,
    poi_coords: Dict[str, Tuple[float, float]],
    cfg: MappingConfig,
    group_by: str,
    min_count: int,
    histogram_bins: int,
    spatial_bins: int,
    grid_size: int,
    top_n: int,
    enable_wasserstein: bool,
    wasserstein_subsample: int,
    drop_missing_demo: bool,
    drop_home_work_points: bool,
    normalize_home_work: bool,
    normalize_scale_by_commute: bool,
    poi_to_category: Dict[str, str],
    category_to_index: Dict[str, int],
    num_categories: int,
    poi_frequency_exclude_home_work_other: bool = False,
    include_poi_other_in_length: bool = False,
    category_include_home_work_other: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    real_demo2 = real_demo[:, :2].astype(np.int64, copy=False)
    syn_demo2 = syn_demo[:, :2].astype(np.int64, copy=False)

    real_mask = np.ones((real_demo2.shape[0],), dtype=bool)
    syn_mask = np.ones((syn_demo2.shape[0],), dtype=bool)
    if drop_missing_demo:
        real_mask &= (real_demo2[:, 0] >= 0) & (real_demo2[:, 1] >= 0)
        syn_mask &= (syn_demo2[:, 0] >= 0) & (syn_demo2[:, 1] >= 0)

    real_keys = _group_keys(real_demo2[real_mask], group_by)
    syn_keys = _group_keys(syn_demo2[syn_mask], group_by)

    real_idx_by_key: Dict[int, np.ndarray] = {}
    for k in np.unique(real_keys):
        real_idx_by_key[int(k)] = np.where(real_keys == k)[0].astype(np.int64)
    syn_idx_by_key: Dict[int, np.ndarray] = {}
    for k in np.unique(syn_keys):
        syn_idx_by_key[int(k)] = np.where(syn_keys == k)[0].astype(np.int64)

    real_kept_idx = np.where(real_mask)[0].astype(np.int64)
    syn_kept_idx = np.where(syn_mask)[0].astype(np.int64)
    common_keys = sorted(set(real_idx_by_key.keys()) & set(syn_idx_by_key.keys()))

    real_counts = _summarize_demo_counts(real_keys, group_by=group_by)
    syn_counts = _summarize_demo_counts(syn_keys, group_by=group_by)
    all_keys = sorted(set(real_counts.keys()) | set(syn_counts.keys()))
    missing_in_real = [k for k in all_keys if k not in real_counts]
    missing_in_syn = [k for k in all_keys if k not in syn_counts]
    low_count = [
        k
        for k in common_keys
        if real_counts.get(k, 0) < int(min_count) or syn_counts.get(k, 0) < int(min_count)
    ]

    if missing_in_real:
        print(
            f"[WARN] {model_name}: groups missing in real (will not be evaluated): "
            f"{_format_group_keys(missing_in_real, group_by=group_by)}"
        )
    if missing_in_syn:
        print(
            f"[WARN] {model_name}: groups missing in model (will not be evaluated): "
            f"{_format_group_keys(missing_in_syn, group_by=group_by)}"
        )
    if low_count:
        print(
            f"[WARN] {model_name}: groups below --min_count={min_count} (will be skipped): "
            f"{_format_group_keys(low_count, group_by=group_by)}"
        )

    metric_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []

    for key in common_keys:
        real_sel = real_kept_idx[real_idx_by_key[key]]
        syn_sel = syn_kept_idx[syn_idx_by_key[key]]
        n_real = int(real_sel.size)
        n_syn = int(syn_sel.size)
        if n_real < min_count or n_syn < min_count:
            continue

        real_seqs_g = [real_seqs[i] for i in real_sel.tolist()]
        syn_seqs_g = [syn_seqs[i] for i in syn_sel.tolist()]
        real_attrs_4d_g = real_attrs[real_sel, :4] if real_attrs.shape[1] >= 4 else None
        syn_attrs_4d_g = syn_attrs[syn_sel, :4] if syn_attrs.shape[1] >= 4 else None

        real_trajs, real_stats, real_hw, real_kept_seqs = _map_sequences_to_coords_embee(
            sequences=real_seqs_g,
            poi_coords=poi_coords,
            attrs_4d=real_attrs_4d_g,
            cfg=cfg,
            label=f"real/{key}",
            drop_home_work_points=bool(drop_home_work_points),
        )
        syn_trajs, syn_stats, syn_hw, syn_kept_seqs = _map_sequences_to_coords_embee(
            sequences=syn_seqs_g,
            poi_coords=poi_coords,
            attrs_4d=syn_attrs_4d_g,
            cfg=cfg,
            label=f"{model_name}/{key}",
            drop_home_work_points=bool(drop_home_work_points),
        )

        travel_mode = "haversine"
        if normalize_home_work:
            travel_mode = "euclidean"

            def _norm_list(trajs: List[np.ndarray], hw: List[Tuple[float, float, float, float]]) -> List[np.ndarray]:
                out: List[np.ndarray] = []
                for t, (hlat, hlon, wlat, wlon) in zip(trajs, hw):
                    if not (np.isfinite(hlat) and np.isfinite(hlon) and np.isfinite(wlat) and np.isfinite(wlon)):
                        out.append(t)
                        continue
                    out.append(
                        _normalize_traj_home_work_frame(
                            t,
                            home_lat=float(hlat),
                            home_lon=float(hlon),
                            work_lat=float(wlat),
                            work_lon=float(wlon),
                            scale_by_commute=bool(normalize_scale_by_commute),
                        )
                    )
                return out

            real_trajs = _norm_list(real_trajs, real_hw)
            syn_trajs = _norm_list(syn_trajs, syn_hw)

        age_bin, gender_id = _unpack_key(int(key), group_by)
        if len(real_trajs) == 0 or len(syn_trajs) == 0:
            print(
                f"[WARN] {model_name}: group (age_bin,gender_id)=({age_bin},{gender_id}) "
                f"has zero mapped sequences (real={int(len(real_trajs))}, model={int(len(syn_trajs))}); "
                "metrics will be skipped for this group. "
                f"(raw counts: real={n_real}, model={n_syn})"
            )
        stats_rows.append(
            {
                "model": model_name,
                "group_by": group_by,
                "coord_frame": ("home_work_frame" if normalize_home_work else "latlon"),
                "travel_distance_mode": travel_mode,
                "drop_home_work_points": bool(drop_home_work_points),
                "normalize_scale_by_commute": bool(normalize_scale_by_commute),
                "key": int(key),
                "age_bin": age_bin,
                "gender_id": gender_id,
                "n_real_raw": n_real,
                "n_model_raw": n_syn,
                "real_kept_sequences": int(real_stats.get("kept_sequences", 0.0)),
                "model_kept_sequences": int(syn_stats.get("kept_sequences", 0.0)),
                "real_dropped_too_short": int(real_stats.get("dropped_sequences_too_short", 0.0)),
                "model_dropped_too_short": int(syn_stats.get("dropped_sequences_too_short", 0.0)),
                "real_dropped_home_work_tokens": int(real_stats.get("dropped_home_work_tokens", 0.0)),
                "model_dropped_home_work_tokens": int(syn_stats.get("dropped_home_work_tokens", 0.0)),
            }
        )

        metrics = _compute_metrics_for_group(
            real_trajs,
            syn_trajs,
            histogram_bins=histogram_bins,
            spatial_bins=spatial_bins,
            grid_size=grid_size,
            top_n=top_n,
            enable_wasserstein=enable_wasserstein,
            wasserstein_subsample=wasserstein_subsample,
            travel_distance_mode=travel_mode,
            real_seqs=real_kept_seqs if include_poi_other_in_length else None,
            syn_seqs=syn_kept_seqs if include_poi_other_in_length else None,
            cfg=cfg if include_poi_other_in_length else None,
            include_poi_other_in_length=bool(include_poi_other_in_length),
        )

        poi_freq_jsd = _compute_poi_frequency_jsd(
            real_seqs_g, syn_seqs_g, cfg=cfg, exclude_home_work_other=poi_frequency_exclude_home_work_other
        )
        if np.isfinite(poi_freq_jsd):
            metrics["poi_frequency_jsd"] = poi_freq_jsd

        cat_jsd = _compute_poi_category_jsd(
            real_seqs_g,
            syn_seqs_g,
            cfg=cfg,
            poi_to_category=poi_to_category,
            category_to_index=category_to_index,
            num_categories=int(num_categories),
            include_home_work_other=bool(category_include_home_work_other),
        )
        if np.isfinite(cat_jsd):
            metrics["poi_category_jsd"] = float(cat_jsd)

        cat_tr_jsd = _compute_poi_category_transition_jsd(
            real_seqs_g,
            syn_seqs_g,
            cfg=cfg,
            poi_to_category=poi_to_category,
            category_to_index=category_to_index,
            num_categories=int(num_categories),
            include_home_work_other=bool(category_include_home_work_other),
        )
        if np.isfinite(cat_tr_jsd):
            metrics["poi_category_transition_jsd"] = float(cat_tr_jsd)

        if not metrics:
            continue
        for metric, value in metrics.items():
            metric_rows.append(
                {
                    "model": model_name,
                    "group_by": group_by,
                    "coord_frame": ("home_work_frame" if normalize_home_work else "latlon"),
                    "key": int(key),
                    "age_bin": age_bin,
                    "gender_id": gender_id,
                    "metric": metric,
                    "value": float(value),
                    "n_real_raw": n_real,
                    "n_model_raw": n_syn,
                    "n_real_mapped": int(len(real_trajs)),
                    "n_model_mapped": int(len(syn_trajs)),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(stats_rows)


def run_eval_by_demo(args) -> None:
    """Run by-demo evaluation: parse args -> load data -> evaluate -> save outputs."""
    os.makedirs(args.save_dir, exist_ok=True)
    if args.drop_missing_demo:
        print("[INFO] drop_missing_demo: enabled (filters age_bin<0 or gender_id<0 before grouping)")
    else:
        print("[INFO] drop_missing_demo: disabled (will include missing demo ids as their own group keys)")

    real_seqs = _load_poi_sequences_from_pkl(args.real_poi_pkl, kind="real")
    real_attrs = _load_attrs_with_demo(args.real_attr_with_demo_npy, name="real_attr_with_demo_npy")
    if len(real_seqs) != real_attrs.shape[0]:
        raise ValueError(f"real_poi_pkl rows ({len(real_seqs)}) != real_attr_with_demo_npy rows ({real_attrs.shape[0]})")
    real_demo = _load_demo_from_attrs(real_attrs, name="real_attr_with_demo_npy")

    syn_seqs = _load_poi_sequences_from_pkl(args.synthetic_poi_pkl, kind="synthetic")
    syn_attrs = _load_attrs_with_demo(args.synthetic_attr_npy, name="synthetic_attr_npy")
    if len(syn_seqs) != syn_attrs.shape[0]:
        raise ValueError(f"synthetic_poi_pkl rows ({len(syn_seqs)}) != synthetic_attr_npy rows ({syn_attrs.shape[0]})")
    syn_demo = _load_demo_from_attrs(syn_attrs, name="synthetic_attr_npy")

    syn2_seqs: Optional[List[List[str]]] = None
    syn2_attrs: Optional[np.ndarray] = None
    syn2_demo: Optional[np.ndarray] = None
    has2 = bool(args.synthetic2_poi_pkl) and bool(args.synthetic2_attr_npy)
    if has2:
        syn2_seqs = _load_poi_sequences_from_pkl(args.synthetic2_poi_pkl, kind="synthetic")
        syn2_attrs = _load_attrs_with_demo(args.synthetic2_attr_npy, name="synthetic2_attr_npy")
        if len(syn2_seqs) != syn2_attrs.shape[0]:
            raise ValueError(f"synthetic2_poi_pkl rows ({len(syn2_seqs)}) != synthetic2_attr_npy rows ({syn2_attrs.shape[0]})")
        syn2_demo = _load_demo_from_attrs(syn2_attrs, name="synthetic2_attr_npy")

    poi_coords = load_poi_coords(args.poi_map_csv)
    poi_to_category, categories, category_to_index = _load_poi_to_category(
        args.poi_map_csv,
        category_column=CATEGORY_COLUMN_DEFAULT,
    )
    cfg = MappingConfig(
        poi_other_token=args.poi_other_token,
        poi_home_token=args.poi_home_token,
        poi_work_token=args.poi_work_token,
        min_mapped_pois=int(args.min_mapped_pois),
    )

    def _eval_kw() -> dict:
        return {
            "real_seqs": real_seqs,
            "real_attrs": real_attrs,
            "real_demo": real_demo,
            "poi_coords": poi_coords,
            "cfg": cfg,
            "group_by": str(args.group_by),
            "min_count": int(args.min_count),
            "histogram_bins": int(args.histogram_bins),
            "spatial_bins": int(args.spatial_bins),
            "grid_size": int(args.grid_size),
            "top_n": int(args.top_n),
            "enable_wasserstein": bool(args.enable_wasserstein),
            "wasserstein_subsample": int(args.wasserstein_subsample),
            "drop_missing_demo": bool(args.drop_missing_demo),
            "drop_home_work_points": bool(args.drop_home_work_points),
            "normalize_home_work": bool(args.normalize_home_work),
            "normalize_scale_by_commute": bool(args.normalize_scale_by_commute),
            "poi_to_category": poi_to_category,
            "category_to_index": category_to_index,
            "num_categories": int(len(categories)),
            "poi_frequency_exclude_home_work_other": bool(args.poi_frequency_exclude_home_work_other),
            "include_poi_other_in_length": bool(args.include_poi_other_in_length),
            "category_include_home_work_other": bool(args.category_include_home_work_other),
        }

    metrics_df1, stats_df1 = _eval_one_model(
        model_name=str(args.synthetic_name),
        syn_seqs=syn_seqs,
        syn_attrs=syn_attrs,
        syn_demo=syn_demo,
        **_eval_kw(),
    )

    metrics_frames = [metrics_df1]
    stats_frames = [stats_df1]

    if has2 and syn2_seqs is not None and syn2_attrs is not None and syn2_demo is not None:
        metrics_df2, stats_df2 = _eval_one_model(
            model_name=str(args.synthetic2_name),
            syn_seqs=syn2_seqs,
            syn_attrs=syn2_attrs,
            syn_demo=syn2_demo,
            **_eval_kw(),
        )
        metrics_frames.append(metrics_df2)
        stats_frames.append(stats_df2)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    stats_df = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()

    metrics_path = os.path.join(args.save_dir, "demo_group_metrics.csv")
    stats_path = os.path.join(args.save_dir, "demo_group_mapping_stats.csv")
    metrics_df.to_csv(metrics_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print("=== by-demo evaluation complete ===")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {stats_path}")
