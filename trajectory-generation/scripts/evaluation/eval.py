#!/usr/bin/env python3
"""
Evaluation of DiT-only models.

Inputs:
  - real_poi_pkl: split_data/.../final_segments_all_train_data.pkl (DataFrame with unique_id_seq)
  - real_attr_npy: all_attr_results.npy (aligned by row with the pickle DataFrame) [optional but recommended]
  - synthetic_poi_pkl: generated_poi_sequences.pkl (from inference)
  - synthetic_attr_npy: sampled_attributes.npy (from inference) [optional; used for POI_HOME/WORK mapping]
  - poi_map_csv: poi_map_feature.csv (for mapping POIs to coordinates)

Metrics
---------------------------------------------------------------------------
- spatial_jsd
- travel_distance_jsd (Haversine distance in km, computed on mapped coordinate trajectories)
- length_jsd (number of mapped coordinate points per trajectory)
- unique_length_jsd (unique coordinate pairs per trajectory; proxy for unique POIs)
- origin_jsd / destination_jsd / trip_jsd
- pattern_score
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

try:
    import ot  # type: ignore

    HAS_OT = True
except ImportError:
    HAS_OT = False


SPECIAL_TOKENS_DEFAULT = {
    "[PAD]", "[CLS]", "[SEP]", "[MASK]",
}
UNK_TOKENS_DEFAULT = {"[UNK]"}


def _as_list(seq) -> List[str]:
    if isinstance(seq, str):
        return seq.strip().split()
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [str(x) for x in seq]
    return [str(seq)]


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


def _hist_1d(values: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    hist = hist.astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def _hist_2d(lon: np.ndarray, lat: np.ndarray, bins: int,
            lon_range: Tuple[float, float], lat_range: Tuple[float, float]) -> np.ndarray:
    h, _, _ = np.histogram2d(
        lon, lat,
        bins=bins,
        range=[lon_range, lat_range],
        density=False,
    )
    h = h.astype(np.float64).reshape(-1)
    return h / max(h.sum(), 1.0)


def _grid_counts(lon: np.ndarray, lat: np.ndarray, grid: int,
                 lon_range: Tuple[float, float], lat_range: Tuple[float, float]) -> np.ndarray:
    # return raw counts per cell (flattened)
    h, _, _ = np.histogram2d(
        lon, lat,
        bins=grid,
        range=[lon_range, lat_range],
        density=False,
    )
    return h.astype(np.float64).reshape(-1)


def _wasserstein_2d(u: np.ndarray, v: np.ndarray, subsample: int = 10000) -> float:
    if u.shape[0] == 0 or v.shape[0] == 0:
        return float("nan")
    if u.shape[0] > subsample:
        u = u[np.random.choice(u.shape[0], subsample, replace=False)]
    if v.shape[0] > subsample:
        v = v[np.random.choice(v.shape[0], subsample, replace=False)]

    if HAS_OT:
        a = np.ones(u.shape[0], dtype=np.float64) / u.shape[0]
        b = np.ones(v.shape[0], dtype=np.float64) / v.shape[0]
        M = ot.dist(u, v, metric="euclidean")  # type: ignore
        return float(ot.emd2(a, b, M))  # type: ignore

    # fallback: PCA-like projection approximation
    all_coords = np.vstack([u, v])
    mean = all_coords.mean(axis=0, keepdims=True)
    u0 = u - mean
    v0 = v - mean
    cov = np.cov(all_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u1 = u0 @ eigvecs[:, 0]
    v1 = v0 @ eigvecs[:, 0]
    u2 = u0 @ eigvecs[:, 1]
    v2 = v0 @ eigvecs[:, 1]
    w1 = wasserstein_distance(u1, v1)
    w2 = wasserstein_distance(u2, v2)
    weights = eigvals[order[:2]]
    weights = weights / max(weights.sum(), 1e-12)
    return float(weights[0] * w1 + weights[1] * w2)


@dataclass
class MappingConfig:
    poi_other_token: str = "POI_OTHER"
    poi_home_token: str = "POI_HOME"
    poi_work_token: str = "POI_WORK"
    special_tokens: Optional[set] = None
    unk_tokens: Optional[set] = None
    min_mapped_pois: int = 2


def load_poi_coords(poi_map_csv: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(poi_map_csv)
    mapping: Dict[str, Tuple[float, float]] = {}
    for row in df.itertuples():
        mapping[str(row.poi_id)] = (float(row.lat), float(row.lon))
    return mapping


def map_sequences_to_coords(
    sequences: List[List[str]],
    poi_coords: Dict[str, Tuple[float, float]],
    attrs_4d: Optional[np.ndarray],
    cfg: MappingConfig,
    label: str,
    *,
    drop_home_work_points: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Convert token sequences into variable-length coordinate trajectories.
    Returns list of (2, L) arrays and stats.
    """
    special = cfg.special_tokens if cfg.special_tokens is not None else SPECIAL_TOKENS_DEFAULT
    unk = cfg.unk_tokens if cfg.unk_tokens is not None else UNK_TOKENS_DEFAULT

    total_tokens = 0
    dropped_special = 0
    dropped_other = 0
    dropped_unk = 0
    injected_home = 0
    injected_work = 0
    mapped_poi = 0
    missing_map = 0
    dropped_short = 0
    dropped_home_work = 0

    coords_out: List[np.ndarray] = []

    if attrs_4d is not None and attrs_4d.shape[0] != len(sequences):
        raise ValueError(f"{label}: attrs rows != sequences count ({attrs_4d.shape[0]} vs {len(sequences)})")

    for i, seq in enumerate(sequences):
        pts: List[Tuple[float, float]] = []
        if attrs_4d is not None:
            work_lat, work_lon, home_lat, home_lon = attrs_4d[i, 0:4].tolist()
        else:
            work_lat = work_lon = home_lat = home_lon = None

        for tok in seq:
            total_tokens += 1
            if tok in special:
                dropped_special += 1
                continue
            if tok == cfg.poi_other_token:
                dropped_other += 1
                continue
            if tok in unk:
                # keep UNK as token for possible downstream sequence-based stats, but it has no coords
                dropped_unk += 1
                continue
            if tok == cfg.poi_home_token:
                if drop_home_work_points:
                    dropped_home_work += 1
                    continue
                if home_lat is not None and home_lon is not None:
                    pts.append((float(home_lat), float(home_lon)))
                    injected_home += 1
                else:
                    missing_map += 1
                continue
            if tok == cfg.poi_work_token:
                if drop_home_work_points:
                    dropped_home_work += 1
                    continue
                if work_lat is not None and work_lon is not None:
                    pts.append((float(work_lat), float(work_lon)))
                    injected_work += 1
                else:
                    missing_map += 1
                continue

            xy = poi_coords.get(tok)
            if xy is None:
                missing_map += 1
                continue
            pts.append((float(xy[0]), float(xy[1])))
            mapped_poi += 1

        if len(pts) < cfg.min_mapped_pois:
            dropped_short += 1
            continue
        lat = np.asarray([p[0] for p in pts], dtype=np.float32)
        lon = np.asarray([p[1] for p in pts], dtype=np.float32)
        coords_out.append(np.stack([lat, lon], axis=0))

    stats = {
        "total_sequences": float(len(sequences)),
        "kept_sequences": float(len(coords_out)),
        "dropped_sequences_too_short": float(dropped_short),
        "total_tokens": float(total_tokens),
        "dropped_special_tokens": float(dropped_special),
        "dropped_poi_other": float(dropped_other),
        "dropped_unk_no_coords": float(dropped_unk),
        "dropped_home_work_tokens": float(dropped_home_work),
        "injected_home": float(injected_home),
        "injected_work": float(injected_work),
        "mapped_poi_tokens": float(mapped_poi),
        "missing_coord_tokens": float(missing_map),
    }
    return coords_out, stats


def flatten_coords(trajs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not trajs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    lats = np.concatenate([t[0, :] for t in trajs], axis=0).astype(np.float64, copy=False)
    lons = np.concatenate([t[1, :] for t in trajs], axis=0).astype(np.float64, copy=False)
    return lats, lons


def origins_destinations(trajs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not trajs:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    origins = np.stack([[t[0, 0], t[1, 0]] for t in trajs], axis=0).astype(np.float64)
    dests = np.stack([[t[0, -1], t[1, -1]] for t in trajs], axis=0).astype(np.float64)
    return origins, dests


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in km; inputs can be scalars or numpy arrays."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _trajectory_travel_distance_km(traj: np.ndarray) -> float:
    """Sum of consecutive haversine distances (km) for a mapped trajectory (2, L)."""
    if traj is None or traj.ndim != 2 or traj.shape[0] != 2 or traj.shape[1] < 2:
        return 0.0
    lat = traj[0].astype(np.float64, copy=False)
    lon = traj[1].astype(np.float64, copy=False)
    return float(np.sum(_haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])))


def _unique_coord_count(traj: np.ndarray) -> int:
    """Proxy for unique POIs: count unique (lat, lon) coordinate pairs in the mapped trajectory."""
    if traj is None or traj.ndim != 2 or traj.shape[0] != 2 or traj.shape[1] == 0:
        return 0
    pts = np.stack([traj[0], traj[1]], axis=1).astype(np.float64, copy=False)  # (L, 2)
    return int(np.unique(pts, axis=0).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DiT-only outputs using POI sequences")
    parser.add_argument("--real_poi_pkl", type=str, required=True,
                        help="Path to real split pickle (final_segments_all_train_data.pkl)")
    parser.add_argument("--real_attr_npy", type=str, default=None,
                        help="Optional: all_attr_results.npy aligned with real_poi_pkl (work_lat, work_lon, home_lat, home_lon)")
    parser.add_argument("--synthetic_poi_pkl", type=str, required=True,
                        help="Path to generated_poi_sequences.pkl")
    parser.add_argument("--synthetic_attr_npy", type=str, default=None,
                        help="Optional: sampled_attributes.npy aligned with synthetic_poi_pkl (work_lat, work_lon, home_lat, home_lon)")
    parser.add_argument("--poi_map_csv", type=str, required=True,
                        help="Path to poi_map_feature.csv for POI->(lat,lon) mapping")

    parser.add_argument("--poi_other_token", type=str, default="POI_OTHER")
    parser.add_argument("--poi_home_token", type=str, default="POI_HOME")
    parser.add_argument("--poi_work_token", type=str, default="POI_WORK")
    parser.add_argument("--min_mapped_pois", type=int, default=2)
    parser.add_argument(
        "--drop_home_work_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, drop POI_HOME/POI_WORK points from trajectories before metric computation.",
    )

    parser.add_argument("--save_dir", type=str, default="evaluation")

    # Metric params (keep compatible knobs, but user should tune for runtime/memory)
    parser.add_argument("--histogram_bins", type=int, default=10000)
    parser.add_argument("--spatial_bins", type=int, default=10000)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--enable_wasserstein", action="store_true", default=False)
    parser.add_argument("--wasserstein_subsample", type=int, default=10000)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load sequences
    real_df = pd.read_pickle(args.real_poi_pkl)
    real_sequences = [_as_list(s) for s in real_df["unique_id_seq"].tolist()]

    with open(args.synthetic_poi_pkl, "rb") as f:
        synthetic_sequences = pickle.load(f)
    synthetic_sequences = [_as_list(s) for s in synthetic_sequences]

    # Load attrs (optional)
    real_attrs = None
    if args.real_attr_npy is not None and os.path.exists(args.real_attr_npy):
        real_attrs = np.load(args.real_attr_npy, allow_pickle=True).astype(np.float32)
        if real_attrs.shape[1] < 4:
            raise ValueError("real_attr_npy must have at least 4 columns: work_lat, work_lon, home_lat, home_lon")
        real_attrs = real_attrs[:, :4]

    synthetic_attrs = None
    if args.synthetic_attr_npy is not None and os.path.exists(args.synthetic_attr_npy):
        synthetic_attrs = np.load(args.synthetic_attr_npy, allow_pickle=True).astype(np.float32)
        if synthetic_attrs.shape[1] < 4:
            raise ValueError("synthetic_attr_npy must have at least 4 columns: work_lat, work_lon, home_lat, home_lon")
        synthetic_attrs = synthetic_attrs[:, :4]

    cfg = MappingConfig(
        poi_other_token=args.poi_other_token,
        poi_home_token=args.poi_home_token,
        poi_work_token=args.poi_work_token,
        min_mapped_pois=int(args.min_mapped_pois),
        special_tokens=SPECIAL_TOKENS_DEFAULT,
        unk_tokens=UNK_TOKENS_DEFAULT,
    )

    poi_coords = load_poi_coords(args.poi_map_csv)

    # Map both sides using the same rules
    real_trajs, real_stats = map_sequences_to_coords(
        sequences=real_sequences,
        poi_coords=poi_coords,
        attrs_4d=real_attrs,
        cfg=cfg,
        label="real",
        drop_home_work_points=bool(args.drop_home_work_points),
    )
    syn_trajs, syn_stats = map_sequences_to_coords(
        sequences=synthetic_sequences,
        poi_coords=poi_coords,
        attrs_4d=synthetic_attrs,
        cfg=cfg,
        label="synthetic",
        drop_home_work_points=bool(args.drop_home_work_points),
    )

    # Dump stats
    stats_out = {
        "real": real_stats,
        "synthetic": syn_stats,
        "config": {
            "poi_other_token": args.poi_other_token,
            "poi_home_token": args.poi_home_token,
            "poi_work_token": args.poi_work_token,
            "min_mapped_pois": int(args.min_mapped_pois),
        },
    }
    with open(os.path.join(args.save_dir, "mapping_stats.json"), "w") as f:
        import json

        json.dump(stats_out, f, indent=2)

    # Build flattened coord sets
    real_lat, real_lon = flatten_coords(real_trajs)
    syn_lat, syn_lon = flatten_coords(syn_trajs)
    if real_lat.size == 0 or syn_lat.size == 0:
        raise RuntimeError("No valid mapped coordinates on one side; check filtering/mapping and attrs inputs.")

    # Common ranges
    lat_min = float(min(real_lat.min(), syn_lat.min()))
    lat_max = float(max(real_lat.max(), syn_lat.max()))
    lon_min = float(min(real_lon.min(), syn_lon.min()))
    lon_max = float(max(real_lon.max(), syn_lon.max()))

    # Metrics
    results: Dict[str, float] = {}

    # Spatial JSD
    real_sp = _hist_2d(real_lon, real_lat, args.spatial_bins, (lon_min, lon_max), (lat_min, lat_max))
    syn_sp = _hist_2d(syn_lon, syn_lat, args.spatial_bins, (lon_min, lon_max), (lat_min, lat_max))
    results["spatial_jsd"] = _jsd(real_sp, syn_sp)

    # Travel distance JSD (km, haversine on mapped coords)
    real_td = np.asarray([_trajectory_travel_distance_km(t) for t in real_trajs], dtype=np.float64)
    syn_td = np.asarray([_trajectory_travel_distance_km(t) for t in syn_trajs], dtype=np.float64)
    if real_td.size > 0 and syn_td.size > 0:
        td_min = float(min(real_td.min(), syn_td.min()))
        td_max = float(max(real_td.max(), syn_td.max()))
        if abs(td_max - td_min) < 1e-12:
            td_max = td_min + 1e-6
        results["travel_distance_jsd"] = _jsd(
            _hist_1d(real_td, args.histogram_bins, td_min, td_max),
            _hist_1d(syn_td, args.histogram_bins, td_min, td_max),
        )
        results["real_travel_distance_mean"] = float(real_td.mean())
        results["synthetic_travel_distance_mean"] = float(syn_td.mean())

    # Length JSD in points
    real_lengths = np.asarray([t.shape[1] for t in real_trajs], dtype=np.int64)
    syn_lengths = np.asarray([t.shape[1] for t in syn_trajs], dtype=np.int64)
    min_len = int(min(real_lengths.min(), syn_lengths.min()))
    max_len = int(max(real_lengths.max(), syn_lengths.max()))
    bins = max_len - min_len + 1
    real_counts = np.bincount(real_lengths - min_len, minlength=bins).astype(np.float64)
    syn_counts = np.bincount(syn_lengths - min_len, minlength=bins).astype(np.float64)
    results["length_jsd"] = _jsd(real_counts, syn_counts)
    results["real_length_mean"] = float(real_lengths.mean())
    results["synthetic_length_mean"] = float(syn_lengths.mean())

    # Unique-length JSD (proxy): unique coordinate pairs per trajectory
    real_u = np.asarray([_unique_coord_count(t) for t in real_trajs], dtype=np.int64)
    syn_u = np.asarray([_unique_coord_count(t) for t in syn_trajs], dtype=np.int64)
    if real_u.size > 0 and syn_u.size > 0:
        min_u = int(min(real_u.min(), syn_u.min()))
        max_u = int(max(real_u.max(), syn_u.max()))
        u_bins = max_u - min_u + 1
        real_u_counts = np.bincount(real_u - min_u, minlength=u_bins).astype(np.float64)
        syn_u_counts = np.bincount(syn_u - min_u, minlength=u_bins).astype(np.float64)
        results["unique_length_jsd"] = _jsd(real_u_counts, syn_u_counts)

    # Trip (origin/destination) JSD on grid
    real_o, real_d = origins_destinations(real_trajs)
    syn_o, syn_d = origins_destinations(syn_trajs)
    # common bounds from all points
    o_lon = np.concatenate([real_o[:, 1], syn_o[:, 1]])
    o_lat = np.concatenate([real_o[:, 0], syn_o[:, 0]])
    d_lon = np.concatenate([real_d[:, 1], syn_d[:, 1]])
    d_lat = np.concatenate([real_d[:, 0], syn_d[:, 0]])
    lon_range = (float(min(o_lon.min(), d_lon.min())), float(max(o_lon.max(), d_lon.max())))
    lat_range = (float(min(o_lat.min(), d_lat.min())), float(max(o_lat.max(), d_lat.max())))
    real_o_hist = _grid_counts(real_o[:, 1], real_o[:, 0], args.grid_size, lon_range, lat_range)
    syn_o_hist = _grid_counts(syn_o[:, 1], syn_o[:, 0], args.grid_size, lon_range, lat_range)
    real_d_hist = _grid_counts(real_d[:, 1], real_d[:, 0], args.grid_size, lon_range, lat_range)
    syn_d_hist = _grid_counts(syn_d[:, 1], syn_d[:, 0], args.grid_size, lon_range, lat_range)
    results["origin_jsd"] = _jsd(real_o_hist, syn_o_hist)
    results["destination_jsd"] = _jsd(real_d_hist, syn_d_hist)
    results["trip_jsd"] = float((results["origin_jsd"] + results["destination_jsd"]) / 2.0)

    # Pattern score: top-n grid cells of all points
    real_cell_counts = _grid_counts(real_lon, real_lat, args.grid_size, (lon_min, lon_max), (lat_min, lat_max))
    syn_cell_counts = _grid_counts(syn_lon, syn_lat, args.grid_size, (lon_min, lon_max), (lat_min, lat_max))
    real_top = set(np.argsort(real_cell_counts)[::-1][: args.top_n].tolist())
    syn_top = set(np.argsort(syn_cell_counts)[::-1][: args.top_n].tolist())
    inter = real_top & syn_top
    precision = len(inter) / max(len(syn_top), 1)
    recall = len(inter) / max(len(real_top), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    results["pattern_score"] = float(f1)
    results["pattern_precision"] = float(precision)
    results["pattern_recall"] = float(recall)

    # Wasserstein (optional)
    if args.enable_wasserstein:
        real_all = np.concatenate([real_lat, real_lon])
        syn_all = np.concatenate([syn_lat, syn_lon])
        results["overall_wasserstein"] = float(wasserstein_distance(real_all, syn_all))
        results["latitude_wasserstein"] = float(wasserstein_distance(real_lat, syn_lat))
        results["longitude_wasserstein"] = float(wasserstein_distance(real_lon, syn_lon))
        results["spatial_wasserstein_2d"] = _wasserstein_2d(
            np.stack([real_lat, real_lon], axis=1),
            np.stack([syn_lat, syn_lon], axis=1),
            subsample=int(args.wasserstein_subsample),
        )
        results["origin_wasserstein_2d"] = _wasserstein_2d(real_o[:, [0, 1]], syn_o[:, [0, 1]], subsample=int(args.wasserstein_subsample))
        results["destination_wasserstein_2d"] = _wasserstein_2d(real_d[:, [0, 1]], syn_d[:, [0, 1]], subsample=int(args.wasserstein_subsample))
        if real_td.size > 0 and syn_td.size > 0:
            # Match Cardiff naming: travel-distance Wasserstein reported as "length_wasserstein"
            results["length_wasserstein"] = float(wasserstein_distance(real_td, syn_td))

    # Save results
    results_df = pd.DataFrame({"Metric": list(results.keys()), "Value": list(results.values())})
    results_df.to_csv(os.path.join(args.save_dir, "evaluation_results.csv"), index=False)

    print("=== evaluation complete ===")
    print(results_df.sort_values("Metric").to_string(index=False))
    print(f"Saved results to: {args.save_dir}")


if __name__ == "__main__":
    import pickle

    main()


