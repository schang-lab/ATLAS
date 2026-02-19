from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .eval_by_demo_deps import (
    MappingConfig,
    SPECIAL_TOKENS_DEFAULT,
    UNK_TOKENS_DEFAULT,
    _grid_counts,
    _hist_1d,
    _hist_2d,
    _jsd,
    _wasserstein_2d,
    flatten_coords,
    origins_destinations,
)
from .eval_by_demo_io import CATEGORY_DROP_POI_TOKENS_DEFAULT
from .eval_by_demo_mapping import _trajectory_path_length_euclid_km, _trajectory_travel_distance_km


def _unique_coord_count(traj: np.ndarray) -> int:
    """Proxy for unique POIs: count unique (lat, lon) coordinate pairs in the mapped trajectory."""
    if traj is None or traj.ndim != 2 or traj.shape[0] != 2 or traj.shape[1] == 0:
        return 0
    pts = np.stack([traj[0], traj[1]], axis=1).astype(np.float64, copy=False)
    return int(np.unique(pts, axis=0).shape[0])


def _compute_poi_frequency_jsd(
    real_seqs: List[List[str]],
    syn_seqs: List[List[str]],
    *,
    cfg: MappingConfig,
    exclude_home_work_other: bool = False,
) -> float:
    """Compute JSD between POI frequency distributions of real and synthetic sequences."""
    special = cfg.special_tokens if cfg.special_tokens is not None else SPECIAL_TOKENS_DEFAULT
    unk = cfg.unk_tokens if cfg.unk_tokens is not None else UNK_TOKENS_DEFAULT
    exclude = set(special) | set(unk) | {cfg.poi_other_token}
    if exclude_home_work_other:
        exclude |= {cfg.poi_home_token, cfg.poi_work_token}

    real_counts: Dict[str, int] = {}
    for seq in real_seqs:
        for tok in seq:
            if tok not in exclude:
                real_counts[tok] = real_counts.get(tok, 0) + 1
    syn_counts: Dict[str, int] = {}
    for seq in syn_seqs:
        for tok in seq:
            if tok not in exclude:
                syn_counts[tok] = syn_counts.get(tok, 0) + 1

    all_pois = sorted(set(real_counts.keys()) | set(syn_counts.keys()))
    if len(all_pois) == 0:
        return float("nan")
    real_hist = np.array([real_counts.get(p, 0) for p in all_pois], dtype=np.float64)
    syn_hist = np.array([syn_counts.get(p, 0) for p in all_pois], dtype=np.float64)
    return _jsd(real_hist, syn_hist)


def _compute_poi_category_jsd(
    real_seqs: List[List[str]],
    syn_seqs: List[List[str]],
    *,
    cfg: MappingConfig,
    poi_to_category: Dict[str, str],
    category_to_index: Dict[str, int],
    num_categories: int,
    drop_poi_tokens: Tuple[str, ...] = CATEGORY_DROP_POI_TOKENS_DEFAULT,
    include_home_work_other: bool = False,
) -> float:
    """JSD between category-frequency distributions induced by raw POI token sequences."""
    C = int(num_categories)
    if C <= 0:
        return float("nan")
    special = cfg.special_tokens if cfg.special_tokens is not None else SPECIAL_TOKENS_DEFAULT
    unk = cfg.unk_tokens if cfg.unk_tokens is not None else UNK_TOKENS_DEFAULT
    if include_home_work_other:
        exclude = set(special) | set(unk)
    else:
        exclude = set(special) | set(unk) | set(drop_poi_tokens) | {cfg.poi_other_token}

    real = np.zeros((C,), dtype=np.float64)
    syn = np.zeros((C,), dtype=np.float64)
    for seq in real_seqs:
        for tok in seq:
            if tok in exclude:
                continue
            cat = poi_to_category.get(tok)
            if cat is None:
                continue
            j = category_to_index.get(cat)
            if j is not None:
                real[int(j)] += 1.0
    for seq in syn_seqs:
        for tok in seq:
            if tok in exclude:
                continue
            cat = poi_to_category.get(tok)
            if cat is None:
                continue
            j = category_to_index.get(cat)
            if j is not None:
                syn[int(j)] += 1.0
    if real.sum() <= 0.0 or syn.sum() <= 0.0:
        return float("nan")
    return _jsd(real, syn)


def _compute_poi_category_transition_jsd(
    real_seqs: List[List[str]],
    syn_seqs: List[List[str]],
    *,
    cfg: MappingConfig,
    poi_to_category: Dict[str, str],
    category_to_index: Dict[str, int],
    num_categories: int,
    drop_poi_tokens: Tuple[str, ...] = CATEGORY_DROP_POI_TOKENS_DEFAULT,
    include_home_work_other: bool = False,
) -> float:
    """JSD between category-bigram (adjacent-step) distributions induced by raw POI sequences."""
    C = int(num_categories)
    if C <= 0:
        return float("nan")
    K = C * C
    special = cfg.special_tokens if cfg.special_tokens is not None else SPECIAL_TOKENS_DEFAULT
    unk = cfg.unk_tokens if cfg.unk_tokens is not None else UNK_TOKENS_DEFAULT
    if include_home_work_other:
        exclude = set(special) | set(unk)
    else:
        exclude = set(special) | set(unk) | set(drop_poi_tokens) | {cfg.poi_other_token}

    def _accum(seqs: List[List[str]]) -> np.ndarray:
        out = np.zeros((K,), dtype=np.float64)
        for seq in seqs:
            if len(seq) < 2:
                continue
            prev = -1
            for tok in seq:
                if tok in exclude:
                    curr = -1
                else:
                    cat = poi_to_category.get(tok)
                    curr = int(category_to_index.get(cat, -1)) if cat is not None else -1
                if prev >= 0 and curr >= 0:
                    out[prev * C + curr] += 1.0
                prev = curr
        return out

    real = _accum(real_seqs)
    syn = _accum(syn_seqs)
    if real.sum() <= 0.0 or syn.sum() <= 0.0:
        return float("nan")
    return _jsd(real, syn)


def _compute_metrics_for_group(
    real_trajs: List[np.ndarray],
    syn_trajs: List[np.ndarray],
    *,
    histogram_bins: int,
    spatial_bins: int,
    grid_size: int,
    top_n: int,
    enable_wasserstein: bool,
    wasserstein_subsample: int,
    travel_distance_mode: str,
    real_seqs: Optional[List[List[str]]] = None,
    syn_seqs: Optional[List[List[str]]] = None,
    cfg: Optional[MappingConfig] = None,
    include_poi_other_in_length: bool = False,
) -> Dict[str, float]:
    real_lat, real_lon = flatten_coords(real_trajs)
    syn_lat, syn_lon = flatten_coords(syn_trajs)
    if real_lat.size == 0 or syn_lat.size == 0:
        return {}

    lat_min = float(min(real_lat.min(), syn_lat.min()))
    lat_max = float(max(real_lat.max(), syn_lat.max()))
    lon_min = float(min(real_lon.min(), syn_lon.min()))
    lon_max = float(max(real_lon.max(), syn_lon.max()))
    results: Dict[str, float] = {}

    real_sp = _hist_2d(real_lon, real_lat, spatial_bins, (lon_min, lon_max), (lat_min, lat_max))
    syn_sp = _hist_2d(syn_lon, syn_lat, spatial_bins, (lon_min, lon_max), (lat_min, lat_max))
    results["spatial_jsd"] = _jsd(real_sp, syn_sp)

    if str(travel_distance_mode).lower().strip() == "euclidean":
        real_td = np.asarray([_trajectory_path_length_euclid_km(t) for t in real_trajs], dtype=np.float64)
        syn_td = np.asarray([_trajectory_path_length_euclid_km(t) for t in syn_trajs], dtype=np.float64)
    else:
        real_td = np.asarray([_trajectory_travel_distance_km(t) for t in real_trajs], dtype=np.float64)
        syn_td = np.asarray([_trajectory_travel_distance_km(t) for t in syn_trajs], dtype=np.float64)
    if real_td.size > 0 and syn_td.size > 0:
        td_min = float(min(real_td.min(), syn_td.min()))
        td_max = float(max(real_td.max(), syn_td.max()))
        if abs(td_max - td_min) < 1e-12:
            td_max = td_min + 1e-6
        results["travel_distance_jsd"] = _jsd(
            _hist_1d(real_td, histogram_bins, td_min, td_max),
            _hist_1d(syn_td, histogram_bins, td_min, td_max),
        )
        results["real_travel_distance_mean"] = float(real_td.mean())
        results["synthetic_travel_distance_mean"] = float(syn_td.mean())

    if include_poi_other_in_length and real_seqs is not None and syn_seqs is not None and cfg is not None:
        special = cfg.special_tokens if cfg.special_tokens is not None else SPECIAL_TOKENS_DEFAULT
        unk = cfg.unk_tokens if cfg.unk_tokens is not None else UNK_TOKENS_DEFAULT
        exclude = set(special) | set(unk)

        def _count_valid_tokens(seq: List[str]) -> int:
            return sum(1 for tok in seq if tok not in exclude)

        real_lengths = np.asarray([_count_valid_tokens(seq) for seq in real_seqs], dtype=np.int64)
        syn_lengths = np.asarray([_count_valid_tokens(seq) for seq in syn_seqs], dtype=np.int64)
    else:
        real_lengths = np.asarray([t.shape[1] for t in real_trajs], dtype=np.int64)
        syn_lengths = np.asarray([t.shape[1] for t in syn_trajs], dtype=np.int64)

    if real_lengths.size > 0 and syn_lengths.size > 0:
        min_len = int(min(real_lengths.min(), syn_lengths.min()))
        max_len = int(max(real_lengths.max(), syn_lengths.max()))
        bins = max_len - min_len + 1
        real_counts = np.bincount(real_lengths - min_len, minlength=bins).astype(np.float64)
        syn_counts = np.bincount(syn_lengths - min_len, minlength=bins).astype(np.float64)
        results["length_jsd"] = _jsd(real_counts, syn_counts)
        results["real_length_mean"] = float(real_lengths.mean())
        results["synthetic_length_mean"] = float(syn_lengths.mean())

    real_u = np.asarray([_unique_coord_count(t) for t in real_trajs], dtype=np.int64)
    syn_u = np.asarray([_unique_coord_count(t) for t in syn_trajs], dtype=np.int64)
    if real_u.size > 0 and syn_u.size > 0:
        min_u = int(min(real_u.min(), syn_u.min()))
        max_u = int(max(real_u.max(), syn_u.max()))
        u_bins = max_u - min_u + 1
        real_u_counts = np.bincount(real_u - min_u, minlength=u_bins).astype(np.float64)
        syn_u_counts = np.bincount(syn_u - min_u, minlength=u_bins).astype(np.float64)
        results["unique_length_jsd"] = _jsd(real_u_counts, syn_u_counts)

    real_o, real_d = origins_destinations(real_trajs)
    syn_o, syn_d = origins_destinations(syn_trajs)
    o_lon = np.concatenate([real_o[:, 1], syn_o[:, 1]])
    o_lat = np.concatenate([real_o[:, 0], syn_o[:, 0]])
    d_lon = np.concatenate([real_d[:, 1], syn_d[:, 1]])
    d_lat = np.concatenate([real_d[:, 0], syn_d[:, 0]])
    lon_range = (float(min(o_lon.min(), d_lon.min())), float(max(o_lon.max(), d_lon.max())))
    lat_range = (float(min(o_lat.min(), d_lat.min())), float(max(o_lat.max(), d_lat.max())))
    real_o_hist = _grid_counts(real_o[:, 1], real_o[:, 0], grid_size, lon_range, lat_range)
    syn_o_hist = _grid_counts(syn_o[:, 1], syn_o[:, 0], grid_size, lon_range, lat_range)
    real_d_hist = _grid_counts(real_d[:, 1], real_d[:, 0], grid_size, lon_range, lat_range)
    syn_d_hist = _grid_counts(syn_d[:, 1], syn_d[:, 0], grid_size, lon_range, lat_range)
    results["origin_jsd"] = _jsd(real_o_hist, syn_o_hist)
    results["destination_jsd"] = _jsd(real_d_hist, syn_d_hist)
    results["trip_jsd"] = float((results["origin_jsd"] + results["destination_jsd"]) / 2.0)

    real_cell_counts = _grid_counts(real_lon, real_lat, grid_size, (lon_min, lon_max), (lat_min, lat_max))
    syn_cell_counts = _grid_counts(syn_lon, syn_lat, grid_size, (lon_min, lon_max), (lat_min, lat_max))
    real_top = set(np.argsort(real_cell_counts)[::-1][:top_n].tolist())
    syn_top = set(np.argsort(syn_cell_counts)[::-1][:top_n].tolist())
    inter = real_top & syn_top
    precision = len(inter) / max(len(syn_top), 1)
    recall = len(inter) / max(len(real_top), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    results["pattern_score"] = float(f1)
    results["pattern_precision"] = float(precision)
    results["pattern_recall"] = float(recall)

    if enable_wasserstein:
        from scipy.stats import wasserstein_distance

        real_all = np.concatenate([real_lat, real_lon])
        syn_all = np.concatenate([syn_lat, syn_lon])
        results["overall_wasserstein"] = float(wasserstein_distance(real_all, syn_all))
        results["latitude_wasserstein"] = float(wasserstein_distance(real_lat, syn_lat))
        results["longitude_wasserstein"] = float(wasserstein_distance(real_lon, syn_lon))
        results["spatial_wasserstein_2d"] = _wasserstein_2d(
            np.stack([real_lat, real_lon], axis=1),
            np.stack([syn_lat, syn_lon], axis=1),
            subsample=int(wasserstein_subsample),
        )
        results["origin_wasserstein_2d"] = _wasserstein_2d(
            real_o[:, [0, 1]], syn_o[:, [0, 1]], subsample=int(wasserstein_subsample)
        )
        results["destination_wasserstein_2d"] = _wasserstein_2d(
            real_d[:, [0, 1]], syn_d[:, [0, 1]], subsample=int(wasserstein_subsample)
        )
        if real_td.size > 0 and syn_td.size > 0:
            results["length_wasserstein"] = float(wasserstein_distance(real_td, syn_td))

    return results
