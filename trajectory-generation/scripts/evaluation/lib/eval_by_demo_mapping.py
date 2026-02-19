from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .eval_by_demo_deps import MappingConfig, SPECIAL_TOKENS_DEFAULT, UNK_TOKENS_DEFAULT


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


def _trajectory_path_length_euclid_km(traj_xy_km: np.ndarray) -> float:
    """
    Sum of consecutive Euclidean distances (km) for a trajectory in a local Cartesian frame (2, L).
    """
    if traj_xy_km is None or traj_xy_km.ndim != 2 or traj_xy_km.shape[0] != 2 or traj_xy_km.shape[1] < 2:
        return 0.0
    x = traj_xy_km[0].astype(np.float64, copy=False)
    y = traj_xy_km[1].astype(np.float64, copy=False)
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def _latlon_to_local_xy_km(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate lat/lon -> local tangent-plane (x=East, y=North) in km using equirectangular projection."""
    R = 6371.0
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0r = np.deg2rad(float(lat0))
    lon0r = np.deg2rad(float(lon0))
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    x = (lonr - lon0r) * np.cos(lat0r) * R
    y = (latr - lat0r) * R
    return x, y


def _normalize_traj_home_work_frame(
    traj_latlon: np.ndarray,
    *,
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    scale_by_commute: bool,
) -> np.ndarray:
    """
    Transform a lat/lon trajectory (2, L) into a home-centered, work-aligned local frame (2, L).
    """
    if traj_latlon is None or traj_latlon.ndim != 2 or traj_latlon.shape[0] != 2:
        return traj_latlon
    lat = traj_latlon[0]
    lon = traj_latlon[1]
    x, y = _latlon_to_local_xy_km(lat, lon, lat0=home_lat, lon0=home_lon)
    wx, wy = _latlon_to_local_xy_km(np.asarray([work_lat]), np.asarray([work_lon]), lat0=home_lat, lon0=home_lon)
    vx, vy = float(wx[0]), float(wy[0])
    theta = float(np.arctan2(vy, vx)) if (abs(vx) + abs(vy)) > 1e-12 else 0.0
    c = float(np.cos(-theta))
    s = float(np.sin(-theta))
    xr = c * x - s * y
    yr = s * x + c * y
    if scale_by_commute:
        denom = float(np.sqrt(vx * vx + vy * vy))
        if denom > 1e-9:
            xr = xr / denom
            yr = yr / denom
    return np.stack([xr.astype(np.float32), yr.astype(np.float32)], axis=0)


def _map_sequences_to_coords_carlos(
    *,
    sequences: List[List[str]],
    poi_coords: Dict[str, Tuple[float, float]],
    attrs_4d: Optional[np.ndarray],
    cfg: MappingConfig,
    label: str,
    drop_home_work_points: bool,
) -> Tuple[List[np.ndarray], Dict[str, float], List[Tuple[float, float, float, float]], List[List[str]]]:
    """
    Wrapper around the mapping logic with drop_home_work_points option.
    Returns: trajs, stats, kept_hw, kept_seqs
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
    kept_hw: List[Tuple[float, float, float, float]] = []
    kept_seqs: List[List[str]] = []

    if attrs_4d is not None:
        if attrs_4d.ndim != 2 or attrs_4d.shape[1] < 4:
            raise ValueError(f"{label}: attrs_4d must be [N,>=4], got {attrs_4d.shape}")
        if attrs_4d.shape[0] != len(sequences):
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
        kept_seqs.append(seq)
        if home_lat is None or home_lon is None or work_lat is None or work_lon is None:
            kept_hw.append((float("nan"), float("nan"), float("nan"), float("nan")))
        else:
            kept_hw.append((float(home_lat), float(home_lon), float(work_lat), float(work_lon)))

    stats = {
        "total_sequences": float(len(sequences)),
        "kept_sequences": float(len(coords_out)),
        "dropped_sequences_too_short": float(dropped_short),
        "total_tokens": float(total_tokens),
        "dropped_special_tokens": float(dropped_special),
        "dropped_poi_other": float(dropped_other),
        "dropped_unk_no_coords": float(dropped_unk),
        "injected_home": float(injected_home),
        "injected_work": float(injected_work),
        "mapped_poi_tokens": float(mapped_poi),
        "missing_coord_tokens": float(missing_map),
        "dropped_home_work_tokens": float(dropped_home_work),
    }
    return coords_out, stats, kept_hw, kept_seqs
