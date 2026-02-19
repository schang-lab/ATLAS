#!/usr/bin/env python3
"""
Convert carlos_data/{traj.csv,demo.csv} into the split-data format used by this repo.

Key points:
- One training sample == one segment_id
- Sequence tokens == poi_raw_id, ordered by (arrival_ts, departure_ts)
- Writes controlled/{train,val,test} folders aligned with traj.csv 'split' column
- Builds poi_map_feature.csv from the most frequent (lat,lon,poi_id_model,location_name) per poi_raw_id
- Builds attrs arrays from demo.csv (home/work coords + optional 5-bin age + gender id)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def age_to_bin5(age: Optional[float]) -> int:
    """Map numeric age to 5 bins: <30, 30-40, 40-50, 50+."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return -1
    try:
        a = float(age)
    except Exception:
        return -1
    if a < 30:
        return 0
    if a < 40:
        return 1
    if a < 50:
        return 2
    return 3


def gender_to_id(gender: Optional[str]) -> int:
    if gender is None or (isinstance(gender, float) and np.isnan(gender)):
        return -1
    g = str(gender).strip().upper()
    if g == "FEMALE":
        return 0
    if g == "MALE":
        return 1
    return -1


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _build_tokenizer(vocab_tokens: List[str], output_tokenizer_dir: Path) -> None:
    """
    Build and save a BertTokenizerFast tokenizer directory.

    Configure tokenizer special tokens in **BERT-style**:
      cls_token=[CLS], sep_token=[SEP], pad_token=[PAD], unk_token=[UNK], mask_token=[MASK]
    """
    from transformers import BertTokenizerFast

    _ensure_dir(output_tokenizer_dir)
    vocab_path = output_tokenizer_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok in vocab_tokens:
            f.write(f"{tok}\n")

    tok = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tok.add_special_tokens(
        {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]",
        }
    )
    tok.save_pretrained(str(output_tokenizer_dir))


def _apply_special_tokens_and_padding(
    seq: List[str],
    max_len: int,
    add_special_tokens: bool,
    pad_to_max_len: bool,
) -> Tuple[List[str], List[int]]:
    """
    Add [CLS]/[SEP] and optionally [PAD] to reach max_len.
    Returns (new_seq, attention_mask).
    """
    seq = [str(s) for s in seq]
    if max_len <= 0:
        return [], []

    if add_special_tokens and max_len >= 2:
        seq = seq[: max_len - 2]
        seq = ["[CLS]"] + seq + ["[SEP]"]
    else:
        seq = seq[:max_len]

    if pad_to_max_len and len(seq) < max_len:
        seq = seq + ["[PAD]"] * (max_len - len(seq))

    attention_mask = [0 if tok == "[PAD]" else 1 for tok in seq]
    return seq, attention_mask


def _load_demo(demo_csv: Path) -> pd.DataFrame:
    demo = pd.read_csv(demo_csv)
    # Standardize types
    demo["panelist_id"] = demo["panelist_id"].astype(str)
    for c in ["home_lat", "home_lon", "work_lat", "work_lon"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    if "age" in demo.columns:
        demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    return demo


def _duckdb_connect(memory_limit: str, threads: int, temp_dir: Optional[str]) -> "duckdb.DuckDBPyConnection":
    import tempfile
    import duckdb  # type: ignore

    con = duckdb.connect(database=":memory:")
    # Avoid OOM on limited RAM: use a low limit so DuckDB can spill to disk instead of crashing.
    # When memory_limit is exceeded, DuckDB spills to temp_directory (prevents recursive terminate).
    con.execute(f"SET memory_limit={_sql_quote(str(memory_limit))}")
    con.execute(f"SET threads={int(threads)}")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        tmp_dir = temp_dir
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="duckdb_")
    con.execute(f"SET temp_directory={_sql_quote(str(tmp_dir))}")
    return con

def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_poi_map_feature(con, has_category: bool = False) -> pd.DataFrame:
    """
    Build a POI metadata table with the most frequent coordinate+label row per poi_raw_id.
    """
    if has_category:
        df = con.execute(
            """
            WITH counts AS (
              SELECT
                poi_raw_id,
                latitude,
                longitude,
                COALESCE(category, poi_id_model) AS top_category,
                location_name,
                COUNT(*) AS cnt
              FROM traj
              WHERE poi_raw_id IS NOT NULL
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
              GROUP BY 1,2,3,4,5
            ),
            ranked AS (
              SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY poi_raw_id ORDER BY cnt DESC) AS rn
              FROM counts
            )
            SELECT
              poi_raw_id AS poi_id,
              latitude AS lat,
              longitude AS lon,
              top_category AS top_category,
              COALESCE(location_name, 'unknown') AS sub_category,
              NULL::VARCHAR AS placekey
            FROM ranked
            WHERE rn = 1
            ORDER BY poi_id
            """
        ).fetch_df()
    else:
        df = con.execute(
            """
            WITH counts AS (
              SELECT
                poi_raw_id,
                latitude,
                longitude,
                poi_id_model AS top_category,
                location_name,
                COUNT(*) AS cnt
              FROM traj
              WHERE poi_raw_id IS NOT NULL
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
              GROUP BY 1,2,3,4,5
            ),
            ranked AS (
              SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY poi_raw_id ORDER BY cnt DESC) AS rn
              FROM counts
            )
            SELECT
              poi_raw_id AS poi_id,
              latitude AS lat,
              longitude AS lon,
              top_category AS top_category,
              COALESCE(location_name, 'unknown') AS sub_category,
              NULL::VARCHAR AS placekey
            FROM ranked
            WHERE rn = 1
            ORDER BY poi_id
            """
        ).fetch_df()
    return df


def build_sequences_for_split(
    con,
    split_name: str,
    token_sql_expr: str,
    max_len: int,
    bucket_count: int = 1,
    bucket_idx: int = 0,
) -> pd.DataFrame:
    """
    Returns one row per segment_id with list columns:
      unique_id_seq, attention_mask, arrival_ts_seq, dwell_min_seq
    """
    bucket_clause = ""
    params: List[object] = [split_name]
    if bucket_count > 1:
        bucket_clause = "AND (abs(hash(segment_id)) % ?) = ?"
        params.extend([bucket_count, bucket_idx])
    params.extend([max_len, max_len])

    df = con.execute(
        f"""
        WITH ordered AS (
          SELECT
            segment_id,
            panelist_id,
            city,
            split,
            {token_sql_expr} AS token,
            arrival_ts,
            departure_ts,
            duration_minutes,
            ROW_NUMBER() OVER (
              PARTITION BY segment_id
              ORDER BY arrival_ts, departure_ts
            ) AS rn
          FROM traj
          WHERE split = ?
            AND {token_sql_expr} IS NOT NULL
            {bucket_clause}
        ),
        limited AS (
          SELECT *
          FROM ordered
          WHERE rn <= ?
        )
        SELECT
          segment_id,
          panelist_id,
          any_value(city) AS city,
          split,
          list(token ORDER BY rn) AS unique_id_seq,
          list(1 ORDER BY rn) AS attention_mask,
          list(arrival_ts ORDER BY rn) AS arrival_ts_seq,
          list(duration_minutes ORDER BY rn) AS dwell_min_seq,
          LEAST(MAX(rn), ?) AS length_id
        FROM limited
        GROUP BY segment_id, panelist_id, split
        """
        ,
        params,
    ).fetch_df()
    return df


def _find_bg_path(cbg_data_dir: Path, year: int, state_fips: str) -> Tuple[str, str]:
    """
    Return (source_type, path) for a BG shapefile:
      - source_type in {"shp", "zip"}
      - path is either a direct .shp path (for source_type="shp")
        or a .zip path (for source_type="zip") to be used with "zip://"
    """
    return _find_tiger_path(census_data_dir=cbg_data_dir, year=year, state_fips=state_fips, layer="bg")


def _find_tract_path(census_data_dir: Path, year: int, state_fips: str) -> Tuple[str, str]:
    """
    Return (source_type, path) for a tract shapefile (TIGER/Line).
    """
    return _find_tiger_path(census_data_dir=census_data_dir, year=year, state_fips=state_fips, layer="tract")


def _find_tiger_path(census_data_dir: Path, year: int, state_fips: str, layer: str) -> Tuple[str, str]:
    """
    Return (source_type, path) for a TIGER/Line shapefile layer:
      - layer in {"bg","tract","county"}
      - source_type in {"shp","zip"}
    """
    layer = str(layer).strip().lower()
    if layer not in {"bg", "tract", "county"}:
        raise ValueError(f"Unsupported TIGER layer: {layer!r} (expected 'bg', 'tract', or 'county')")
    state_fips2 = str(state_fips).zfill(2)
    shp = census_data_dir / str(year) / state_fips2 / f"tl_{year}_{state_fips2}_{layer}.shp"
    if shp.exists():
        return "shp", str(shp)
    z = census_data_dir / str(year) / layer.upper() / f"tl_{year}_{state_fips2}_{layer}.zip"
    if z.exists():
        return "zip", str(z)
    raise FileNotFoundError(
        f"TIGER {layer} shapefile not found for year={year} state={state_fips2} under {census_data_dir}"
    )


def _find_gdb_layer(census_data_dir: Path, year: int, state_fips: str, layer: str) -> Tuple[str, str]:
    """
    Best-effort fallback to locate a .gdb containing the requested layer.
    Returns (gdb_path, layer_name).
    """
    import geopandas as gpd  # type: ignore

    layer = str(layer).strip().lower()
    want = layer  # Use the layer name directly (tract, bg, or county)
    state_fips2 = str(state_fips).zfill(2)

    # For county, also accept COUSUB (County Subdivision) as a fallback
    search_terms = [want.upper()]
    if want == "county":
        search_terms.append("COUSUB")

    candidates: List[Path] = []
    for base in [census_data_dir, census_data_dir / str(year)]:
        if base.exists():
            candidates.extend(sorted(base.glob("*.gdb")))

    def score_path(p: Path) -> int:
        name = p.name.upper()
        score = 0
        if str(year) in name:
            score += 2
        if state_fips2 in name:
            score += 2
        # Score if any search term matches
        for term in search_terms:
            if term in name:
                score += 1
                break
        return score

    candidates = sorted(candidates, key=score_path, reverse=True)
    for gdb in candidates:
        try:
            layers = gpd.list_layers(gdb)
        except Exception:
            continue
        # Prefer layers whose name includes the requested layer type.
        layer_names = layers.get("name", []).tolist() if hasattr(layers, "get") else list(layers["name"])
        # Build preferred list: layers matching any search term
        preferred = []
        for ln in layer_names:
            ln_upper = str(ln).upper()
            if any(term in ln_upper for term in search_terms):
                preferred.append(ln)
        # Try preferred first, then others
        for ln in preferred + [ln for ln in layer_names if ln not in preferred]:
            try:
                gdf = gpd.read_file(gdb, layer=ln)
            except Exception:
                continue
            if "GEOID" in gdf.columns and "geometry" in gdf.columns:
                return str(gdb), str(ln)

    raise FileNotFoundError(
        f"No .gdb with layer '{want}' (or COUSUB for county) found under {census_data_dir} for year={year} state={state_fips2}"
    )


def _load_bg_gdf_with_centroids(
    cbg_data_dir: Path,
    year: int,
    state_fips: str,
    geoid_col: str = "GEOID",
) -> "pd.DataFrame":
    """
    Load BG polygons and compute deterministic centroids.
    Returns a GeoDataFrame with columns: GEOID, geometry, centroid_lat, centroid_lon (WGS84).
    """
    return _load_tiger_gdf_with_centroids(census_data_dir=cbg_data_dir, year=year, state_fips=state_fips, layer="bg", geoid_col=geoid_col)


def _load_tract_gdf_with_centroids(
    census_data_dir: Path,
    year: int,
    state_fips: str,
    geoid_col: str = "GEOID",
) -> "pd.DataFrame":
    """
    Load tract polygons and compute deterministic centroids.
    Returns a GeoDataFrame with columns: GEOID, geometry, centroid_lat, centroid_lon (WGS84).
    """
    return _load_tiger_gdf_with_centroids(census_data_dir=census_data_dir, year=year, state_fips=state_fips, layer="tract", geoid_col=geoid_col)


def _load_county_gdf_with_centroids(
    census_data_dir: Path,
    year: int,
    state_fips: str,
    geoid_col: str = "GEOID",
) -> "pd.DataFrame":
    """
    Load county polygons and compute deterministic centroids.
    Returns a GeoDataFrame with columns: GEOID, geometry, centroid_lat, centroid_lon (WGS84).
    """
    return _load_tiger_gdf_with_centroids(census_data_dir=census_data_dir, year=year, state_fips=state_fips, layer="county", geoid_col=geoid_col)


def _load_tiger_gdf_with_centroids(
    census_data_dir: Path,
    year: int,
    state_fips: str,
    layer: str,
    geoid_col: str = "GEOID",
) -> "pd.DataFrame":
    """
    Load a TIGER/Line layer and compute deterministic centroids.
    Returns a GeoDataFrame with columns: GEOID, geometry, centroid_lat, centroid_lon (WGS84).
    """
    import geopandas as gpd  # type: ignore

    try:
        src_type, path = _find_tiger_path(census_data_dir=census_data_dir, year=year, state_fips=state_fips, layer=layer)
        if src_type == "zip":
            gdf = gpd.read_file("zip://" + path)
        else:
            gdf = gpd.read_file(path)
    except FileNotFoundError:
        gdb_path, layer_name = _find_gdb_layer(census_data_dir=census_data_dir, year=year, state_fips=state_fips, layer=layer)
        gdf = gpd.read_file(gdb_path, layer=layer_name)

    if geoid_col not in gdf.columns:
        raise ValueError(f"Expected column '{geoid_col}' in TIGER {layer} shapefile, got columns={list(gdf.columns)}")

    gdf[geoid_col] = gdf[geoid_col].astype(str)
    if gdf.crs is None:
        raise ValueError(f"TIGER {layer} shapefile CRS is None: year={year} state={str(state_fips).zfill(2)}")

    # Centroids in projected CRS, then convert back to WGS84.
    gdf_proj = gdf.to_crs("EPSG:3857")
    cent_proj = gdf_proj.geometry.centroid
    cent_wgs84 = gpd.GeoSeries(cent_proj, crs="EPSG:3857").to_crs("EPSG:4326")
    gdf = gdf.copy()
    gdf["centroid_lat"] = cent_wgs84.y
    gdf["centroid_lon"] = cent_wgs84.x
    return gdf


def _map_points_to_bg(
    points_df: pd.DataFrame,
    bg2020: "pd.DataFrame",
    bg2015: Optional["pd.DataFrame"],
    lat_col: str,
    lon_col: str,
    geoid_col: str = "GEOID",
) -> pd.DataFrame:
    """
    Map lat/lon points to BG polygons with precedence:
      within 2020 -> within 2015 (optional) -> nearest (2020 centroids)
    Returns points_df with added columns:
      bg_geoid, bg_year, match_method, centroid_lat, centroid_lon
    """
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
    from shapely.strtree import STRtree  # type: ignore

    return _map_points_to_tiger(
        points_df=points_df,
        gdf2020=bg2020,
        gdf2015=bg2015,
        lat_col=lat_col,
        lon_col=lon_col,
        geoid_col=geoid_col,
        out_prefix="bg",
    )


def _map_points_to_tract(
    points_df: pd.DataFrame,
    tract2020: "pd.DataFrame",
    tract2015: Optional["pd.DataFrame"],
    lat_col: str,
    lon_col: str,
    geoid_col: str = "GEOID",
) -> pd.DataFrame:
    """
    Map lat/lon points to tract polygons. See _map_points_to_bg docstring for matching behavior.
    """
    return _map_points_to_tiger(
        points_df=points_df,
        gdf2020=tract2020,
        gdf2015=tract2015,
        lat_col=lat_col,
        lon_col=lon_col,
        geoid_col=geoid_col,
        out_prefix="tract",
    )


def _map_points_to_county(
    points_df: pd.DataFrame,
    county2020: "pd.DataFrame",
    county2015: Optional["pd.DataFrame"],
    lat_col: str,
    lon_col: str,
    geoid_col: str = "GEOID",
) -> pd.DataFrame:
    """
    Map lat/lon points to county polygons. See _map_points_to_bg docstring for matching behavior.
    """
    return _map_points_to_tiger(
        points_df=points_df,
        gdf2020=county2020,
        gdf2015=county2015,
        lat_col=lat_col,
        lon_col=lon_col,
        geoid_col=geoid_col,
        out_prefix="county",
    )


def _map_points_to_tiger(
    points_df: pd.DataFrame,
    gdf2020: "pd.DataFrame",
    gdf2015: Optional["pd.DataFrame"],
    lat_col: str,
    lon_col: str,
    geoid_col: str = "GEOID",
    out_prefix: str = "geo",
) -> pd.DataFrame:
    """
    Map lat/lon points to TIGER polygons with precedence:
      within 2020 -> within 2015 (optional) -> nearest (2020 centroids)
    Returns points_df with added columns:
      <out_prefix>_geoid, <out_prefix>_year, match_method, centroid_lat, centroid_lon
    """
    import geopandas as gpd  # type: ignore
    from shapely.strtree import STRtree  # type: ignore

    out = points_df.copy()
    geoid_out = f"{out_prefix}_geoid"
    year_out = f"{out_prefix}_year"
    out[geoid_out] = pd.Series([None] * len(out), dtype="object")
    out[year_out] = pd.Series([None] * len(out), dtype="object")
    out["match_method"] = pd.Series([None] * len(out), dtype="object")
    out["centroid_lat"] = pd.Series([np.nan] * len(out), dtype="float64")
    out["centroid_lon"] = pd.Series([np.nan] * len(out), dtype="float64")

    valid = out[lat_col].notna() & out[lon_col].notna()
    if valid.sum() == 0:
        return out

    pts = gpd.GeoDataFrame(
        out.loc[valid, [lat_col, lon_col]].copy(),
        geometry=gpd.points_from_xy(out.loc[valid, lon_col], out.loc[valid, lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    gdf2020_gdf = gdf2020[[geoid_col, "geometry", "centroid_lat", "centroid_lon"]].copy()
    gdf2020_gdf = gdf2020_gdf.to_crs("EPSG:3857")
    joined20 = gpd.sjoin(pts, gdf2020_gdf[[geoid_col, "geometry"]], how="left", predicate="within")
    out.loc[joined20.index, geoid_out] = joined20[geoid_col].where(joined20[geoid_col].notna(), None).astype("object")
    out.loc[joined20.index, year_out] = np.where(joined20[geoid_col].notna(), 2020, None)
    out.loc[joined20.index, "match_method"] = np.where(joined20[geoid_col].notna(), "within", None)

    # Optional 2015 fallback for still-unmatched points
    if gdf2015 is not None:
        still = valid & out[geoid_out].isna()
        if still.sum() > 0:
            pts15 = gpd.GeoDataFrame(
                out.loc[still, [lat_col, lon_col]].copy(),
                geometry=gpd.points_from_xy(out.loc[still, lon_col], out.loc[still, lat_col]),
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")
            gdf2015_gdf = gdf2015[[geoid_col, "geometry", "centroid_lat", "centroid_lon"]].copy().to_crs("EPSG:3857")
            joined15 = gpd.sjoin(pts15, gdf2015_gdf[[geoid_col, "geometry"]], how="left", predicate="within")
            out.loc[joined15.index, geoid_out] = joined15[geoid_col].where(joined15[geoid_col].notna(), None).astype("object")
            out.loc[joined15.index, year_out] = np.where(joined15[geoid_col].notna(), 2015, out.loc[joined15.index, year_out])
            out.loc[joined15.index, "match_method"] = np.where(
                joined15[geoid_col].notna(), "within_fallback_2015", out.loc[joined15.index, "match_method"]
            )

    # Nearest fallback for any remaining unmatched (always use 2020 centroids)
    still = valid & out[geoid_out].isna()
    if still.sum() > 0:
        cent_pts = gdf2020_gdf.geometry.centroid
        tree = STRtree(list(cent_pts))
        pts_near = gpd.GeoDataFrame(
            out.loc[still, [lat_col, lon_col]].copy(),
            geometry=gpd.points_from_xy(out.loc[still, lon_col], out.loc[still, lat_col]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        nearest_idx = [int(tree.nearest(geom)) for geom in pts_near.geometry]
        nearest_geoids = gdf2020_gdf.iloc[nearest_idx][geoid_col].astype(str).tolist()
        out.loc[pts_near.index, geoid_out] = nearest_geoids
        out.loc[pts_near.index, year_out] = 2020
        out.loc[pts_near.index, "match_method"] = "nearest_2020"

    # Attach centroid coords for matched geoids (per year used)
    m20 = out[year_out].astype("object") == 2020
    if m20.sum() > 0:
        geo2cent20 = gdf2020.set_index(geoid_col)[["centroid_lat", "centroid_lon"]]
        cent = geo2cent20.reindex(out.loc[m20, geoid_out].astype(str))
        out.loc[m20, "centroid_lat"] = cent["centroid_lat"].to_numpy()
        out.loc[m20, "centroid_lon"] = cent["centroid_lon"].to_numpy()
    if gdf2015 is not None:
        m15 = out[year_out].astype("object") == 2015
        if m15.sum() > 0:
            geo2cent15 = gdf2015.set_index(geoid_col)[["centroid_lat", "centroid_lon"]]
            cent = geo2cent15.reindex(out.loc[m15, geoid_out].astype(str))
            out.loc[m15, "centroid_lat"] = cent["centroid_lat"].to_numpy()
            out.loc[m15, "centroid_lon"] = cent["centroid_lon"].to_numpy()

    return out


def _format_geo_token(prefix: str, geo: str, lat: float, lon: float, decimals: int = 6) -> str:
    return f"{prefix}_{geo}_{lat:.{decimals}f}_{lon:.{decimals}f}"


def _format_cbg_token(prefix: str, lat: float, lon: float, decimals: int = 6) -> str:
    return _format_geo_token(prefix=prefix, geo="CBG", lat=lat, lon=lon, decimals=decimals)


def _format_tract_token(prefix: str, lat: float, lon: float, decimals: int = 6) -> str:
    return _format_geo_token(prefix=prefix, geo="TRACT", lat=lat, lon=lon, decimals=decimals)


def _format_county_token(prefix: str, lat: float, lon: float, decimals: int = 6) -> str:
    return _format_geo_token(prefix=prefix, geo="COUNTY", lat=lat, lon=lon, decimals=decimals)


def pad_time_arrays(
    arrival_ts_seqs: List[List],
    dwell_min_seqs: List[List],
    max_len: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build padded timestamp + dwell arrays, plus length ids.

    Returns:
      timestamps: datetime64[ns] [M, max_len]
      dwell_sec: float32 [M, max_len]
      length_ids: int64 [M]
    """
    lengths = np.asarray([len(s) for s in arrival_ts_seqs], dtype=np.int64)
    max_len = int(max_len)

    timestamps = np.full((len(arrival_ts_seqs), max_len), np.datetime64("NaT"), dtype="datetime64[ns]")
    dwell_sec = np.zeros((len(arrival_ts_seqs), max_len), dtype=np.float32)

    for i, (ts_list, dm_list) in enumerate(zip(arrival_ts_seqs, dwell_min_seqs)):
        L = min(len(ts_list), max_len)
        if L <= 0:
            continue
        # timestamps
        ts_arr = np.asarray(pd.to_datetime(ts_list[:L], errors="coerce"), dtype="datetime64[ns]")
        timestamps[i, :L] = ts_arr
        # dwell seconds
        dm = pd.to_numeric(pd.Series(dm_list[:L]), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        dwell_sec[i, :L] = dm * 60.0

    length_ids = lengths.astype(np.int64, copy=False)
    return timestamps, dwell_sec, length_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traj-csv",
        type=str,
        default="/path/to/traj.csv",
        help="Path to traj.csv (change to your own folder path).",
    )
    ap.add_argument(
        "--demo-csv",
        type=str,
        default="/path/to/demo.csv",
        help="Path to demo.csv (change to your own folder path).",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="/path/to/YOUR_DATA_FOLDER/controlled",
        help="Output root containing train/val/test subfolders (change to your own folder path).",
    )
    ap.add_argument(
        "--city",
        type=str,
        default="carlos",
        help="City label to store in final_segments_all_train_data.pkl (required by some training scripts).",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Fixed max sequence length to store in arrays (sequences are truncated to this length).",
    )
    ap.add_argument(
        "--no-special-tokens",
        dest="add_special_tokens",
        action="store_false",
        default=True,
        help="If set, do not add [CLS]/[SEP] to sequences.",
    )
    ap.add_argument(
        "--no-pad-to-max-len",
        dest="pad_to_max_len",
        action="store_false",
        default=True,
        help="If set, do not pad sequences to max_len with [PAD].",
    )
    ap.add_argument(
        "--token-col",
        type=str,
        default="poi_id_model",
        choices=["poi_id_model", "poi_raw_id", "poi_token"],
        help="Which traj.csv column to use as the sequence token.",
    )
    ap.add_argument(
        "--include-demo",
        action="store_true",
        help="If set, also write all_attr_results_with_demo.npy (age_bin5 + gender_id).",
    )
    ap.add_argument(
        "--drop-missing-demo",
        action="store_true",
        help="If set, drop segments whose panelist_id is missing home/work coords in demo.csv.",
    )
    ap.add_argument(
        "--replace-home-work-with-cbg",
        action="store_true",
        help="If set, replace POI_HOME/POI_WORK tokens with HOME_CBG_<centroidLat>_<centroidLon> / WORK_CBG_<...> (TIGER BG; 2020 preferred, 2015 fallback, nearest fallback).",
    )
    ap.add_argument(
        "--replace-home-work-with-tract",
        action="store_true",
        help="If set, replace POI_HOME/POI_WORK tokens with HOME_TRACT_<centroidLat>_<centroidLon> / WORK_TRACT_<...> (TIGER tract; 2020 preferred, 2015 fallback, nearest fallback).",
    )
    ap.add_argument(
        "--replace-home-work-with-county",
        action="store_true",
        help="If set, replace POI_HOME/POI_WORK tokens with HOME_COUNTY_<centroidLat>_<centroidLon> / WORK_COUNTY_<...> (TIGER county; 2020 preferred, 2015 fallback, nearest fallback).",
    )
    ap.add_argument(
        "--cbg-data-dir",
        type=str,
        default="/path/to/census_data",
        help="Directory containing TIGER shapefiles (change to your own folder path). Expects {year}/{state_fips}/tl_{year}_{state_fips}_{bg|tract|county}.shp or {year}/{BG|TRACT|COUNTY}/tl_{year}_{state_fips}_{bg|tract|county}.zip.",
    )
    ap.add_argument(
        "--cbg-state-fips",
        type=str,
        default="51",
        help="2-digit state FIPS used for BG polygons (default 51=VA).",
    )
    ap.add_argument(
        "--cbg-token-decimals",
        type=int,
        default=6,
        help="Decimal places to round centroid lat/lon when forming tokens.",
    )
    ap.add_argument(
        "--cbg-cache-pkl",
        type=str,
        default="",
        help="Optional path to write/read cached panelist_id->home/work CBG centroid tokens (pickle). Defaults to <demo_dir>/home_work_cbg_tokens_<state>.pkl",
    )
    ap.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default="4GB",
        help="DuckDB memory limit (e.g. 4GB, 8GB). Lower values reduce OOM risk.",
    )
    ap.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB threads. Fewer threads use less memory.",
    )
    ap.add_argument(
        "--duckdb-temp-dir",
        type=str,
        default="",
        help="Optional temp directory for DuckDB spill files. Defaults to a new temp folder.",
    )
    ap.add_argument(
        "--segment-buckets",
        type=int,
        default=1,
        help="Split each split into N hash buckets to reduce DuckDB memory usage (1 = no bucketing).",
    )
    args = ap.parse_args()

    traj_csv = Path(args.traj_csv)
    demo_csv = Path(args.demo_csv)
    output_root = Path(args.output_root)

    if not traj_csv.exists():
        raise FileNotFoundError(f"traj.csv not found: {traj_csv}")
    if not demo_csv.exists():
        raise FileNotFoundError(f"demo.csv not found: {demo_csv}")

    # Load demo
    demo = _load_demo(demo_csv)
    demo = demo.drop_duplicates(subset=["panelist_id"], keep="first").copy()
    demo["gender_id"] = demo.get("gender", pd.Series([None] * len(demo))).apply(gender_to_id)
    demo["age_bin5"] = demo.get("age", pd.Series([None] * len(demo))).apply(age_to_bin5)

    # Optional: map home/work to BG, tract, or county centroids and build replacement tokens
    demo_tokens = None
    geo_token_rows: Optional[pd.DataFrame] = None
    replace_home_work = bool(args.replace_home_work_with_cbg) or bool(args.replace_home_work_with_tract) or bool(args.replace_home_work_with_county)
    replace_count = sum([bool(args.replace_home_work_with_cbg), bool(args.replace_home_work_with_tract), bool(args.replace_home_work_with_county)])
    if replace_count > 1:
        raise ValueError("Specify only one of --replace-home-work-with-cbg, --replace-home-work-with-tract, or --replace-home-work-with-county")
    if replace_home_work:
        if bool(args.replace_home_work_with_cbg):
            geo_kind = "cbg"
        elif bool(args.replace_home_work_with_tract):
            geo_kind = "tract"
        else:
            geo_kind = "county"
        cbg_data_dir = Path(args.cbg_data_dir)
        state_fips = str(args.cbg_state_fips).zfill(2)
        cache_pkl = (
            Path(args.cbg_cache_pkl)
            if str(args.cbg_cache_pkl).strip()
            else (demo_csv.parent / f"home_work_{geo_kind}_tokens_{state_fips}.pkl")
        )

        if cache_pkl.exists():
            cached = pd.read_pickle(cache_pkl)
            if not isinstance(cached, pd.DataFrame):
                raise ValueError(f"Unexpected cache format (expected DataFrame): {cache_pkl}")
            demo_tokens = cached.copy()
            print(f"[INFO] Loaded cached home/work {geo_kind.upper()} tokens: {cache_pkl} (n={len(demo_tokens)})")
        else:
            if geo_kind == "cbg":
                gdf2020 = _load_bg_gdf_with_centroids(cbg_data_dir=cbg_data_dir, year=2020, state_fips=state_fips)
                try:
                    gdf2015 = _load_bg_gdf_with_centroids(cbg_data_dir=cbg_data_dir, year=2015, state_fips=state_fips)
                except Exception:
                    gdf2015 = None
                map_fn = _map_points_to_bg
                tok_fn = _format_cbg_token
            elif geo_kind == "tract":
                gdf2020 = _load_tract_gdf_with_centroids(census_data_dir=cbg_data_dir, year=2020, state_fips=state_fips)
                try:
                    gdf2015 = _load_tract_gdf_with_centroids(census_data_dir=cbg_data_dir, year=2015, state_fips=state_fips)
                except Exception:
                    gdf2015 = None
                map_fn = _map_points_to_tract
                tok_fn = _format_tract_token
            else:  # county
                gdf2020 = _load_county_gdf_with_centroids(census_data_dir=cbg_data_dir, year=2020, state_fips=state_fips)
                try:
                    gdf2015 = _load_county_gdf_with_centroids(census_data_dir=cbg_data_dir, year=2015, state_fips=state_fips)
                except Exception:
                    gdf2015 = None
                map_fn = _map_points_to_county
                tok_fn = _format_county_token

            # Home mapping
            home_pts = demo[["panelist_id", "home_lat", "home_lon"]].copy()
            home_mapped = map_fn(
                home_pts.rename(columns={"home_lat": "lat", "home_lon": "lon"}),
                gdf2020,
                gdf2015,
                lat_col="lat",
                lon_col="lon",
            )
            # Work mapping
            work_pts = demo[["panelist_id", "work_lat", "work_lon"]].copy()
            work_mapped = map_fn(
                work_pts.rename(columns={"work_lat": "lat", "work_lon": "lon"}),
                gdf2020,
                gdf2015,
                lat_col="lat",
                lon_col="lon",
            )

            # Build per-panelist tokens
            decimals = int(args.cbg_token_decimals)
            if geo_kind == "cbg":
                year_col = "bg_year"
            elif geo_kind == "tract":
                year_col = "tract_year"
            else:  # county
                year_col = "county_year"
            demo_tokens = pd.DataFrame({"panelist_id": demo["panelist_id"].astype(str)})
            demo_tokens = demo_tokens.merge(
                home_mapped[["panelist_id", "centroid_lat", "centroid_lon", "match_method", year_col]].rename(
                    columns={
                        "centroid_lat": "home_centroid_lat",
                        "centroid_lon": "home_centroid_lon",
                        "match_method": "home_match_method",
                        year_col: f"home_{geo_kind}_year",
                    }
                ),
                on="panelist_id",
                how="left",
            ).merge(
                work_mapped[["panelist_id", "centroid_lat", "centroid_lon", "match_method", year_col]].rename(
                    columns={
                        "centroid_lat": "work_centroid_lat",
                        "centroid_lon": "work_centroid_lon",
                        "match_method": "work_match_method",
                        year_col: f"work_{geo_kind}_year",
                    }
                ),
                on="panelist_id",
                how="left",
            )

            def _mk(prefix: str, lat, lon):
                if lat is None or lon is None or (isinstance(lat, float) and np.isnan(lat)) or (isinstance(lon, float) and np.isnan(lon)):
                    return None
                return tok_fn(prefix, float(lat), float(lon), decimals=decimals)

            demo_tokens["home_token"] = [
                _mk("HOME", la, lo) for la, lo in zip(demo_tokens["home_centroid_lat"].tolist(), demo_tokens["home_centroid_lon"].tolist())
            ]
            demo_tokens["work_token"] = [
                _mk("WORK", la, lo) for la, lo in zip(demo_tokens["work_centroid_lat"].tolist(), demo_tokens["work_centroid_lon"].tolist())
            ]

            # Print match stats
            home_unmatched = demo_tokens["home_token"].isna().sum()
            work_unmatched = demo_tokens["work_token"].isna().sum()
            print(
                f"[INFO] Home/work -> {geo_kind.upper()} tokens: n={len(demo_tokens)} | "
                f"home_unmatched={home_unmatched} work_unmatched={work_unmatched} | "
                f"state_fips={state_fips} (2020 preferred, 2015 fallback, nearest fallback)"
            )

            demo_tokens.to_pickle(cache_pkl)
            print(f"[INFO] Wrote cached home/work {geo_kind.upper()} tokens: {cache_pkl}")

        # Build a unique token table to append into poi_map_feature (lat/lon from centroid)
        ht = demo_tokens[["home_token", "home_centroid_lat", "home_centroid_lon"]].dropna().rename(
            columns={"home_token": "poi_id", "home_centroid_lat": "lat", "home_centroid_lon": "lon"}
        )
        ht["top_category"] = "POI_HOME"
        ht["sub_category"] = f"home_{geo_kind}"
        ht["placekey"] = None
        wt = demo_tokens[["work_token", "work_centroid_lat", "work_centroid_lon"]].dropna().rename(
            columns={"work_token": "poi_id", "work_centroid_lat": "lat", "work_centroid_lon": "lon"}
        )
        wt["top_category"] = "POI_WORK"
        wt["sub_category"] = f"work_{geo_kind}"
        wt["placekey"] = None
        geo_token_rows = pd.concat([ht, wt], ignore_index=True)
        geo_token_rows = geo_token_rows.drop_duplicates(subset=["poi_id"]).copy()

    # DuckDB aggregate
    con = _duckdb_connect(
        memory_limit=args.duckdb_memory_limit,
        threads=args.duckdb_threads,
        temp_dir=args.duckdb_temp_dir,
    )

    # If using home/work replacement, register per-panelist token mapping for use in SQL expressions
    if replace_home_work and demo_tokens is not None:
        con.register("demo_tokens", demo_tokens[["panelist_id", "home_token", "work_token"]].copy())

    # Check if category column exists in traj.csv
    try:
        columns_result = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto({_sql_quote(str(traj_csv))}, header=True)").fetchall()
        available_columns = [row[0] for row in columns_result]
        has_category = "category" in available_columns
    except Exception:
        available_columns = []
        has_category = False

    # Support multiple schemas: some traj.csv use poi_token instead of poi_id_model
    if "poi_id_model" in available_columns:
        model_token_source = "poi_id_model"
    elif "poi_token" in available_columns:
        model_token_source = "poi_token"
    else:
        raise RuntimeError(
            "traj.csv schema missing expected model-token column (expected one of: poi_id_model, poi_token). "
            f"Available columns: {available_columns}"
        )

    # Create view once (optionally left-join demo_tokens for home/work replacement)
    if has_category:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW traj AS
            SELECT
              t.segment_id::VARCHAR AS segment_id,
              t.panelist_id::VARCHAR AS panelist_id,
              t.city::VARCHAR AS city,
              try_cast(t.arrival_ts AS TIMESTAMP) AS arrival_ts,
              try_cast(t.departure_ts AS TIMESTAMP) AS departure_ts,
              t.{model_token_source}::VARCHAR AS poi_id_model,
              t.poi_raw_id::VARCHAR AS poi_raw_id,
              try_cast(t.latitude AS DOUBLE) AS latitude,
              try_cast(t.longitude AS DOUBLE) AS longitude,
              t.location_name::VARCHAR AS location_name,
              try_cast(t.duration_minutes AS DOUBLE) AS duration_minutes,
              t.split::VARCHAR AS split,
              t.category::VARCHAR AS category
              {", dt.home_token AS home_token, dt.work_token AS work_token" if replace_home_work else ""}
            FROM read_csv_auto({_sql_quote(str(traj_csv))}, header=True) t
            {"LEFT JOIN demo_tokens dt ON dt.panelist_id = t.panelist_id::VARCHAR" if replace_home_work else ""}
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW traj AS
            SELECT
              t.segment_id::VARCHAR AS segment_id,
              t.panelist_id::VARCHAR AS panelist_id,
              t.city::VARCHAR AS city,
              try_cast(t.arrival_ts AS TIMESTAMP) AS arrival_ts,
              try_cast(t.departure_ts AS TIMESTAMP) AS departure_ts,
              t.{model_token_source}::VARCHAR AS poi_id_model,
              t.poi_raw_id::VARCHAR AS poi_raw_id,
              try_cast(t.latitude AS DOUBLE) AS latitude,
              try_cast(t.longitude AS DOUBLE) AS longitude,
              t.location_name::VARCHAR AS location_name,
              try_cast(t.duration_minutes AS DOUBLE) AS duration_minutes,
              t.split::VARCHAR AS split
              {", dt.home_token AS home_token, dt.work_token AS work_token" if replace_home_work else ""}
            FROM read_csv_auto({_sql_quote(str(traj_csv))}, header=True) t
            {"LEFT JOIN demo_tokens dt ON dt.panelist_id = t.panelist_id::VARCHAR" if replace_home_work else ""}
            """
        )

    # POI map (shared)
    # If using poi_id_model or poi_token as the token, poi_map_feature should be keyed by that token.
    # This provides per-token coords and a lightweight category label for phase2 mappings.
    poi_map_df = build_poi_map_feature(con, has_category=has_category)
    if args.token_col == "poi_id_model" or args.token_col == "poi_token":
        # Re-key poi_map to model tokens by recomputing with poi_id_model as the identifier.
        # Note: poi_token is mapped to poi_id_model in the traj view, so we use poi_id_model here.
        # We treat poi_id_model as the token and keep representative lat/lon.
        if has_category:
            poi_map_df = con.execute(
                """
                WITH counts AS (
                  SELECT
                    poi_id_model AS tok,
                    latitude,
                    longitude,
                    COALESCE(category, poi_id_model) AS top_category,
                    location_name,
                    COUNT(*) AS cnt
                  FROM traj
                  WHERE poi_id_model IS NOT NULL
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                  GROUP BY 1,2,3,4,5
                ),
                ranked AS (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY tok ORDER BY cnt DESC) AS rn
                  FROM counts
                )
                SELECT
                  tok AS poi_id,
                  latitude AS lat,
                  longitude AS lon,
                  top_category AS top_category,
                  COALESCE(location_name, 'unknown') AS sub_category,
                  NULL::VARCHAR AS placekey
                FROM ranked
                WHERE rn = 1
                ORDER BY poi_id
                """
            ).fetch_df()
        else:
            poi_map_df = con.execute(
                """
                WITH counts AS (
                  SELECT
                    poi_id_model AS tok,
                    latitude,
                    longitude,
                    poi_id_model AS top_category,
                    location_name,
                    COUNT(*) AS cnt
                  FROM traj
                  WHERE poi_id_model IS NOT NULL
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                  GROUP BY 1,2,3,4,5
                ),
                ranked AS (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY tok ORDER BY cnt DESC) AS rn
                  FROM counts
                )
                SELECT
                  tok AS poi_id,
                  latitude AS lat,
                  longitude AS lon,
                  top_category AS top_category,
                  COALESCE(location_name, 'unknown') AS sub_category,
                  NULL::VARCHAR AS placekey
                FROM ranked
                WHERE rn = 1
                ORDER BY poi_id
                """
            ).fetch_df()

    # If replacing home/work, remove old home/work tokens from the POI map and add centroid tokens
    if replace_home_work:
        before = len(poi_map_df)
        # Drop common home/work token patterns and model tokens
        poi_id_s = poi_map_df["poi_id"].astype(str)
        drop_mask = (
            poi_id_s.isin(["POI_HOME", "POI_WORK"])
            | poi_id_s.str.startswith("home_", na=False)
            | poi_id_s.str.startswith("work_", na=False)
        )
        if "top_category" in poi_map_df.columns:
            drop_mask = drop_mask | poi_map_df["top_category"].astype(str).isin(["POI_HOME", "POI_WORK"])
        poi_map_df = poi_map_df.loc[~drop_mask].copy()
        after = len(poi_map_df)
        if after < before:
            print(f"[INFO] Dropped {before - after} home/work tokens from poi_map_feature prior to adding centroid tokens")
        if geo_token_rows is not None and not geo_token_rows.empty:
            poi_map_df = pd.concat([poi_map_df, geo_token_rows], ignore_index=True)
            poi_map_df = poi_map_df.drop_duplicates(subset=["poi_id"]).sort_values(by="poi_id").reset_index(drop=True)

    # Tokenizer vocab tokens (shared)
    # NOTE: We intentionally build tokenizer + write poi_map_feature.csv *after* we generate
    # all split sequences, so we can prune tokens that never appear in any segment sequences.
    # This prevents "dead" vocab entries (e.g., HOME/WORK CBG tokens present in demo but
    # absent from all train/val/test sequences).
    bert_specials = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]

    # Determine which splits exist in traj
    splits = con.execute("SELECT DISTINCT split FROM traj WHERE split IS NOT NULL ORDER BY split").fetchall()
    split_names = [s[0] for s in splits]
    wanted = ["train", "val", "test"]
    split_names = [s for s in wanted if s in split_names]
    if not split_names:
        raise RuntimeError("No split values found (expected train/val/test in traj.csv 'split' column).")

    # Ensure output dirs
    for split_name in split_names:
        split_dir = output_root / split_name
        _ensure_dir(split_dir)

    # Build each split
    used_tokens_all: set = set()
    for split_name in split_names:
        split_dir = output_root / split_name
        if args.token_col == "poi_id_model" or args.token_col == "poi_token":
            # poi_token is mapped to poi_id_model in the traj view, so use poi_id_model for both
            token_expr_base = "poi_id_model"
        else:
            token_expr_base = "poi_raw_id"
        if replace_home_work:
            # Prefer mapping using poi_id_model to identify home/work rows, regardless of token base.
            token_expr = (
                f"CASE "
                f"WHEN poi_id_model = 'POI_HOME' AND home_token IS NOT NULL THEN home_token "
                f"WHEN poi_id_model = 'POI_WORK' AND work_token IS NOT NULL THEN work_token "
                f"ELSE {token_expr_base} END"
            )
        else:
            token_expr = token_expr_base
        bucket_count = max(1, int(args.segment_buckets))
        if bucket_count == 1:
            seq_df = build_sequences_for_split(
                con,
                split_name,
                token_sql_expr=token_expr,
                max_len=int(args.max_len),
            )
        else:
            parts: List[pd.DataFrame] = []
            for bucket_idx in range(bucket_count):
                part = build_sequences_for_split(
                    con,
                    split_name,
                    token_sql_expr=token_expr,
                    max_len=int(args.max_len),
                    bucket_count=bucket_count,
                    bucket_idx=bucket_idx,
                )
                if not part.empty:
                    parts.append(part)
            seq_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if seq_df.empty:
            print(f"[WARN] Split {split_name}: no sequences found")
            continue

        # Rename for downstream consistency
        seq_df = seq_df.rename(columns={"panelist_id": "individual_id"}).copy()

        # Add special tokens and pad to max_len to match Richmond-style sequences.
        # Keep length_id untouched (it reflects the raw sequence length before specials/pad).
        if args.add_special_tokens or args.pad_to_max_len:
            max_len = int(args.max_len)
            seqs: List[List[str]] = []
            masks: List[List[int]] = []
            for seq in seq_df["unique_id_seq"].tolist():
                new_seq, new_mask = _apply_special_tokens_and_padding(
                    seq=seq,
                    max_len=max_len,
                    add_special_tokens=bool(args.add_special_tokens),
                    pad_to_max_len=bool(args.pad_to_max_len),
                )
                seqs.append(new_seq)
                masks.append(new_mask)
            seq_df["unique_id_seq"] = seqs
            seq_df["attention_mask"] = masks

        # Merge demo for attributes
        merged = seq_df.merge(
            demo[
                [
                    "panelist_id",
                    "home_lat",
                    "home_lon",
                    "work_lat",
                    "work_lon",
                    "age_bin5",
                    "gender_id",
                ]
            ].rename(columns={"panelist_id": "individual_id"}),
            on="individual_id",
            how="left",
        )

        # Drop missing demo if requested
        if args.drop_missing_demo:
            before = len(merged)
            merged = merged.dropna(subset=["home_lat", "home_lon", "work_lat", "work_lon"]).copy()
            after = len(merged)
            if after < before:
                print(f"[INFO] Split {split_name}: dropped {before - after} segments missing home/work coords")

        # Core dataframe to pickle
        city_series = merged.get("city")
        if city_series is None:
            city_values = [str(args.city)] * len(merged)
        else:
            # Fill missing cities with fallback label
            city_values = city_series.fillna(str(args.city)).astype(str).tolist()

        out_df = pd.DataFrame(
            {
                "segment_id": merged["segment_id"].astype(str).tolist(),
                "individual_id": merged["individual_id"].astype(str).tolist(),
                "city": city_values,
                "unique_id_seq": merged["unique_id_seq"].tolist(),
                "attention_mask": merged["attention_mask"].tolist(),
            }
        )
        out_df.to_pickle(split_dir / "final_segments_all_train_data.pkl")
        # Track which tokens actually appear in any generated sequence (across all splits).
        # We'll use this to prune poi_map_feature + tokenizer vocab to avoid dead tokens.
        for seq in out_df["unique_id_seq"].tolist():
            used_tokens_all.update(seq)

        # Attributes arrays
        attrs4 = np.stack(
            [
                merged["work_lat"].fillna(0.0).to_numpy(dtype=np.float32),
                merged["work_lon"].fillna(0.0).to_numpy(dtype=np.float32),
                merged["home_lat"].fillna(0.0).to_numpy(dtype=np.float32),
                merged["home_lon"].fillna(0.0).to_numpy(dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        np.save(split_dir / "all_attr_results.npy", attrs4)

        if args.include_demo:
            age_bin5 = merged["age_bin5"].fillna(-1).to_numpy(dtype=np.float32)
            gender_id = merged["gender_id"].fillna(-1).to_numpy(dtype=np.float32)
            attrs6 = np.concatenate([attrs4, age_bin5[:, None], gender_id[:, None]], axis=1).astype(
                np.float32, copy=False
            )
            np.save(split_dir / "all_attr_results_with_demo.npy", attrs6)

        # Time arrays + length ids
        arrival_ts_seqs = merged["arrival_ts_seq"].tolist()
        dwell_min_seqs = merged["dwell_min_seq"].tolist()
        timestamps, dwell_sec, _ = pad_time_arrays(arrival_ts_seqs, dwell_min_seqs, max_len=int(args.max_len))
        length_ids = merged["length_id"].to_numpy(dtype=np.int64, copy=False)
        np.save(split_dir / "all_timestamp.npy", timestamps)
        np.save(split_dir / "all_dwell.npy", dwell_sec)
        np.save(split_dir / "trajectory_length_ids.npy", length_ids)

        print(
            f"[DONE] {split_name}: {len(out_df)} segments | "
            f"max_len={timestamps.shape[1]} (token_col={args.token_col})"
        )

    # --- Prune poi_map_feature + tokenizer vocab to tokens actually used in sequences ---
    used_tokens_all = set(map(str, used_tokens_all))
    # Exclude special tokens; poi_map_feature should only contain real POI tokens.
    used_tokens_all = used_tokens_all - set(bert_specials)
    poi_id_s = poi_map_df["poi_id"].astype(str)
    missing_in_poi_map = used_tokens_all - set(poi_id_s.tolist())
    if missing_in_poi_map:
        # Fail loudly: downstream often assumes every token has a row in poi_map_feature.csv.
        raise ValueError(
            "Some tokens appear in generated sequences but are missing from poi_map_feature.csv. "
            f"Count={len(missing_in_poi_map)} examples={sorted(list(missing_in_poi_map))[:10]}"
        )

    pruned_poi_map_df = poi_map_df.loc[poi_id_s.isin(used_tokens_all)].copy()
    pruned_poi_map_df = pruned_poi_map_df.drop_duplicates(subset=["poi_id"]).sort_values(by="poi_id").reset_index(drop=True)

    # Build tokenizer vocab from the pruned POI map (plus BERT specials)
    poi_tokens = pruned_poi_map_df["poi_id"].astype(str).tolist()
    vocab_tokens: List[str] = []
    seen = set()
    for t in bert_specials + poi_tokens:
        if t not in seen:
            vocab_tokens.append(t)
            seen.add(t)

    # Write shared tokenizer once under output_root/tokenizer, then copy into each split
    base_tokenizer_dir = output_root / "tokenizer"
    if base_tokenizer_dir.exists():
        shutil.rmtree(base_tokenizer_dir)
    _build_tokenizer(vocab_tokens, base_tokenizer_dir)

    # Overwrite poi_map_feature.csv + tokenizer/ in each split with pruned versions
    for split_name in split_names:
        split_dir = output_root / split_name
        pruned_poi_map_df.to_csv(split_dir / "poi_map_feature.csv", index=False)
        tok_dir = split_dir / "tokenizer"
        if tok_dir.exists():
            shutil.rmtree(tok_dir)
        shutil.copytree(base_tokenizer_dir, tok_dir)

    dead_tokens = set(poi_map_df["poi_id"].astype(str).tolist()) - set(pruned_poi_map_df["poi_id"].astype(str).tolist())
    if dead_tokens:
        print(f"[INFO] Pruned dead tokens from vocab/poi_map_feature: {len(dead_tokens)}")


if __name__ == "__main__":
    main()


