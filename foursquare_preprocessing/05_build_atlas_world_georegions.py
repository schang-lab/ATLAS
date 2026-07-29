#!/usr/bin/env python3
"""
Build ATLAS-world directories for Foursquare NYC data using GEOGRAPHIC regions.

Unlike 05_build_atlas_world_georegions.py, this does NOT rely on home/work
coordinates (which are all-zero for this dataset). Instead each trajectory is
anchored to its MOST-VISITED POI, whose lat/lon (from poi_map_feature.csv) is
classified into an NYC borough, then grouped into 4 georegions:

  - Manhattan
  - Brooklyn
  - Outer       (Queens + Bronx + Staten Island; merged for balance)
  - New_Jersey  (Hoboken / Jersey City / Newark etc., west of the Hudson)

Regions are assigned by TRUE point-in-polygon against census county/state
boundaries (New York county 061 -> Manhattan, 047 -> Brooklyn, 081/005/085
-> Outer; New Jersey state 34 -> New_Jersey). Waterfront/pier POIs that fall
outside every polygon are snapped to the nearest region. This correctly
separates New Jersey from Manhattan along the actual Hudson River state line
(a simple lon threshold is only ~85% accurate there: it mislabels Hudson
piers, the GW Bridge, Liberty Island, etc.). If the shapefiles or geopandas
are unavailable, it falls back to the approximate lat/lon rule (`_georegion`).

Manhattan holds ~58% of trajectories, so it is further split into 3 contiguous
latitude bands (Manhattan_S / Manhattan_M / Manhattan_N) to balance region
sizes (imbalance 6.1x -> 2.2x). Set _SPLIT_MANHATTAN = False to keep it whole.

Approx trajectory counts (train+val+test): Manhattan_S ~1810, Outer ~1800,
Manhattan_M ~1720, Manhattan_N ~1670, New_Jersey ~1110, Brooklyn ~750.

Outputs mirror the demo-group world (04_build_atlas_world_demogroups.py) so the
result is drop-in compatible with the downstream marginals / conditioning /
length-distribution pipeline:
  - world_{train,val,test}_georegions/<region>/{generated_sequences,
      all_attr_results.demographics, selected_indices, original_split_indices}.npy
  - world_{train,val,test}_georegions/{region_summary.csv, cbgs.txt, metadata.json}
  - length_dists_train_georegions.json
  - configs/{split}/{poi_marginals,cache_cbg_conditionals}_atlas_world.yaml
  - aggregates/{split}/, cache/{split}/

Usage:
    python foursquare_preprocessing/05_build_atlas_world_georegions.py \
        --split_data_root data/foursquare_nyc/controlled \
        --out_root atlas_world/foursquare_nyc_georegions
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAJGEN = REPO_ROOT / "trajectory-generation"
HELPERS = REPO_ROOT / "helpers"
BUILD_LENGTHS = HELPERS / "build_length_dists_from_atlas_world.py"
BUILD_MARGINALS = TRAJGEN / "scripts" / "precompute" / "build_poi_marginals.py"
CACHE_CONDITIONALS = TRAJGEN / "scripts" / "precompute" / "cache_cbg_conditionals.py"

# Census block-group shapefiles used for true point-in-polygon region assignment.
SHP_NJ = REPO_ROOT / "census_shapefiles" / "2023" / "BG" / "tl_2023_34_bg.zip"  # New Jersey
SHP_NY = REPO_ROOT / "census_shapefiles" / "2023" / "BG" / "tl_2023_36_bg.zip"  # New York
# NY county FIPS -> georegion (Queens/Bronx/Staten Island merged into "Outer").
_COUNTY_TO_REGION = {"061": "Manhattan", "047": "Brooklyn", "081": "Outer", "005": "Outer", "085": "Outer"}

# Manhattan alone holds ~58% of trajectories, so it is split into 3 contiguous
# latitude bands to balance the regions (imbalance 6.1x -> 2.2x). Cut points are
# the train most-visited-POI latitude terciles; bands map to real geography:
#   Manhattan_S  <= 40.7438   below ~23rd St (Downtown / Lower Manhattan)
#   Manhattan_M  <= 40.7582   ~23rd-50th St (Midtown core)
#   Manhattan_N   > 40.7582   above ~50th St (Midtown North + Upper Manhattan)
_MANHATTAN_LAT_CUTS = (40.7438, 40.7582)
_SPLIT_MANHATTAN = True


# --------------------------------------------------------------------------
# Region assignment
# --------------------------------------------------------------------------
# Approximate Hudson-River shoreline: longitude of Manhattan's west edge as a
# function of latitude. A point that is west of this line and north of Staten
# Island (lat >= 40.65) is New Jersey (Hoboken / Jersey City / Newark / etc.).
_HUDSON_LAT = np.array([40.68, 40.70, 40.74, 40.76, 40.80, 40.83, 40.86, 40.90])
_HUDSON_LON = np.array([-74.025, -74.021, -74.013, -74.010, -73.972, -73.945, -73.930, -73.918])


def _hudson_lon(lat: float) -> float:
    return float(np.interp(lat, _HUDSON_LAT, _HUDSON_LON))


def _borough6(lat: float, lon: float) -> str:
    """Classify a POI coordinate into an NYC borough, or New Jersey (approx)."""
    # Staten Island first (SW island, a real NYC borough)
    if lon < -74.05 and lat < 40.65:
        return "Staten Island"
    # New Jersey: west of the Hudson, opposite Manhattan
    if lat >= 40.65 and lon < _hudson_lon(lat):
        return "New Jersey"
    if lat >= 40.80 and lon >= -73.933:
        return "Bronx"
    if -74.03 <= lon <= -73.907 and 40.70 <= lat <= 40.882:
        return "Manhattan"
    if lat < 40.739 and lon >= -74.05:
        return "Brooklyn"
    if lon >= -73.86:
        return "Queens"
    if lat >= 40.739 and lon >= -73.907:
        return "Queens"
    if lat < 40.70:
        return "Brooklyn"
    return "Manhattan"


def _georegion(lat: float, lon: float) -> str:
    """Approximate 4-region grouping from lat/lon (fallback only). Prefer
    `_build_poi_region_map`, which uses true county/state polygons."""
    b = _borough6(lat, lon)
    if b in ("Manhattan", "Brooklyn"):
        return b
    if b == "New Jersey":
        return "New_Jersey"
    return "Outer"  # Queens + Bronx + Staten Island


def _build_poi_region_map(poi_ids, lat, lon) -> Dict[str, str]:
    """Map each POI id -> georegion by point-in-polygon against real
    county/state boundaries, snapping water/pier points to the nearest region.
    Falls back to the approximate lat/lon `_georegion` if geopandas or the
    shapefiles are unavailable."""
    try:
        import geopandas as gpd  # noqa: WPS433
        from shapely.geometry import Point  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] geopandas unavailable ({exc}); using approximate lat/lon regions.")
        return {p: _georegion(lat[p], lon[p]) for p in poi_ids}
    if not SHP_NJ.exists() or not SHP_NY.exists():
        print(f"[WARN] boundary shapefiles not found; using approximate lat/lon regions.")
        return {p: _georegion(lat[p], lon[p]) for p in poi_ids}

    nj = gpd.read_file(SHP_NJ)[["geometry"]].to_crs(4326)
    nj["region"] = "New_Jersey"
    ny = gpd.read_file(SHP_NY)[["COUNTYFP", "geometry"]].to_crs(4326)
    ny = ny[ny["COUNTYFP"].isin(_COUNTY_TO_REGION)].copy()
    ny["region"] = ny["COUNTYFP"].map(_COUNTY_TO_REGION)
    polys = gpd.GeoDataFrame(
        pd.concat([nj[["region", "geometry"]], ny[["region", "geometry"]]], ignore_index=True),
        crs=4326,
    ).dissolve(by="region").reset_index()

    pts = gpd.GeoDataFrame(
        {"poi": list(poi_ids)},
        geometry=[Point(lon[p], lat[p]) for p in poi_ids],
        crs=4326,
    )
    joined = gpd.sjoin(pts, polys, how="left", predicate="within").drop_duplicates("poi")
    region = dict(zip(joined["poi"], joined["region"]))

    missing = [p for p in poi_ids if not isinstance(region.get(p), str)]
    if missing:  # water / outside every polygon -> nearest region (projected CRS)
        pj = pts[pts["poi"].isin(missing)].to_crs(3857)
        nn = gpd.sjoin_nearest(pj, polys.to_crs(3857), how="left").drop_duplicates("poi")
        for p, r in zip(nn["poi"], nn["region"]):
            region[p] = r
    n_water = len(missing)
    print(f"  region map: {len(poi_ids)} POIs "
          f"({len(poi_ids) - n_water} in-polygon, {n_water} snapped-to-nearest)")

    if _SPLIT_MANHATTAN:
        c1, c2 = _MANHATTAN_LAT_CUTS
        for p in poi_ids:
            if region.get(p) == "Manhattan":
                region[p] = ("Manhattan_S" if lat[p] <= c1
                             else "Manhattan_M" if lat[p] <= c2
                             else "Manhattan_N")
    return region


def _load_poi_coords(split_root: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    frames = []
    for split in ("train", "val", "test"):
        p = split_root / split / "poi_map_feature.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No poi_map_feature.csv under {split_root}")
    poi = pd.concat(frames).drop_duplicates("poi_id")
    lat = dict(zip(poi.poi_id.astype(str), poi.lat.astype(float)))
    lon = dict(zip(poi.poi_id.astype(str), poi.lon.astype(float)))
    return lat, lon


# --------------------------------------------------------------------------
# Vocab / sequence helpers (mirror 05_build_atlas_world_georegions.py)
# --------------------------------------------------------------------------
def _read_vocab(vocab_path: Path) -> Tuple[Dict[str, int], int]:
    lines = vocab_path.read_text(encoding="utf-8").splitlines()
    vocab = {tok: i for i, tok in enumerate(lines) if tok}
    unk_id = vocab.get("[UNK]")
    if unk_id is None:
        raise ValueError("Could not find [UNK] in vocab.")
    return vocab, int(unk_id)


def _parse_sequence(value) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        import ast

        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return [str(x) for x in ast.literal_eval(s)]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _tokens_to_ids(tokens: List[str], vocab: Dict[str, int], unk_id: int) -> np.ndarray:
    return np.array([vocab.get(tok, unk_id) for tok in tokens], dtype=np.int64)


def _anchor_region(tokens: List[str], lat: Dict[str, float], lon: Dict[str, float]) -> str:
    """Region of the trajectory's most-visited real POI."""
    real = [t for t in tokens if t in lat]
    if not real:
        return "Manhattan"  # fallback; trajectories with no mapped POI are rare
    modal = Counter(real).most_common(1)[0][0]
    return _georegion(lat[modal], lon[modal])


def _write_region(
    out_dir: Path,
    region_id: str,
    indices: np.ndarray,
    sequences_ids: List[np.ndarray],
    attrs: np.ndarray,
    original_indices: np.ndarray,
    anchor_lat: np.ndarray,
    anchor_lon: np.ndarray,
) -> Dict:
    region_path = out_dir / region_id
    region_path.mkdir(parents=True, exist_ok=True)

    seqs = [sequences_ids[i] for i in indices.tolist()]
    np.save(region_path / "generated_sequences.npy", np.array(seqs, dtype=object), allow_pickle=True)
    np.save(region_path / "all_attr_results.demographics.npy", attrs[indices].astype(np.float32, copy=False))
    np.save(region_path / "selected_indices.npy", np.arange(len(seqs), dtype=np.int64))
    np.save(region_path / "original_split_indices.npy", original_indices[indices].astype(np.int64, copy=False))

    region_attrs = attrs[indices]
    age = region_attrs[:, 4].astype(int)
    gender = region_attrs[:, 5].astype(int)
    counts = pd.Series(list(zip(age.tolist(), gender.tolist()))).value_counts().sort_index()
    demo_json = {f"a{a}_g{g}": int(n) for (a, g), n in counts.items()}

    a_lat = anchor_lat[indices]
    a_lon = anchor_lon[indices]
    return {
        "region_id": region_id,
        "num_traj": len(seqs),
        "anchor_lat_min": float(np.min(a_lat)),
        "anchor_lat_max": float(np.max(a_lat)),
        "anchor_lon_min": float(np.min(a_lon)),
        "anchor_lon_max": float(np.max(a_lon)),
        "demo_counts": demo_json,
    }


# --------------------------------------------------------------------------
# YAML configs (identical shape to the demo builder)
# --------------------------------------------------------------------------
def _write_poi_marginals_yaml(path: Path, *, world_root: str, vocab_path: str, output_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Auto-generated for Foursquare NYC georegion ATLAS-world POI marginals
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
    print(f"  Wrote {path}")


def _write_cache_conditionals_yaml(path: Path, *, world_root: str, output_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Auto-generated for Foursquare NYC georegion ATLAS-world conditioning cache
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
    print(f"  Wrote {path}")


def _run(cmd: list, desc: str) -> None:
    print(f"\n{'=' * 60}\n  {desc}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {result.returncode}")
        sys.exit(1)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build NYC georegion ATLAS world for Foursquare")
    parser.add_argument("--split_data_root", default="data/foursquare_nyc/controlled")
    parser.add_argument("--out_root", default="atlas_world/foursquare_nyc_georegions")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_root = Path(args.split_data_root).resolve()
    out_root = Path(args.out_root).resolve()
    vocab_path = split_root / "tokenizer" / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    vocab, unk_id = _read_vocab(vocab_path)
    lat_map, lon_map = _load_poi_coords(split_root)
    print("Building POI -> georegion map from county/state polygons ...")
    poi_region = _build_poi_region_map(list(lat_map.keys()), lat_map, lon_map)
    out_root.mkdir(parents=True, exist_ok=True)

    all_regions = (["Manhattan_S", "Manhattan_M", "Manhattan_N"] if _SPLIT_MANHATTAN
                   else ["Manhattan"]) + ["Brooklyn", "Outer", "New_Jersey"]

    # Step 1: split into georegions
    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        pkl_path = split_dir / "final_segments_all_train_data.pkl"
        attrs_path = split_dir / "all_attr_results_with_demo.npy"
        if not pkl_path.exists() or not attrs_path.exists():
            print(f"[SKIP] {split} missing pkl or attrs")
            continue

        print(f"\n{'=' * 60}\n  Building NYC georegion world ({split})\n{'=' * 60}")
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        attrs = np.load(attrs_path).astype(np.float32, copy=False)

        sequences_ids: List[np.ndarray] = []
        regions = np.empty(len(df), dtype=object)
        anchor_lat = np.zeros(len(df), dtype=np.float32)
        anchor_lon = np.zeros(len(df), dtype=np.float32)
        for i in range(len(df)):
            toks = _parse_sequence(df.iloc[i]["unique_id_seq"])
            sequences_ids.append(_tokens_to_ids(toks, vocab, unk_id))
            real = [t for t in toks if t in lat_map]
            if real:
                modal = Counter(real).most_common(1)[0][0]
                anchor_lat[i] = lat_map[modal]
                anchor_lon[i] = lon_map[modal]
                regions[i] = poi_region.get(modal) or _georegion(lat_map[modal], lon_map[modal])
            else:
                regions[i] = all_regions[0]  # rare: no mapped POI in the sequence

        world_dir = out_root / f"world_{split}_georegions"
        world_dir.mkdir(parents=True, exist_ok=True)
        original_indices = np.arange(len(attrs), dtype=np.int64)

        summaries = []
        for region_id in all_regions:
            indices = np.where(regions == region_id)[0].astype(np.int64)
            if indices.size == 0:
                print(f"  {region_id}: 0 trajectories (skipped)")
                continue
            summary = _write_region(
                world_dir, region_id, indices, sequences_ids, attrs,
                original_indices, anchor_lat, anchor_lon,
            )
            summaries.append(summary)
            print(f"  {region_id}: {summary['num_traj']} trajectories")

        written_ids = [s["region_id"] for s in summaries]
        pd.DataFrame(summaries).to_csv(world_dir / "region_summary.csv", index=False)
        (world_dir / "cbgs.txt").write_text("\n".join(written_ids) + "\n", encoding="utf-8")
        meta = {
            "split_root": str(split_dir),
            "tokenizer_vocab": str(vocab_path),
            "out_root": str(world_dir),
            "region_mode": "nyc_georegion",
            "region_info": [
                "Georegions from most-visited POI, assigned by point-in-polygon "
                "against census county/state boundaries: Manhattan (NY 061), "
                "Brooklyn (NY 047), Outer (NY 081/005/085 = Queens+Bronx+Staten Island), "
                "New_Jersey (state 34). Manhattan is split into 3 latitude bands "
                "(S/M/N) to balance region sizes." if _SPLIT_MANHATTAN else
                "Georegions from most-visited POI via point-in-polygon: Manhattan, "
                "Brooklyn, Outer (Queens+Bronx+Staten Island), New_Jersey."
            ],
            "anchor": "most_visited_poi",
            "region_assignment": "point_in_polygon(census_bg_2023, state34+ny_counties)",
            "manhattan_split": (_MANHATTAN_LAT_CUTS if _SPLIT_MANHATTAN else None),
            "seed": args.seed,
            "input_rows": int(attrs.shape[0]),
            "kept_rows": int(attrs.shape[0]),
            "regions_written": len(written_ids),
        }
        with open(world_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Step 2: length distributions from train
    train_world = out_root / "world_train_georegions"
    length_json = out_root / "length_dists_train_georegions.json"
    if train_world.exists():
        _run(
            [sys.executable, str(BUILD_LENGTHS),
             "--world-root", str(train_world),
             "--out-json", str(length_json),
             "--max-length", str(args.max_length),
             "--num-special-tokens", "5",
             "--min-length", "1", "--clip"],
            desc="Build length distributions (train)",
        )

    # Step 3: YAML configs
    print(f"\n{'=' * 60}\n  Writing YAML configs\n{'=' * 60}")
    for split in ["train", "val", "test"]:
        world_dir = out_root / f"world_{split}_georegions"
        if not world_dir.exists():
            continue
        _write_poi_marginals_yaml(
            out_root / "configs" / split / "poi_marginals_atlas_world.yaml",
            world_root=str(world_dir), vocab_path=str(vocab_path),
            output_dir=str(out_root / "aggregates" / split),
        )
        _write_cache_conditionals_yaml(
            out_root / "configs" / split / "cache_cbg_conditionals_atlas_world.yaml",
            world_root=str(world_dir), output_dir=str(out_root / "cache" / split),
        )

    # Step 4: aggregation + caching
    for split in ["train", "val", "test"]:
        cfg_dir = out_root / "configs" / split
        marginals_cfg = cfg_dir / "poi_marginals_atlas_world.yaml"
        cache_cfg = cfg_dir / "cache_cbg_conditionals_atlas_world.yaml"
        if marginals_cfg.exists():
            _run([sys.executable, str(BUILD_MARGINALS), "--config", str(marginals_cfg)],
                 desc=f"Build POI marginals ({split})")
        if cache_cfg.exists():
            _run([sys.executable, str(CACHE_CONDITIONALS), "--config", str(cache_cfg)],
                 desc=f"Cache CBG conditionals ({split})")

    print(f"\n{'=' * 60}\n  LLP World NYC Georegion Summary\n{'=' * 60}")
    print(f"Output: {out_root}")
    for split in ["train", "val", "test"]:
        f = out_root / f"world_{split}_georegions" / "cbgs.txt"
        if f.exists():
            print(f"  {split}: {f.read_text().split()}")


if __name__ == "__main__":
    main()
