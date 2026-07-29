#!/usr/bin/env python3
"""
Phase 0: Extract & Link Foursquare TIST2015 checkins with UbiComp2016 user profiles.

Steps:
  0a. Load TIST2015 checkins + POIs (separate files), join to get lat/lon/category
  0b. Load UbiComp2016 profiles (gender)
  0c. Inner-join on user_id, filter to city bounding box, filter min checkins
  0d. Detect home/work from "Home (private)" and "Office" venue categories
  0e. Build poi_map_feature.csv

Outputs (written to --output_dir):
  linked_checkins.csv    — all checkins for qualifying users, sorted by (user, time)
  user_home_work.csv     — per-user home/work lat/lon
  poi_map_feature.csv    — per-venue metadata (lat, lon, category)
  user_profiles.csv      — per-user profile (gender, twitter stats)
  stats.json             — summary statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


CITY_BOUNDS = {
    "NYC": (40.49, 40.92, -74.27, -73.68),
    "TKY": (35.45, 35.90, 139.45, 139.95),
    "TOKYO": (35.45, 35.90, 139.45, 139.95),
}


def load_tist2015_pois(path: str) -> Dict[str, Tuple[float, float, str]]:
    """Load TIST2015 POI file: venue_id -> (lat, lon, category)."""
    pois = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                vid = parts[0]
                try:
                    lat, lon = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                cat = parts[3]
                pois[vid] = (lat, lon, cat)
    return pois


def load_tist2015_checkins(
    path: str,
    keep_users: set,
    pois: Dict[str, Tuple[float, float, str]],
    bounds: Tuple[float, float, float, float] | None = None,
    city_label: str = "city",
) -> pd.DataFrame:
    """
    Load TIST2015 checkins, join with POIs, filter to keep_users and optionally a city area.
    TIST2015 columns: user_id, venue_id, utc_time, tz_offset_min
    """
    lat_min = lat_max = lon_min = lon_max = None
    if bounds is not None:
        lat_min, lat_max, lon_min, lon_max = bounds

    rows = []
    n_scanned = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            n_scanned += 1
            uid, vid = parts[0], parts[1]
            if uid not in keep_users:
                continue
            if vid not in pois:
                continue
            lat, lon, cat = pois[vid]
            if bounds is not None and not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
            rows.append({
                "user_id": uid,
                "venue_id": vid,
                "lat": lat,
                "lon": lon,
                "venue_cat_name": cat,
                "venue_cat_id": vid,
                "utc_time_raw": parts[2],
                "tz_offset_min": parts[3],
            })
            if len(rows) % 100000 == 0:
                print(f"  collected {len(rows)} {city_label} checkins ({n_scanned / 1e6:.1f}M scanned)...")

    print(f"  scanned {n_scanned} total, collected {len(rows)} checkins")
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["utc_time"] = pd.to_datetime(df["utc_time_raw"], format="mixed", utc=True)
        df.drop(columns=["utc_time_raw"], inplace=True)
    return df


def load_ubicomp2016(path: str) -> pd.DataFrame:
    """Load UbiComp2016 user profile TSV (no header)."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "gender", "twitter_friends", "twitter_followers"],
        dtype={"user_id": str, "gender": str},
    )
    return df


def detect_home_work(
    checkins: pd.DataFrame,
    home_category: str = "Home (private)",
    work_category: str = "Office",
    fallback: bool = True,
) -> pd.DataFrame:
    """
    Detect home/work coordinates per user.
    Primary: most-visited venue with category == home_category / work_category.
    Fallback: most-visited / 2nd-most-visited venue overall.
    """
    records = []
    for uid, grp in checkins.groupby("user_id"):
        home_lat, home_lon = np.nan, np.nan
        work_lat, work_lon = np.nan, np.nan

        # Try explicit home category
        home_visits = grp[grp["venue_cat_name"] == home_category]
        if len(home_visits) > 0:
            top_home = (
                home_visits.groupby("venue_id")
                .agg(cnt=("venue_id", "count"), lat=("lat", "first"), lon=("lon", "first"))
                .sort_values("cnt", ascending=False)
                .iloc[0]
            )
            home_lat, home_lon = top_home["lat"], top_home["lon"]

        # Try explicit work category
        work_visits = grp[grp["venue_cat_name"] == work_category]
        if len(work_visits) > 0:
            top_work = (
                work_visits.groupby("venue_id")
                .agg(cnt=("venue_id", "count"), lat=("lat", "first"), lon=("lon", "first"))
                .sort_values("cnt", ascending=False)
                .iloc[0]
            )
            work_lat, work_lon = top_work["lat"], top_work["lon"]

        # Fallback: most/2nd-most visited venue
        if fallback and (np.isnan(home_lat) or np.isnan(work_lat)):
            venue_counts = (
                grp.groupby("venue_id")
                .agg(cnt=("venue_id", "count"), lat=("lat", "first"), lon=("lon", "first"))
                .sort_values("cnt", ascending=False)
            )
            if np.isnan(home_lat) and len(venue_counts) >= 1:
                home_lat = venue_counts.iloc[0]["lat"]
                home_lon = venue_counts.iloc[0]["lon"]
            if np.isnan(work_lat) and len(venue_counts) >= 2:
                work_lat = venue_counts.iloc[1]["lat"]
                work_lon = venue_counts.iloc[1]["lon"]

        records.append({
            "user_id": uid,
            "home_lat": home_lat,
            "home_lon": home_lon,
            "work_lat": work_lat,
            "work_lon": work_lon,
            "has_explicit_home": len(home_visits) > 0,
            "has_explicit_work": len(work_visits) > 0,
        })

    return pd.DataFrame(records)


def build_poi_map(checkins: pd.DataFrame) -> pd.DataFrame:
    """Build poi_map_feature.csv: most frequent (lat,lon,category) per venue_id."""
    grouped = (
        checkins.groupby(["venue_id", "lat", "lon", "venue_cat_name"])
        .size()
        .reset_index(name="cnt")
    )
    idx = grouped.groupby("venue_id")["cnt"].idxmax()
    poi_map = grouped.loc[idx, ["venue_id", "lat", "lon", "venue_cat_name"]].copy()
    poi_map.columns = ["poi_id", "lat", "lon", "top_category"]
    poi_map["sub_category"] = poi_map["top_category"]
    poi_map["placekey"] = None
    poi_map = poi_map.sort_values("poi_id").reset_index(drop=True)
    return poi_map


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Extract & link TIST2015 + UbiComp2016")
    parser.add_argument("--tist2015_checkins", required=True,
                        help="Path to dataset_TIST2015_Checkins.txt")
    parser.add_argument("--tist2015_pois", required=True,
                        help="Path to dataset_TIST2015_POIs.txt")
    parser.add_argument("--ubicomp2016", required=True,
                        help="Path to dataset_UbiComp2016_UserProfile_NYC.txt")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--city", default="NYC",
                        help="City label/bounding box to use. Built-ins: NYC, TKY/TOKYO")
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--home_category", default="Home (private)")
    parser.add_argument("--work_category", default="Office")
    parser.add_argument("--fallback", action="store_true", default=True)
    parser.add_argument("--min_checkins", type=int, default=10,
                        help="Minimum NYC checkins per user to include")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    city = args.city.strip().upper()
    custom_bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if any(v is not None for v in custom_bounds):
        if not all(v is not None for v in custom_bounds):
            raise ValueError("Provide all of --lat_min, --lat_max, --lon_min, --lon_max for custom bounds")
        bounds = custom_bounds
    else:
        if city not in CITY_BOUNDS:
            raise ValueError(f"Unknown city {args.city!r}. Use one of {sorted(CITY_BOUNDS)} or provide custom bounds.")
        bounds = CITY_BOUNDS[city]

    # Load profiles first to know which users to keep
    print("Loading UbiComp2016 profiles...")
    profiles = load_ubicomp2016(args.ubicomp2016)
    profile_uids = set(profiles["user_id"].unique())
    print(f"  {len(profile_uids)} profiles")

    # Load POIs
    print("Loading TIST2015 POIs...")
    pois = load_tist2015_pois(args.tist2015_pois)
    print(f"  {len(pois)} POIs loaded")

    # Load and filter checkins
    print(f"Loading TIST2015 checkins (filtering to {city} + profile users)...")
    checkins = load_tist2015_checkins(
        args.tist2015_checkins,
        keep_users=profile_uids,
        pois=pois,
        bounds=bounds,
        city_label=city,
    )
    print(f"  {len(checkins)} {city} checkins from profile users")

    # Filter by min checkins
    user_counts = checkins.groupby("user_id").size()
    qualifying_users = set(user_counts[user_counts >= args.min_checkins].index)
    checkins = checkins[checkins["user_id"].isin(qualifying_users)].copy()
    checkins = checkins.sort_values(["user_id", "utc_time"]).reset_index(drop=True)
    print(f"  After min_checkins={args.min_checkins} filter: {len(qualifying_users)} users, {len(checkins)} checkins")

    overlap_profiles = profiles[profiles["user_id"].isin(qualifying_users)].copy()

    # Detect home/work
    print("Detecting home/work locations...")
    home_work = detect_home_work(
        checkins,
        home_category=args.home_category,
        work_category=args.work_category,
        fallback=args.fallback,
    )
    n_home = home_work["home_lat"].notna().sum()
    n_work = home_work["work_lat"].notna().sum()
    n_explicit_home = home_work["has_explicit_home"].sum()
    n_explicit_work = home_work["has_explicit_work"].sum()
    print(f"  Users with home: {n_home}/{len(home_work)} (explicit: {n_explicit_home})")
    print(f"  Users with work: {n_work}/{len(home_work)} (explicit: {n_explicit_work})")

    # Build POI map
    print("Building POI map...")
    poi_map = build_poi_map(checkins)
    print(f"  {len(poi_map)} unique venues")

    # Save outputs
    print("Saving outputs...")
    checkins.to_csv(out / "linked_checkins.csv", index=False)
    home_work.to_csv(out / "user_home_work.csv", index=False)
    poi_map.to_csv(out / "poi_map_feature.csv", index=False)
    overlap_profiles.to_csv(out / "user_profiles.csv", index=False)

    # Per-user stats
    user_stats = checkins.groupby("user_id").agg(
        checkins=("venue_id", "count"),
        unique_venues=("venue_id", "nunique"),
        unique_categories=("venue_cat_name", "nunique"),
    )

    stats = {
        "n_users": len(qualifying_users),
        "n_checkins": len(checkins),
        "n_venues": checkins["venue_id"].nunique(),
        "n_categories": checkins["venue_cat_name"].nunique(),
        "n_users_with_home": int(n_home),
        "n_users_with_work": int(n_work),
        "n_users_with_explicit_home": int(n_explicit_home),
        "n_users_with_explicit_work": int(n_explicit_work),
        "gender_distribution": overlap_profiles["gender"].value_counts().to_dict(),
        "checkins_per_user": {
            "min": int(user_stats["checkins"].min()),
            "max": int(user_stats["checkins"].max()),
            "median": float(user_stats["checkins"].median()),
            "mean": float(user_stats["checkins"].mean()),
        },
        "min_checkins_threshold": args.min_checkins,
        "city": city,
        "bounds": {
            "lat_min": bounds[0],
            "lat_max": bounds[1],
            "lon_min": bounds[2],
            "lon_max": bounds[3],
        },
    }

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. Outputs written to {out}/")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
