#!/usr/bin/env python3
"""
Step 2: Age pseudo-label inference for Foursquare NYC users.

The Foursquare user-profile release provides self-reported *gender* but no age.
To complete the K=8 (4 age bins x 2 genders) demographic groups we therefore
*infer* an ordinal age label per user. To avoid circularity with the
spatial/sequential trajectory model, inference reads only aggregated,
coarse-grained behavioral summaries of a user's check-in history -- never the
raw trajectories.

Two providers are available:

  rule_based (default, used for all reported results)
      A scalar "age score" from the aggregated features, exactly as described in
      the paper appendix (Foursquare NYC Dataset / Demographic labeling):
      nightlife categories add s_c, fitness categories add 0.5*s_c, medical and
      religious categories subtract s_c; a nighttime check-in share above 0.15
      adds 0.2 and below 0.05 subtracts 0.1; venue diversity above 0.5 adds 0.1
      and below 0.3 subtracts 0.1. See `rule_based_age_score`.

  openai
      An optional LLM provider (OpenAI-compatible endpoint) kept for reference.
      It is NOT used for any result reported in the paper.

Note that this step emits a *provisional* age bin from fixed score thresholds.
The bin assignment used in the paper is produced downstream by
`03_build_reduced_vocab_split.py --age-mode balanced_segments`, which re-sorts
users by the same score and cuts the four bins so each accumulates roughly a
quarter of the trajectory segments.

These are pseudo-labels encoding a *relative* behavioral ordering, not measured
ages. See the paper appendix for the interpretation caveat.

Inputs:
  linked_checkins.csv   — from step 1
  user_profiles.csv     — from step 1 (gender)
  user_home_work.csv    — from step 1 (home/work coords)

Outputs:
  demo.csv              — panelist_id, home_lat, home_lon, work_lat, work_lon, age, gender
  user_features.csv     — per-user aggregated features (consumed by step 3)
  llm_responses.json    — raw responses, only written for the openai provider
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from age_rule import rule_based_age_bin  # noqa: E402


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_user_features(checkins: pd.DataFrame, home_work: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregated mobility features."""
    records = []
    hw = home_work.set_index("user_id")

    for uid, grp in checkins.groupby("user_id"):
        n_checkins = len(grp)
        n_unique_venues = grp["venue_id"].nunique()
        n_unique_cats = grp["venue_cat_name"].nunique()

        # Category distribution (top 10)
        cat_counts = grp["venue_cat_name"].value_counts(normalize=True)
        top_cats = cat_counts.head(10).to_dict()

        # Temporal patterns
        hours = grp["utc_time"].dt.hour
        # Time-of-day buckets: morning(6-11), afternoon(12-17), evening(18-23), night(0-5)
        morning_pct = ((hours >= 6) & (hours < 12)).mean()
        afternoon_pct = ((hours >= 12) & (hours < 18)).mean()
        evening_pct = ((hours >= 18) & (hours < 24)).mean()
        night_pct = (hours < 6).mean()

        # Weekday vs weekend
        weekday_pct = (grp["utc_time"].dt.dayofweek < 5).mean()

        # Mobility radius from home
        if uid in hw.index:
            hlat, hlon = hw.loc[uid, "home_lat"], hw.loc[uid, "home_lon"]
            if not (np.isnan(hlat) or np.isnan(hlon)):
                dists = grp.apply(lambda r: haversine_km(hlat, hlon, r["lat"], r["lon"]), axis=1)
                radius_km = dists.max()
                median_dist_km = dists.median()
            else:
                radius_km = np.nan
                median_dist_km = np.nan
        else:
            radius_km = np.nan
            median_dist_km = np.nan

        # Venue diversity
        diversity = n_unique_venues / n_checkins

        records.append({
            "user_id": uid,
            "n_checkins": n_checkins,
            "n_unique_venues": n_unique_venues,
            "n_unique_categories": n_unique_cats,
            "diversity": round(diversity, 3),
            "radius_km": round(radius_km, 1) if not np.isnan(radius_km) else None,
            "median_dist_km": round(median_dist_km, 1) if not np.isnan(median_dist_km) else None,
            "morning_pct": round(morning_pct, 3),
            "afternoon_pct": round(afternoon_pct, 3),
            "evening_pct": round(evening_pct, 3),
            "night_pct": round(night_pct, 3),
            "weekday_pct": round(weekday_pct, 3),
            "top_categories": json.dumps(
                {k: round(v, 3) for k, v in list(top_cats.items())[:10]}
            ),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

def build_age_prompt(user_feat: dict, gender: str) -> str:
    """Build a prompt for age classification from aggregated features."""
    top_cats = json.loads(user_feat["top_categories"])
    cat_str = ", ".join(f"{cat} ({pct*100:.1f}%)" for cat, pct in top_cats.items())

    prompt = f"""You are classifying a Foursquare user's age group based on their mobility patterns.

User profile:
- Gender: {gender}
- Total check-ins: {user_feat['n_checkins']}
- Unique venues visited: {user_feat['n_unique_venues']}
- Unique venue categories: {user_feat['n_unique_categories']}
- Venue diversity (unique/total): {user_feat['diversity']}
- Mobility radius from home: {user_feat['radius_km']} km
- Median distance from home: {user_feat['median_dist_km']} km
- Time-of-day distribution: morning {user_feat['morning_pct']*100:.1f}%, afternoon {user_feat['afternoon_pct']*100:.1f}%, evening {user_feat['evening_pct']*100:.1f}%, night {user_feat['night_pct']*100:.1f}%
- Weekday check-in ratio: {user_feat['weekday_pct']*100:.1f}%
- Top venue categories: {cat_str}

Based on these mobility patterns, classify this user into one of these age groups:
- 0: Under 30 (young adult)
- 1: 30-40 (early career)
- 2: 40-50 (mid career)
- 3: 50+ (senior)

Consider: younger users tend to visit bars/nightlife more, have more evening/night activity, and visit more diverse venues. Older users tend to have more routine patterns and visit offices/medical/residential venues more.

Return ONLY valid JSON: {{"age_bin": <0-3>, "reasoning": "<brief explanation>"}}"""
    return prompt


def query_llm_openai(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 256,
    timeout: int = 60,
) -> str:
    """Query OpenAI-compatible endpoint (vLLM, OpenAI, etc.)."""
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})


def parse_age_response(text: str) -> tuple[int, str]:
    """Parse LLM response, return (age_bin, reasoning)."""
    try:
        # Try direct JSON parse
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                return -1, f"parse_error: {text[:200]}"
        else:
            return -1, f"no_json: {text[:200]}"

    age_bin = obj.get("age_bin", -1)
    if isinstance(age_bin, str):
        try:
            age_bin = int(age_bin)
        except ValueError:
            age_bin = -1
    if age_bin not in (0, 1, 2, 3):
        age_bin = -1
    reasoning = obj.get("reasoning", "")
    return age_bin, reasoning


def rule_based_age(user_feat: dict) -> int:
    """Provisional rule-based age bin (see `age_rule` for the scoring rule)."""
    return rule_based_age_bin(user_feat)


# Age bin → representative age for demo.csv
AGE_BIN_TO_AGE = {0: 25, 1: 35, 2: 45, 3: 55, -1: 35}
GENDER_MAP = {"male": "MALE", "female": "FEMALE"}


def main():
    parser = argparse.ArgumentParser(description="Phase 1: LLM age pseudo-labeling")
    parser.add_argument("--input_dir", default="outputs", help="Directory with Phase 0 outputs")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    # Age inference provider. rule_based is the default and is what produced
    # every age pseudo-label reported in the paper; openai is optional.
    parser.add_argument("--provider", default="rule_based", choices=["openai", "rule_based"],
                        help="rule_based (default, used for all reported results) or an OpenAI-compatible LLM")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="Only needed for --provider openai. Defaults to $OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Phase 0 outputs...")
    checkins = pd.read_csv(inp / "linked_checkins.csv", parse_dates=["utc_time"])
    profiles = pd.read_csv(inp / "user_profiles.csv", dtype={"user_id": str})
    home_work = pd.read_csv(inp / "user_home_work.csv", dtype={"user_id": str})

    print("Building per-user features...")
    features = build_user_features(checkins, home_work)
    features.to_csv(out / "user_features.csv", index=False)
    print(f"  {len(features)} user feature profiles built")

    # Merge gender
    gender_map = profiles.set_index("user_id")["gender"].to_dict()

    # Assign age bins
    print(f"Assigning age bins via {args.provider}...")
    llm_responses = []
    age_results = []

    for _, row in features.iterrows():
        uid = row["user_id"]
        gender = gender_map.get(str(uid), "unknown")
        feat = row.to_dict()

        if args.provider == "openai":
            prompt = build_age_prompt(feat, gender)
            response = query_llm_openai(
                prompt,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            age_bin, reasoning = parse_age_response(response)
            if age_bin == -1:
                age_bin = rule_based_age(feat)
                reasoning = f"llm_fallback: {reasoning}"
            llm_responses.append({
                "user_id": uid,
                "response": response,
                "age_bin": age_bin,
                "reasoning": reasoning,
            })
        else:
            age_bin = rule_based_age(feat)
            reasoning = "rule_based"
            llm_responses.append({
                "user_id": uid,
                "age_bin": age_bin,
                "reasoning": reasoning,
            })

        age_results.append({"user_id": uid, "age_bin": age_bin})

    # Save LLM responses
    with open(out / "llm_responses.json", "w") as f:
        json.dump(llm_responses, f, indent=2)

    # Build demo.csv
    age_df = pd.DataFrame(age_results)
    demo = home_work[["user_id", "home_lat", "home_lon", "work_lat", "work_lon"]].copy()
    demo = demo.rename(columns={"user_id": "panelist_id"})
    demo["panelist_id"] = demo["panelist_id"].astype(str)
    age_df = age_df.rename(columns={"user_id": "panelist_id"})
    age_df["panelist_id"] = age_df["panelist_id"].astype(str)
    demo = demo.merge(age_df, on="panelist_id", how="left")
    demo["age"] = demo["age_bin"].map(AGE_BIN_TO_AGE)
    demo["gender"] = demo["panelist_id"].map(
        {str(k): GENDER_MAP.get(v, v) for k, v in gender_map.items()}
    )
    demo = demo[["panelist_id", "home_lat", "home_lon", "work_lat", "work_lon", "age", "gender"]]
    demo.to_csv(out / "demo.csv", index=False)

    print(f"\nAge bin distribution:")
    for b in range(4):
        n = (age_df["age_bin"] == b).sum()
        label = {0: "<30", 1: "30-40", 2: "40-50", 3: "50+"}[b]
        print(f"  {label}: {n}")

    print(f"\nGender distribution:")
    print(demo["gender"].value_counts().to_string())

    print(f"\nOutputs written to {out}/")
    print(f"  demo.csv: {len(demo)} users")


if __name__ == "__main__":
    main()
