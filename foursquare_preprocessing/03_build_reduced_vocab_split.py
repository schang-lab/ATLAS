#!/usr/bin/env python3
"""
Build a reduced-vocabulary Foursquare split from Phase 0/1 outputs.

This variant:
  - can infer home/work from explicit categories only, with no frequency fallback;
  - can replace per-user explicit home/work venues with POI_HOME / POI_WORK;
  - can alternatively keep all visits as raw POI tokens and omit attrs;
  - keeps the top-K tokens by frequency and drops all other POI visits;
  - re-segments each user's remaining trajectory into chunks of max_len - 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from age_rule import rule_based_age_score  # noqa: E402


BERT_SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def age_to_bin4(age: float) -> int:
    if pd.isna(age):
        return -1
    age = float(age)
    if age < 30:
        return 0
    if age < 40:
        return 1
    if age < 50:
        return 2
    return 3


def gender_to_id(gender: str) -> int:
    if pd.isna(gender):
        return -1
    gender = str(gender).strip().upper()
    if gender == "FEMALE":
        return 0
    if gender == "MALE":
        return 1
    return -1


def rule_age_score_from_features(user_feat: dict) -> float:
    """Age score behind the `balanced_segments` bin assignment (see `age_rule`)."""
    return rule_based_age_score(user_feat)


def load_rule_age_scores(input_dir: Path) -> pd.DataFrame:
    features_path = input_dir / "user_features.csv"
    if not features_path.exists():
        return pd.DataFrame(columns=["panelist_id", "age_score"])
    features = pd.read_csv(features_path, dtype={"user_id": str})
    features["age_score"] = features.apply(lambda row: rule_age_score_from_features(row.to_dict()), axis=1)
    return features[["user_id", "age_score"]].rename(columns={"user_id": "panelist_id"})


def assign_balanced_age_bins(
    demo: pd.DataFrame,
    segment_weights: pd.Series,
    input_dir: Path,
) -> pd.Series:
    scores = load_rule_age_scores(input_dir)
    work = demo[["panelist_id"]].copy()
    work["panelist_id"] = work["panelist_id"].astype(str)
    work = work.merge(scores, on="panelist_id", how="left")
    work["age_score"] = work["age_score"].fillna(0.0)
    work["segment_weight"] = work["panelist_id"].map(segment_weights).fillna(0).astype(int)
    work = work.sort_values(["age_score", "panelist_id"], ascending=[False, True]).reset_index(drop=True)

    positive = work[work["segment_weight"] > 0].copy()
    total_weight = int(positive["segment_weight"].sum())
    if total_weight <= 0:
        return pd.Series(0, index=demo.index, dtype=int)

    target = total_weight / 4.0
    assignments: dict[str, int] = {}
    current_bin = 0
    current_weight = 0
    for _, row in positive.iterrows():
        if current_bin < 3 and current_weight >= target:
            current_bin += 1
            current_weight = 0
        user_id = str(row["panelist_id"])
        assignments[user_id] = current_bin
        current_weight += int(row["segment_weight"])

    # Zero-length users do not contribute segments; keep them ordered by score for completeness.
    for _, row in work[work["segment_weight"] <= 0].iterrows():
        assignments[str(row["panelist_id"])] = min(3, current_bin)

    return demo["panelist_id"].astype(str).map(assignments).fillna(0).astype(int)


def infer_explicit_home_work(
    checkins: pd.DataFrame,
    home_category: str,
    work_category: str,
) -> pd.DataFrame:
    records = []
    for user_id, group in checkins.groupby("user_id", sort=False):
        record = {
            "panelist_id": str(user_id),
            "home_venue_id": None,
            "home_lat": np.nan,
            "home_lon": np.nan,
            "work_venue_id": None,
            "work_lat": np.nan,
            "work_lon": np.nan,
            "has_explicit_home": False,
            "has_explicit_work": False,
        }

        home_visits = group[group["venue_cat_name"] == home_category]
        if not home_visits.empty:
            top_home = (
                home_visits.groupby("venue_id", sort=False)
                .agg(cnt=("venue_id", "count"), lat=("lat", "first"), lon=("lon", "first"))
                .sort_values(["cnt", "venue_id"], ascending=[False, True])
                .iloc[0]
            )
            record["home_venue_id"] = str(top_home.name)
            record["home_lat"] = float(top_home["lat"])
            record["home_lon"] = float(top_home["lon"])
            record["has_explicit_home"] = True

        work_visits = group[group["venue_cat_name"] == work_category]
        if not work_visits.empty:
            top_work = (
                work_visits.groupby("venue_id", sort=False)
                .agg(cnt=("venue_id", "count"), lat=("lat", "first"), lon=("lon", "first"))
                .sort_values(["cnt", "venue_id"], ascending=[False, True])
                .iloc[0]
            )
            record["work_venue_id"] = str(top_work.name)
            record["work_lat"] = float(top_work["lat"])
            record["work_lon"] = float(top_work["lon"])
            record["has_explicit_work"] = True

        records.append(record)
    return pd.DataFrame(records)


def split_users(demo: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    users = demo[["panelist_id", "gender"]].drop_duplicates("panelist_id").copy()
    users["gender"] = users["gender"].fillna("unknown").astype(str)
    rest_ratio = val_ratio + test_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=rest_ratio, random_state=seed)
    train_idx, rest_idx = next(sss1.split(users, users["gender"]))

    rest_users = users.iloc[rest_idx].copy()
    val_frac = val_ratio / rest_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_frac, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(rest_users, rest_users["gender"]))

    split_map = {}
    for uid in users.iloc[train_idx]["panelist_id"].astype(str):
        split_map[uid] = "train"
    for uid in rest_users.iloc[val_idx_rel]["panelist_id"].astype(str):
        split_map[uid] = "val"
    for uid in rest_users.iloc[test_idx_rel]["panelist_id"].astype(str):
        split_map[uid] = "test"
    return split_map


def choose_vocab(tokens: pd.Series, vocab_size: int, always_keep: List[str]) -> List[str]:
    counts = tokens.value_counts()
    selected = [token for token in always_keep if token in counts.index]
    selected_set = set(selected)
    for token in counts.index.astype(str):
        if len(selected) >= vocab_size:
            break
        if token in selected_set:
            continue
        selected.append(token)
        selected_set.add(token)
    return selected


def add_specials_and_pad(seq: List[str], max_len: int) -> Tuple[List[str], List[int]]:
    seq = seq[: max_len - 2]
    seq = ["[CLS]"] + seq + ["[SEP]"]
    if len(seq) < max_len:
        seq = seq + ["[PAD]"] * (max_len - len(seq))
    mask = [0 if token == "[PAD]" else 1 for token in seq]
    return seq, mask


def pad_time_arrays(arrival_ts_seqs: List[List], dwell_min_seqs: List[List], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.full((len(arrival_ts_seqs), max_len), np.datetime64("NaT"), dtype="datetime64[ns]")
    dwell_sec = np.zeros((len(arrival_ts_seqs), max_len), dtype=np.float32)

    for i, (ts_list, dm_list) in enumerate(zip(arrival_ts_seqs, dwell_min_seqs)):
        length = min(len(ts_list), max_len)
        if length == 0:
            continue
        timestamps[i, :length] = np.asarray(pd.to_datetime(ts_list[:length], errors="coerce"), dtype="datetime64[ns]")
        dwell_sec[i, :length] = pd.to_numeric(pd.Series(dm_list[:length]), errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        ) * 60.0
    return timestamps, dwell_sec


def build_tokenizer(vocab_tokens: List[str], output_dir: Path) -> None:
    from transformers import BertTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in BERT_SPECIALS + vocab_tokens:
            f.write(f"{token}\n")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.add_special_tokens(
        {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]",
        }
    )
    tokenizer.save_pretrained(str(output_dir))


def describe_lengths(lengths: pd.Series) -> dict:
    if lengths.empty:
        return {}
    quantiles = lengths.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "n": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p10": float(quantiles.loc[0.1]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing linked_checkins.csv, demo.csv, user_profiles.csv")
    parser.add_argument("--output-root", required=True, help="Output root, e.g. split_data_Foursquare_TK_vocab5000/controlled")
    parser.add_argument("--city", default="TKY")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--home-category", default="Home (private)")
    parser.add_argument("--work-category", default="Office")
    parser.add_argument(
        "--raw-poi-tokens",
        action="store_true",
        help="Keep original venue IDs as tokens; do not replace explicit home/work venues with POI_HOME/POI_WORK.",
    )
    parser.add_argument(
        "--no-attrs",
        action="store_true",
        help="Do not write all_attr_results.npy or all_attr_results_with_demo.npy.",
    )
    parser.add_argument(
        "--attr-mode",
        choices=["full", "gender", "none"],
        default="full",
        help=(
            "Attribute output mode. full writes home/work coords + age/gender; "
            "gender writes zero coordinate placeholders + age/gender; "
            "none writes no attribute arrays."
        ),
    )
    parser.add_argument(
        "--age-mode",
        choices=["constant", "rule", "balanced_segments"],
        default="constant",
        help=(
            "Age labels stored in attrs. constant writes age_bin=0; rule uses the existing rule-based age; "
            "balanced_segments orders users by the rule score but assigns age bins to balance final segment counts."
        ),
    )
    args = parser.parse_args()
    attr_mode = "none" if args.no_attrs else args.attr_mode

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    max_content_len = int(args.max_len) - 2
    if max_content_len <= 0:
        raise ValueError("--max-len must be at least 3")

    # Step 1 writes linked_checkins.csv; accept a gzipped copy too (pandas reads either).
    checkins_path = input_dir / "linked_checkins.csv"
    if not checkins_path.exists():
        checkins_path = input_dir / "linked_checkins.csv.gz"
    checkins = pd.read_csv(
        checkins_path,
        parse_dates=["utc_time"],
        dtype={"user_id": str, "venue_id": str},
    )
    demo_source = pd.read_csv(input_dir / "demo.csv", dtype={"panelist_id": str})
    profiles = pd.read_csv(input_dir / "user_profiles.csv", dtype={"user_id": str})

    explicit_hw = infer_explicit_home_work(checkins, args.home_category, args.work_category)
    same_home_work = (
        explicit_hw["home_venue_id"].notna()
        & explicit_hw["work_venue_id"].notna()
        & (explicit_hw["home_venue_id"] == explicit_hw["work_venue_id"])
    )

    demo = explicit_hw.merge(
        demo_source[["panelist_id", "age", "gender"]],
        on="panelist_id",
        how="left",
    )
    if "gender" not in demo or demo["gender"].isna().any():
        gender_map = {str(row.user_id): str(row.gender).upper() for row in profiles.itertuples(index=False)}
        demo["gender"] = demo["gender"].fillna(demo["panelist_id"].map(gender_map))
    demo["gender_id"] = demo["gender"].apply(gender_to_id)

    split_map = split_users(demo, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    checkins["split"] = checkins["user_id"].map(split_map)

    hw_map = explicit_hw.set_index("panelist_id")[["home_venue_id", "work_venue_id"]]
    checkins = checkins.join(hw_map, on="user_id")
    checkins["token"] = checkins["venue_id"].astype(str)
    is_home = checkins["home_venue_id"].notna() & (checkins["venue_id"] == checkins["home_venue_id"])
    is_work = checkins["work_venue_id"].notna() & (checkins["venue_id"] == checkins["work_venue_id"])
    if not args.raw_poi_tokens:
        checkins.loc[is_work, "token"] = "POI_WORK"
        checkins.loc[is_home, "token"] = "POI_HOME"

    original_user_lengths = checkins.groupby("user_id").size()
    always_keep = [] if args.raw_poi_tokens else ["POI_HOME", "POI_WORK"]
    vocab_tokens = choose_vocab(checkins["token"], int(args.vocab_size), always_keep)
    vocab_set = set(vocab_tokens)

    filtered = checkins[checkins["token"].isin(vocab_set)].copy()
    filtered = filtered.sort_values(["user_id", "utc_time"]).reset_index(drop=True)
    pruned_lengths = filtered.groupby("user_id").size()
    all_user_ids = pd.Index(demo["panelist_id"].astype(str).unique())
    pruned_lengths_all = pruned_lengths.reindex(all_user_ids, fill_value=0)
    segment_weights = np.ceil(pruned_lengths_all / max_content_len).astype(int)

    if args.age_mode == "constant":
        demo["age_bin5"] = 0
    elif args.age_mode == "rule":
        demo["age_bin5"] = demo["age"].apply(age_to_bin4)
    else:
        demo["age_bin5"] = assign_balanced_age_bins(demo, segment_weights, input_dir)

    split_frames = {split: [] for split in ["train", "val", "test"]}
    split_attrs = {split: [] for split in ["train", "val", "test"]}
    split_attrs_demo = {split: [] for split in ["train", "val", "test"]}
    split_timestamps = {split: [] for split in ["train", "val", "test"]}
    split_dwell = {split: [] for split in ["train", "val", "test"]}
    split_lengths = {split: [] for split in ["train", "val", "test"]}

    demo_by_user = demo.set_index("panelist_id")
    for user_id, group in filtered.groupby("user_id", sort=False):
        split = split_map.get(str(user_id))
        if split not in split_frames:
            continue
        group = group.sort_values("utc_time")
        tokens = group["token"].astype(str).tolist()
        timestamps = group["utc_time"].tolist()
        dwell = [30.0] * len(tokens)
        user_demo = demo_by_user.loc[str(user_id)]
        attrs4 = [
            float(user_demo["work_lat"]) if pd.notna(user_demo["work_lat"]) else 0.0,
            float(user_demo["work_lon"]) if pd.notna(user_demo["work_lon"]) else 0.0,
            float(user_demo["home_lat"]) if pd.notna(user_demo["home_lat"]) else 0.0,
            float(user_demo["home_lon"]) if pd.notna(user_demo["home_lon"]) else 0.0,
        ]
        attrs6 = attrs4 + [float(user_demo["age_bin5"]), float(user_demo["gender_id"])]

        for seg_idx, start in enumerate(range(0, len(tokens), max_content_len)):
            content = tokens[start : start + max_content_len]
            ts_content = timestamps[start : start + max_content_len]
            dwell_content = dwell[start : start + max_content_len]
            if not content:
                continue
            padded_seq, attention_mask = add_specials_and_pad(content, int(args.max_len))
            split_frames[split].append(
                {
                    "segment_id": f"{user_id}_seg{seg_idx}",
                    "individual_id": str(user_id),
                    "city": str(args.city),
                    "unique_id_seq": padded_seq,
                    "attention_mask": attention_mask,
                }
            )
            split_attrs[split].append(attrs4)
            split_attrs_demo[split].append(attrs6)
            split_timestamps[split].append(ts_content)
            split_dwell[split].append(dwell_content)
            split_lengths[split].append(len(content))

    used_tokens = set()
    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(split_frames[split])
        out_df.to_pickle(split_dir / "final_segments_all_train_data.pkl")
        for seq in out_df.get("unique_id_seq", []):
            used_tokens.update(seq)

        if attr_mode != "none":
            attrs4_arr = np.asarray(split_attrs[split], dtype=np.float32).reshape((-1, 4))
            attrs6_arr = np.asarray(split_attrs_demo[split], dtype=np.float32).reshape((-1, 6))
            if attr_mode == "gender":
                attrs4_arr = np.zeros_like(attrs4_arr, dtype=np.float32)
                attrs6_arr[:, :4] = 0.0
            np.save(split_dir / "all_attr_results.npy", attrs4_arr)
            np.save(split_dir / "all_attr_results_with_demo.npy", attrs6_arr)
            np.save(split_dir / "all_attr_results.demographics.npy", attrs6_arr[:, 4:6].astype(np.float32, copy=False))
        np.save(split_dir / "trajectory_length_ids.npy", np.asarray(split_lengths[split], dtype=np.int64))
        timestamps_arr, dwell_arr = pad_time_arrays(split_timestamps[split], split_dwell[split], int(args.max_len))
        np.save(split_dir / "all_timestamp.npy", timestamps_arr)
        np.save(split_dir / "all_dwell.npy", dwell_arr)

    used_real_tokens = sorted((used_tokens - set(BERT_SPECIALS)), key=lambda t: vocab_tokens.index(t) if t in vocab_set else 10**9)
    build_tokenizer(used_real_tokens, output_root / "tokenizer")
    for split in ["train", "val", "test"]:
        build_tokenizer(used_real_tokens, output_root / split / "tokenizer")

    poi_meta = (
        filtered[filtered["token"].isin(used_real_tokens)]
        .groupby(["token", "lat", "lon", "venue_cat_name"], sort=False)
        .size()
        .reset_index(name="cnt")
    )
    if not poi_meta.empty:
        idx = poi_meta.groupby("token")["cnt"].idxmax()
        poi_map = poi_meta.loc[idx, ["token", "lat", "lon", "venue_cat_name"]].rename(
            columns={"token": "poi_id", "lat": "lat", "lon": "lon", "venue_cat_name": "top_category"}
        )
    else:
        poi_map = pd.DataFrame(columns=["poi_id", "lat", "lon", "top_category"])
    poi_map["sub_category"] = poi_map["top_category"]
    poi_map["placekey"] = None
    poi_map = poi_map.sort_values("poi_id").reset_index(drop=True)
    for split in ["train", "val", "test"]:
        poi_map.to_csv(output_root / split / "poi_map_feature.csv", index=False)

    stats = {
        "city": args.city,
        "raw_poi_tokens": bool(args.raw_poi_tokens),
        "attr_mode": attr_mode,
        "age_mode": args.age_mode,
        "attrs_written": attr_mode != "none",
        "vocab_size_requested": int(args.vocab_size),
        "data_vocab_size": len(used_real_tokens),
        "tokenizer_vocab_size": len(used_real_tokens) + len(BERT_SPECIALS),
        "original_users": int(len(all_user_ids)),
        "users_after_pruning": int((pruned_lengths_all > 0).sum()),
        "zero_length_users_after_pruning": int((pruned_lengths_all == 0).sum()),
        "original_checkins": int(len(checkins)),
        "retained_checkins": int(len(filtered)),
        "retained_checkin_ratio": float(len(filtered) / len(checkins)) if len(checkins) else 0.0,
        "unique_pois_before_replacement": int(checkins["venue_id"].nunique()),
        "unique_tokens_after_home_work_processing": int(checkins["token"].nunique()),
        "explicit_home_users": int(explicit_hw["has_explicit_home"].sum()),
        "explicit_work_users": int(explicit_hw["has_explicit_work"].sum()),
        "explicit_home_and_work_users": int((explicit_hw["has_explicit_home"] & explicit_hw["has_explicit_work"]).sum()),
        "same_home_work_venue_users": int(same_home_work.sum()),
        "explicit_home_visit_count": int(is_home.sum()),
        "explicit_work_visit_count": int(is_work.sum()),
        "poi_home_count_before_pruning": int((checkins["token"] == "POI_HOME").sum()),
        "poi_work_count_before_pruning": int((checkins["token"] == "POI_WORK").sum()),
        "poi_home_count_after_pruning": int((filtered["token"] == "POI_HOME").sum()),
        "poi_work_count_after_pruning": int((filtered["token"] == "POI_WORK").sum()),
        "original_user_length_distribution": describe_lengths(original_user_lengths),
        "pruned_user_length_distribution": describe_lengths(pruned_lengths_all),
        "final_segment_length_distribution": describe_lengths(pd.Series(np.concatenate([
            np.asarray(split_lengths["train"], dtype=np.int64),
            np.asarray(split_lengths["val"], dtype=np.int64),
            np.asarray(split_lengths["test"], dtype=np.int64),
        ]))),
        "age_bin_user_counts": {
            str(int(k)): int(v) for k, v in demo["age_bin5"].value_counts().sort_index().items()
        },
        "age_bin_segment_counts": {
            str(age): int(count)
            for age, count in pd.Series(
                np.concatenate([
                    np.asarray(split_attrs_demo["train"], dtype=np.float32).reshape((-1, 6))[:, 4],
                    np.asarray(split_attrs_demo["val"], dtype=np.float32).reshape((-1, 6))[:, 4],
                    np.asarray(split_attrs_demo["test"], dtype=np.float32).reshape((-1, 6))[:, 4],
                ])
                if any(split_attrs_demo[s] for s in ["train", "val", "test"])
                else np.asarray([], dtype=np.float32)
            ).astype(int).value_counts().sort_index().items()
        },
        "splits": {
            split: {
                "users": int(sum(1 for uid, s in split_map.items() if s == split)),
                "users_after_pruning": int(filtered.loc[filtered["split"] == split, "user_id"].nunique()),
                "segments": int(len(split_frames[split])),
                "checkins_after_pruning": int((filtered["split"] == split).sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    with open(output_root / "reduced_vocab_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
