"""Rule-based age score for Foursquare NYC users.

Single definition of the "age score" described in the paper appendix
(Foursquare NYC Dataset / Demographic labeling). Both `02_infer_age_labels.py`
and `03_build_reduced_vocab_split.py` import it so the rule cannot drift
between the labeling step and the binning step.

The score reads only aggregated behavioral summaries of a user's check-in
history (category shares, nighttime share, venue diversity), never the raw
trajectory, so that age inference stays independent of the spatial/sequential
structure the trajectory model is trained on.
"""

from __future__ import annotations

import json

# Category keyword groups. `s_c` below is the share of category `c` among a
# user's ten most frequent venue categories.
NIGHTLIFE_KEYWORDS = ("bar", "nightclub", "club", "lounge", "brewery")
FITNESS_KEYWORDS = ("gym", "fitness", "yoga", "rock climbing")
MEDICAL_KEYWORDS = ("medical", "doctor", "pharmacy", "hospital")
RELIGIOUS_KEYWORDS = ("church", "synagogue", "mosque", "religious")

# Age bin schema: 0 = <30, 1 = 30-40, 2 = 40-50, 3 = 50+.
NUM_AGE_BINS = 4


def rule_based_age_score(user_feat: dict) -> float:
    """Return the scalar age score; higher means behaviorally younger.

    Adds `s_c` for nightlife categories and `0.5 * s_c` for fitness categories;
    subtracts `s_c` for medical and religious categories. A nighttime check-in
    share (00:00-05:59) above 0.15 adds 0.2, below 0.05 subtracts 0.1. Venue
    diversity (unique venues / total check-ins) above 0.5 adds 0.1, below 0.3
    subtracts 0.1.
    """
    raw_categories = user_feat.get("top_categories", "{}")
    top_cats = json.loads(raw_categories) if isinstance(raw_categories, str) else dict(raw_categories)

    young_score = 0.0
    old_score = 0.0

    for category, share in top_cats.items():
        name = str(category).lower()
        share = float(share)
        if any(k in name for k in NIGHTLIFE_KEYWORDS):
            young_score += share
        if any(k in name for k in FITNESS_KEYWORDS):
            young_score += share * 0.5
        if any(k in name for k in MEDICAL_KEYWORDS):
            old_score += share
        if any(k in name for k in RELIGIOUS_KEYWORDS):
            old_score += share

    night_pct = float(user_feat.get("night_pct", 0.0) or 0.0)
    if night_pct > 0.15:
        young_score += 0.2
    elif night_pct < 0.05:
        old_score += 0.1

    diversity = float(user_feat.get("diversity", 0.5) or 0.5)
    if diversity > 0.5:
        young_score += 0.1
    elif diversity < 0.3:
        old_score += 0.1

    return young_score - old_score


def rule_based_age_bin(user_feat: dict) -> int:
    """Provisional age bin from fixed score thresholds.

    Used only for the intermediate `demo.csv`. The bin assignment behind the
    reported results is produced by `03_build_reduced_vocab_split.py
    --age-mode balanced_segments`, which ranks users by `rule_based_age_score`
    and cuts bins to equalize trajectory-segment counts instead.
    """
    score = rule_based_age_score(user_feat)
    if score > 0.15:
        return 0  # <30
    if score > 0.0:
        return 1  # 30-40
    if score > -0.1:
        return 2  # 40-50
    return 3  # 50+
