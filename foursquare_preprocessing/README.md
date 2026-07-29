# Foursquare NYC preprocessing

This directory turns the public Foursquare check-in releases into the
split-data format ATLAS trains on, and builds the ATLAS worlds used for the
NYC experiments in the paper.

**No data is distributed with this repository.** Download the two releases from
the authors' page and run steps 1–3 below to regenerate the corpus exactly as
used in the paper.

## Download the raw data

Both releases are available from Dingqi Yang's dataset page:

<https://sites.google.com/site/yangdingqi/home/foursquare-dataset>

| File | Release |
| --- | --- |
| `dataset_TIST2015/dataset_TIST2015_Checkins.txt` | Global-scale check-in dataset (Yang et al., 2015) |
| `dataset_TIST2015/dataset_TIST2015_POIs.txt` | POI metadata for the above |
| `dataset_UbiComp2016/dataset_UbiComp2016_UserProfile_NYC.txt` | User profiles with self-reported gender (Yang et al., 2016; 2018) |

Place them under `dataset-foursquare/` (the default paths in
[`configs/foursquare_nyc.yaml`](configs/foursquare_nyc.yaml)). They are
distributed by their original authors for research use under their own terms.

## Pipeline

Run from the repository root. Steps 1–3 take a few minutes on a laptop.

```bash
# 1. Join check-ins to POI metadata, link users to profiles, restrict to the
#    NYC bounding box (40.49-40.92 N, -74.27--73.68 W), keep users with >=10 check-ins
python foursquare_preprocessing/01_extract_and_link.py \
  --tist2015_checkins dataset-foursquare/dataset_TIST2015/dataset_TIST2015_Checkins.txt \
  --tist2015_pois dataset-foursquare/dataset_TIST2015/dataset_TIST2015_POIs.txt \
  --ubicomp2016 dataset-foursquare/dataset_UbiComp2016/dataset_UbiComp2016_UserProfile_NYC.txt \
  --city NYC \
  --output_dir foursquare_preprocessing/outputs

# 2. Infer age pseudo-labels from aggregated behavioral features
python foursquare_preprocessing/02_infer_age_labels.py \
  --input_dir foursquare_preprocessing/outputs \
  --output_dir foursquare_preprocessing/outputs \
  --provider rule_based

# 3. Build the reduced-vocabulary 80/10/10 split
python foursquare_preprocessing/03_build_reduced_vocab_split.py \
  --input-dir foursquare_preprocessing/outputs \
  --output-root data/foursquare_nyc/controlled \
  --vocab-size 3000 --max-len 64 \
  --age-mode balanced_segments --attr-mode gender --raw-poi-tokens
```

Step 1 writes `linked_checkins.csv`, `user_profiles.csv`,
`user_features.csv` (aggregated behavioral features), and `stats.json`
(linkage statistics). Step 2 adds `demo.csv` (per-user gender + provisional
age bin). Step 3 produces the split-data tree
(`controlled/{train,val,test}/`) described in the top-level README, using a
user-level split stratified by gender (deterministic, `--seed 42`).

Step 4 onwards builds the ATLAS worlds — see
[Build ATLAS worlds](../README.md#b5-build-atlas-worlds) in the top-level
README.

## Expected corpus statistics

Rerunning steps 1–3 reproduces these paper-reported quantities exactly (they
are independent of the split ratios):

| Quantity | Value |
| --- | --- |
| Linked gender-labeled users | 7,036 |
| Users after run-length pruning | 6,991 (45 dropped to zero length) |
| Retained check-ins | 257,772 |
| Trajectory segments | 8,851 |
| Data vocabulary | 3,000 POI tokens (3,005 with specials) |
| Median segment length | 22 tokens (p95 = 62) |
| Segments per age bin | 2,213 / 2,213 / 2,213 / 2,212 |

## Demographic labels

`gender` is the self-reported attribute from the public user-profile release.

Age is an **inferred pseudo-label, not a measured age**. Step 3
(`--age-mode balanced_segments`) ranks users by the rule-based behavioral score
in [`age_rule.py`](age_rule.py) (computed from `user_features.csv`) and cuts
four ordinal bins that equalize trajectory-segment counts. See the paper
appendix for the interpretation caveat. ATLAS never sees these labels during
training — they are used only to construct region partitions and to evaluate.

The scoring rule has a single definition in `age_rule.py`, shared by steps 2
and 3. Step 2's `--provider openai` selects an optional LLM-based alternative
that was **not** used for any reported result.

The Foursquare release provides no reliable home/work coordinates, so the
generated `all_attr_results.npy` coordinates are zeros: the Phase 1 model for
this dataset is unconditional and Phase 2 conditions on demographics only.

## Georegions

`05_build_atlas_world_georegions.py` assigns each trajectory to one of six NYC
georegions — Manhattan South / Middle / North, Brooklyn, Outer (Queens + Bronx 
/+ Staten Island), and New Jersey. Each trajectory is anchored to its
most-visited POI and assigned by point-in-polygon against 2023 Census
boundaries; run `fetch_census_shapefiles.sh` first. Without the shapefiles the
script falls back to an approximate lat/lon rule (~85% accurate along the
Hudson).
