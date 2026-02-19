#!/usr/bin/env bash
# Build mixture / rank-deficient / messy ATLAS worlds from demogroups.
# Change the paths below to your own folder paths before running.
#
# Prereq: <ATLAS_WORLD_ROOT>/world_*_demogroups must contain demo_a0_g0, demo_a0_g1, ... with
#   generated_sequences.npy and all_attr_results.demographics.npy. If missing, run step 0 below.
#
# Usage: from repo root, run:
#   bash helpers/example_create_mix.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SPLIT_ROOT="${SPLIT_ROOT:-$REPO/YOUR_DATA_FOLDER/controlled}"
VOCAB="$SPLIT_ROOT/tokenizer/vocab.txt"
ATLAS_WORLD_ROOT="${ATLAS_WORLD_ROOT:-$REPO/atlas_world_ca}"
HELPERS_DIR="$REPO/helpers"
TRAJGEN="$REPO/trajectory-generation"
MAX_LEN=64

# Pi matrices
PI_FULL_RANK="${PI_FULL_RANK:-$REPO/helpers/example_pi_matrix.json}"
PI_RANK_DEF="${PI_RANK_DEF:-$REPO/helpers/pi_matrix_rank_deficient.json}"
PI_MESSY="${PI_MESSY:-$REPO/helpers/pi_matrix_very_messy.json}"

# --- Step 0: Ensure CA demogroups have demo_* subdirs with .npy ---
need_demogroups=
for split in train val test; do
  if ! test -d "$ATLAS_WORLD_ROOT/world_${split}_demogroups/demo_a0_g0"; then
    need_demogroups=1
    break
  fi
done
if [ -n "$need_demogroups" ]; then
  echo "[STEP 0] Populating ATLAS world_*_demogroups from your split data..."
  for split in train val test; do
    python "$HELPERS_DIR/make_llp_world_from_split.py" \
      --split-root "$SPLIT_ROOT/$split" \
      --tokenizer-vocab "$VOCAB" \
      --out-root "$ATLAS_WORLD_ROOT/world_${split}_demogroups" \
      --region-mode demo_group \
      --min-region-size 500
  done
  echo "[STEP 0] Done."
else
  echo "[STEP 0] Skipped (demo_* dirs already present)."
fi

# --- Step 1: Build mixture worlds (full-rank, rank-deficient, messy) for train/val/test ---
build_mixture() {
  local pi_json="$1"
  local out_base="$2"
  local name="$3"
  echo "[STEP 1] Building $name worlds..."
  for split in train val test; do
    python "$HELPERS_DIR/make_llp_mixture_world_from_demogroups.py" \
      --source-world-root "$ATLAS_WORLD_ROOT/world_${split}_demogroups" \
      --pi-json "$pi_json" \
      --out-root "$out_base/world_${split}_${name}"
  done
  echo "[STEP 1] $name done."
}

build_mixture "$PI_FULL_RANK" "$REPO/atlas_world_ca_mixture" "mixture"
build_mixture "$PI_RANK_DEF"  "$REPO/atlas_world_ca_deficient" "rank_deficient"
build_mixture "$PI_MESSY"     "$REPO/atlas_world_ca_messy" "messy"

# --- Step 2 & 3: Write configs, build POI marginals and CBG cache for each world type ---
for WORLD_NAME in atlas_world_ca_mixture atlas_world_ca_deficient atlas_world_ca_messy; do
  WORLD_ROOT="$REPO/$WORLD_NAME"
  echo "[STEP 2–3] Configs + aggregates + cache for $WORLD_NAME..."
  for split in train val test; do
    WROOT="$WORLD_ROOT/world_${split}_mixture"
    if [ "$WORLD_NAME" = "atlas_world_ca_deficient" ]; then WROOT="$WORLD_ROOT/world_${split}_rank_deficient"; fi
    if [ "$WORLD_NAME" = "atlas_world_ca_messy" ]; then WROOT="$WORLD_ROOT/world_${split}_messy"; fi
    CFG="$WORLD_ROOT/configs/$split"
    AGG="$WORLD_ROOT/aggregates/$split"
    CACHE="$WORLD_ROOT/cache/$split"
    mkdir -p "$CFG" "$AGG" "$CACHE"
    python "$HELPERS_DIR/write_llp_configs.py" \
      --world-root "$WROOT" \
      --vocab-path "$VOCAB" \
      --out-config-dir "$CFG" \
      --out-aggregates-dir "$AGG" \
      --out-cache-dir "$CACHE"
    python "$TRAJGEN/scripts/precompute/build_poi_marginals.py" --config "$CFG/poi_marginals_llp_world.yaml"
    python "$TRAJGEN/scripts/precompute/cache_cbg_conditionals.py" --config "$CFG/cache_cbg_conditionals_llp_world.yaml"
  done
done

# --- Step 4: Length distributions for training (from train mixture worlds) ---
for WORLD_NAME in atlas_world_ca_mixture atlas_world_ca_deficient atlas_world_ca_messy; do
  WORLD_ROOT="$REPO/$WORLD_NAME"
  WROOT="$WORLD_ROOT/world_train_mixture"
  if [ "$WORLD_NAME" = "atlas_world_ca_deficient" ]; then WROOT="$WORLD_ROOT/world_train_rank_deficient"; fi
  if [ "$WORLD_NAME" = "atlas_world_ca_messy" ]; then WROOT="$WORLD_ROOT/world_train_messy"; fi
  python "$HELPERS_DIR/build_length_dists_from_llp_world.py" \
    --world-root "$WROOT" \
    --out-json "$WORLD_ROOT/length_dists_train_demogroups.json" \
    --max-length "$MAX_LEN"
  echo "[STEP 4] length_dists for $WORLD_NAME done."
done

echo "[DONE] ATLAS mixture / deficient / messy worlds and configs/cache/length_dists are ready."
