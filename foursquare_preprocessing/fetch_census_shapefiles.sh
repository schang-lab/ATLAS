#!/bin/bash
# Download the 2023 TIGER/Line census block-group shapefiles used by
# 05_build_atlas_world_georegions.py for point-in-polygon region assignment.
#
# Only New York (36) and New Jersey (34) are needed for the NYC georegions.
# If these files are absent, step 05 falls back to an approximate lat/lon rule
# (~85% accurate along the Hudson: it mislabels piers, the GW Bridge, and
# Liberty Island), so downloading them is recommended to reproduce the paper.
#
# Usage:  bash foursquare_preprocessing/fetch_census_shapefiles.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/census_shapefiles/2023/BG"
BASE_URL="https://www2.census.gov/geo/tiger/TIGER2023/BG"

mkdir -p "${DEST}"

for state in 34 36; do
  file="tl_2023_${state}_bg.zip"
  if [[ -f "${DEST}/${file}" ]]; then
    echo "already present: ${DEST}/${file}"
    continue
  fi
  echo "downloading ${file} ..."
  curl -fsSL -o "${DEST}/${file}" "${BASE_URL}/${file}"
done

echo "done -> ${DEST}"
