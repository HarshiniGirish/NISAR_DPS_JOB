#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="conda run --live-stream -p /opt/conda/envs/nisar_access_subset python"

mkdir -p output

ACCESS_MODE="${1:-auto}"
HTTPS_HREF="${2:-}"
S3_HREF="${3:-}"
VARS="${4:-HHHH}"
GROUP="${5:-/science/LSAR/GCOV/grids/frequencyA}"
BBOX="${6:-}"
BBOX_CRS="${7:-}"
OUT_NAME="${8:-nisar_subset.zarr}"

ARGS=(
  "--access_mode" "${ACCESS_MODE}"
  "--vars" "${VARS}"
  "--group" "${GROUP}"
  "--out_dir" "output"
  "--out_name" "${OUT_NAME}"
)

if [[ -n "${HTTPS_HREF}" ]]; then
  ARGS+=("--https_href" "${HTTPS_HREF}")
fi

if [[ -n "${S3_HREF}" ]]; then
  ARGS+=("--s3_href" "${S3_HREF}")
fi

if [[ -n "${BBOX}" ]]; then
  ARGS+=("--bbox" "${BBOX}")
fi

if [[ -n "${BBOX_CRS}" ]]; then
  ARGS+=("--bbox_crs" "${BBOX_CRS}")
fi

LOGFILE="output/nisar_access_subset.log"

echo "Running NISAR access subset job"
echo "Arguments: ${ARGS[*]}"

set -x
${PYTHON_BIN} "${basedir}/nisar_access_subset.py" "${ARGS[@]}" 2>&1 | tee "${LOGFILE}"
set +x

echo "Listing output contents"
find output -maxdepth 3 -print || true

echo "Run completed successfully."
