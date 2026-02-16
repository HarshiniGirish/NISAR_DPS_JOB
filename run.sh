# !/usr/bin/env bash
set -euo pipefail
basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

PY='conda run --live-stream -p /opt/conda/envs/nisar_subset_env python'

# Always write products into ./output (relative to DPS working dir)
mkdir -p output

# ---- DPS positional inputs --------------------------------------------------
# 1 access_mode, 2 vars, 3 https_href, 4 s3_href, 5 bbox, 6 bbox_crs, 7 out_name
ACCESS_MODE="${1:-https}"
VARS="${2:-HHHH}"
HTTPS_HREF="${3:-}"
S3_HREF="${4:-}"
BBOX="${5:-}"
BBOX_CRS="${6:-}"
OUT_NAME="${7:-nisar_subset.zarr}"

# ---- Build CLI for Python ---------------------------------------------------
ARGS=()
ARGS+=("--access_mode" "${ACCESS_MODE}")
ARGS+=("--vars" "${VARS}")

[[ -n "${HTTPS_HREF}" ]] && ARGS+=("--https_href" "${HTTPS_HREF}")
[[ -n "${S3_HREF}"   ]] && ARGS+=("--s3_href"   "${S3_HREF}")
[[ -n "${BBOX}"      ]] && ARGS+=("--bbox"      "${BBOX}")
[[ -n "${BBOX_CRS}"  ]] && ARGS+=("--bbox_crs"  "${BBOX_CRS}")

ARGS+=("--out_dir" "output")
ARGS+=("--out_name" "${OUT_NAME}")

# ---- Run & capture stderr to triage ----------------------------------------
logfile="_nisar_subset.log"
set -x
${PY} "${basedir}/nisar_access_subset.py" "${ARGS[@]}" 2>"${logfile}"

# Include stdio + log in products (same trick as OPERA)
cp -v _stderr.txt _stdout.txt output/ 2>/dev/null || true
mv -v "${logfile}" output/ || true
set +x
