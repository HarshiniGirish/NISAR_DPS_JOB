#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p output

PYTHON_CMD=(
  conda run
  --live-stream
  -p /opt/conda/envs/nisar_access_subset
  python
  "${basedir}/nisar_access_subset.py"
)

NORMALIZED_ARGS=()

while (($#)); do
  case "$1" in
    --access_mode|--https_href|--s3_href|--short_name|--count|--granule_index|--asf_s3_creds_url|--group|--vars|--x_path|--y_path|--bbox|--bbox_crs|--out_dir|--out_name)
      if (($# < 2)); then
        echo "ERROR: missing value for $1" >&2
        exit 2
      fi
      NORMALIZED_ARGS+=("${1}=${2}")
      shift 2
      ;;
    *)
      NORMALIZED_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "NORMALIZED_ARGS: ${NORMALIZED_ARGS[*]}"

"${PYTHON_CMD[@]}" "${NORMALIZED_ARGS[@]}"

find output -maxdepth 3 -print || true
