#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTDIR="${USER_OUTPUT_DIR:-${OUTPUT_DIR:-output}}"

mkdir -p "${OUTDIR}"

conda run --live-stream -p /opt/conda/envs/nisar_access_subset \
  python "${basedir}/nisar_access_subset.py" --out_dir "${OUTDIR}" "$@"

find "${OUTDIR}" -maxdepth 3 -print || true
