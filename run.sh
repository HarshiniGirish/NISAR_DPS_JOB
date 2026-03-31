#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p output

conda run --live-stream -p /opt/conda/envs/nisar_access_subset \
  python "${basedir}/nisar_access_subset.py" "$@"

find output -maxdepth 3 -print || true
