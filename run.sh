#!/usr/bin/env bash
set -euo pipefail

echo "[run.sh] Starting NISAR subset job..."
echo "[run.sh] Args: $*"

OUT_DIR="${USER_OUTPUT_DIR:-${OUTPUT_DIR:-output}}"
mkdir -p "$OUT_DIR"

PY="/opt/app/nisar_access_subset.py"
if [[ ! -f "$PY" ]]; then
  echo "[run.sh] ERROR: Missing $PY"
  exit 2
fi

python "$PY" "$@"

echo "[run.sh] Listing outputs in $OUT_DIR"
ls -lah "$OUT_DIR" || true
find "$OUT_DIR" -maxdepth 3 -print || true

echo "[run.sh] Done."
