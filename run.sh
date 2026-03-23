#!/usr/bin/env bash
set -euo pipefail

echo "[run.sh] Starting NISAR subset job..."
echo "[run.sh] Args: $*"

# Prefer the MAAP/HySDS-style published output dir.
# Fallbacks:
#   1) USER_OUTPUT_DIR
#   2) OUTPUT_DIR
#   3) relative ./output
OUT_DIR="${USER_OUTPUT_DIR:-${OUTPUT_DIR:-output}}"
mkdir -p "$OUT_DIR"

PY="/opt/app/nisar_access_subset.py"
if [[ ! -f "$PY" ]]; then
  echo "[run.sh] ERROR: Missing $PY (did build.sh copy it?)"
  exit 2
fi

# ------------------------------------------------------------
# Positional mode:
#   run.sh https HHHH <https_href> <s3_href> <bbox> <bbox_crs> <out_name>
# ------------------------------------------------------------
if [[ $# -ge 2 && "$1" != --* ]]; then
  ACCESS_MODE="${1:-auto}"
  VARS="${2:-HHHH}"
  HTTPS_HREF="${3:-}"
  S3_HREF="${4:-}"
  BBOX="${5:-}"
  BBOX_CRS="${6:-}"
  OUT_NAME="${7:-nisar_subset.zarr}"

  CMD=(
    python "$PY"
    --access_mode "$ACCESS_MODE"
    --vars "$VARS"
    --out_dir "$OUT_DIR"
    --out_name "$OUT_NAME"
  )

  [[ -n "$HTTPS_HREF" ]] && CMD+=(--https_href "$HTTPS_HREF")
  [[ -n "$S3_HREF"   ]] && CMD+=(--s3_href "$S3_HREF")
  [[ -n "$BBOX"      ]] && CMD+=(--bbox "$BBOX")
  [[ -n "$BBOX_CRS"  ]] && CMD+=(--bbox_crs "$BBOX_CRS")

  echo "[run.sh] Running (positional-mode): ${CMD[*]}"
  "${CMD[@]}"

else
  # ----------------------------------------------------------
  # Flag-style passthrough:
  #   run.sh --access_mode https --vars HHHH ...
  #
  # If --out_dir is not explicitly passed, inject the publishable OUT_DIR.
  # ----------------------------------------------------------
  HAS_OUT_DIR=0
  for arg in "$@"; do
    if [[ "$arg" == "--out_dir" ]]; then
      HAS_OUT_DIR=1
      break
    fi
  done

  if [[ "$HAS_OUT_DIR" -eq 1 ]]; then
    echo "[run.sh] Running (flag-mode): python $PY $*"
    python "$PY" "$@"
  else
    echo "[run.sh] Running (flag-mode): python $PY --out_dir $OUT_DIR $*"
    python "$PY" --out_dir "$OUT_DIR" "$@"
  fi
fi

echo "[run.sh] Listing outputs in $OUT_DIR"
ls -lah "$OUT_DIR" || true

echo "[run.sh] Recursive output listing"
find "$OUT_DIR" -maxdepth 3 -print || true

echo "[run.sh] Done."
