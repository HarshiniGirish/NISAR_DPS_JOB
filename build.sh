# !/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[build.sh] Repo dir: $REPO_DIR"
echo "[build.sh] Installing Python deps..."

if [[ -f requirements.txt ]]; then
  pip install --no-cache-dir -r requirements.txt
else
  echo "[build.sh] WARNING: requirements.txt not found. Skipping pip install."
fi

echo "[build.sh] Staging algorithm files into /opt/app ..."
mkdir -p /opt/app

# Copy your algorithm entrypoint + CWL into /opt/app
cp -v nisar_access_subset.py /opt/app/nisar_access_subset.py
cp -v nisar_access_subset.cwl /opt/app/nisar_access_subset.cwl

chmod 755 /opt/app/nisar_access_subset.py

echo "[build.sh] Done."
