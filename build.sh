#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /opt/app

cp -v "$SCRIPT_DIR/nisar_access_subset.py" /opt/app/nisar_access_subset.py
cp -v "$SCRIPT_DIR/nisar_access_subset.cwl" /opt/app/nisar_access_subset.cwl
cp -v "$SCRIPT_DIR/run.sh" /opt/app/run.sh

chmod 755 /opt/app/nisar_access_subset.py
chmod 755 /opt/app/run.sh
