#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_PREFIX="/opt/conda/envs/nisar_access_subset"

conda env create -f "${basedir}/env.yml" --prefix "${ENV_PREFIX}"
conda clean -afy

conda run -p "${ENV_PREFIX}" python - <<'PY'
import earthaccess
import fsspec
import h5py
import h5netcdf
import numpy
import s3fs
import xarray
import zarr
print("Environment validation successful.")
PY
