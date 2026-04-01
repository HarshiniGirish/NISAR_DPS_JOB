#!/usr/bin/env bash
set -euo pipefail

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_PREFIX="/opt/conda/envs/nisar_access_subset"

conda env remove -p "${ENV_PREFIX}" -y || true
conda env create -f "${basedir}/env.yml" --prefix "${ENV_PREFIX}"
conda clean -afy

conda run -p "${ENV_PREFIX}" python - <<'PY'
import earthaccess
import fsspec
import s3fs
import aiobotocore
import botocore
import boto3
import h5py
import h5netcdf
import numpy
import requests
import xarray
import zarr
import numcodecs

print("Environment validation successful.")
print("earthaccess", earthaccess.__version__)
print("fsspec", fsspec.__version__)
print("s3fs", s3fs.__version__)
print("aiobotocore", aiobotocore.__version__)
print("botocore", botocore.__version__)
print("boto3", boto3.__version__)
print("h5py", h5py.__version__)
print("h5netcdf", h5netcdf.__version__)
print("numpy", numpy.__version__)
print("requests", requests.__version__)
print("xarray", xarray.__version__)
print("zarr", zarr.__version__)
print("numcodecs", numcodecs.__version__)
PY
