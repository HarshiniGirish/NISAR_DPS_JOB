#!/usr/bin/env python3
import argparse
import os
import numpy as np
import h5py
import earthaccess

def parse_args():
    p = argparse.ArgumentParser(description="NISAR GCOV access + small window subset (S3 or HTTPS)")
    p.add_argument("--short_name", default="NISAR_L2_GCOV_BETA_V1")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--granule_index", type=int, default=0)

    p.add_argument("--access_mode", choices=["s3", "https"], default="s3",
                   help="s3 requires MAAP temp creds. https works anywhere with Earthdata Login.")
    p.add_argument("--var", default="HHHH", help="GCOV variable (e.g., HHHH, HVHV)")
    p.add_argument("--h5_path", default="/science/LSAR/GCOV/grids/frequencyA",
                   help="Group path containing variables")

    # small window subset (row/col slicing)
    p.add_argument("--row0", type=int, default=0)
    p.add_argument("--row1", type=int, default=1024)
    p.add_argument("--col0", type=int, default=0)
    p.add_argument("--col1", type=int, default=1024)

    p.add_argument("--out_dir", default=os.environ.get("OUTPUT_DIR", "/tmp/output"))
    return p.parse_args()

def open_file_like(access_mode, https_link, s3_link):
    # Cloud-optimized I/O tuning copied from MAAP tutorial defaults:
    # fsspec blockcache + HDF5 driver cache knobs. :contentReference[oaicite:5]{index=5}
    fsspec_params = {"cache_type": "blockcache", "block_size": 8 * 1024 * 1024}
    h5py_driver_kwds = {"page_buf_size": 16 * 1024 * 1024, "rdcc_nbytes": 4 * 1024 * 1024}

    if access_mode == "https":
        fs = earthaccess.get_fsspec_https_session()
        return fs.open(https_link, mode="rb", **fsspec_params), h5py_driver_kwds

    # access_mode == "s3"
    # This requires MAAP environment + maap-py. :contentReference[oaicite:6]{index=6}
    from maap.maap import MAAP
    import s3fs

    maap = MAAP()
    asf_s3 = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"
    creds = maap.aws.earthdata_s3_credentials(asf_s3)

    fs_s3 = s3fs.S3FileSystem(
        anon=False,
        key=creds["accessKeyId"],
        secret=creds["secretAccessKey"],
        token=creds["sessionToken"],
    )
    return fs_s3.open(s3_link, mode="rb", **fsspec_params), h5py_driver_kwds

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Earthdata discovery :contentReference[oaicite:7]{index=7}
    earthaccess.login()
    results = earthaccess.search_data(
        short_name=args.short_name,
        count=args.count,
        cloud_hosted=True,
    )
    if not results:
        raise RuntimeError("No granules found.")

    g = results[args.granule_index]

    # Resolve links: https + direct s3 :contentReference[oaicite:8]{index=8}
    https_link = g.data_links()[0]
    s3_link = g.data_links(access="direct")[0]

    file_obj, driver_kwds = open_file_like(args.access_mode, https_link, s3_link)

    # Use h5py on the file-like object; read a small window from the variable
    grp = f"{args.h5_path}/{args.var}"
    with h5py.File(file_obj, "r", driver_kwds=driver_kwds) as f:
        dset = f[grp]
        arr = dset[args.row0:args.row1, args.col0:args.col1]

    out_path = os.path.join(args.out_dir, f"{args.var}_r{args.row0}-{args.row1}_c{args.col0}-{args.col1}.npy")
    np.save(out_path, arr)

    # Print outputs for DPS logs
    print("WROTE:", out_path)
    print("HTTPS:", https_link)
    print("S3:", s3_link)

if __name__ == "__main__":
    main()
