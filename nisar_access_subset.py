
#!/usr/bin/env python3
"""Subset selected variables from a NISAR GCOV granule and write a Zarr store."""

import argparse
import json
import os
import shutil
from typing import List, Optional, Tuple

import earthaccess
import h5py
import numpy as np
import xarray as xr
import zarr

DEFAULT_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
DEFAULT_X = f"{DEFAULT_GROUP}/xCoordinates"
DEFAULT_Y = f"{DEFAULT_GROUP}/yCoordinates"
DEFAULT_ASF_S3_CREDS_URL = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _normalize_blank(value: Optional[str]) -> str:
    value = (value or "").strip()
    return "" if value in {"none", "None", "null", "NULL", '""', "''"} else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset variables from a NISAR GCOV granule and write an intermediate Zarr store."
    )
    parser.add_argument("--access_mode", choices=["auto", "s3", "https"], default="auto")
    parser.add_argument("--https_href", default="")
    parser.add_argument("--s3_href", default="")
    parser.add_argument("--short_name", default="NISAR_L2_GCOV_BETA_V1")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--granule_index", type=int, default=0)
    parser.add_argument("--asf_s3_creds_url", default=DEFAULT_ASF_S3_CREDS_URL)

    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--vars", default="HHHH")
    parser.add_argument("--x_path", default=DEFAULT_X)
    parser.add_argument("--y_path", default=DEFAULT_Y)

    parser.add_argument("--bbox", default="")
    parser.add_argument("--bbox_crs", default="")

    parser.add_argument("--out_dir", default=os.environ.get("USER_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or "output")
    parser.add_argument("--out_name", default="nisar_subset.zarr")

    args = parser.parse_args()
    for attr in ("https_href", "s3_href", "bbox", "bbox_crs", "out_dir", "out_name"):
        setattr(args, attr, _normalize_blank(getattr(args, attr, "")))
    if not args.out_name:
        args.out_name = "nisar_subset.zarr"
    return args


def resolve_granule_hrefs(args: argparse.Namespace) -> Tuple[str, str]:
    https_href = args.https_href
    s3_href = args.s3_href
    if https_href or s3_href:
        return https_href, s3_href

    try:
        earthaccess.login(strategy="environment")
    except Exception:
        earthaccess.login()

    results = earthaccess.search_data(
        short_name=args.short_name,
        count=args.count,
        cloud_hosted=True,
    )
    if not results:
        raise RuntimeError("No granules found. Provide --https_href and/or --s3_href explicitly.")

    if args.granule_index >= len(results):
        raise IndexError(f"--granule_index {args.granule_index} is out of range for {len(results)} result(s).")

    granule = results[args.granule_index]
    https_links = granule.data_links() or []
    s3_links = granule.data_links(access="direct") or []
    return (https_links[0] if https_links else "", s3_links[0] if s3_links else "")


def parse_bbox(value: str) -> Optional[Tuple[float, float, float, float]]:
    if not value:
        return None
    parts = _split_csv(value)
    if len(parts) != 4:
        raise ValueError("bbox must be 'minx,miny,maxx,maxy'")
    minx, miny, maxx, maxy = (float(p) for p in parts)
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox min values must be smaller than max values")
    return minx, miny, maxx, maxy


def bbox_to_slices(
    x: np.ndarray,
    y: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[slice, slice]:
    minx, miny, maxx, maxy = bbox
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size == 0 or y.size == 0:
        raise RuntimeError("Coordinate vectors are empty.")

    x_asc = x[0] <= x[-1]
    y_asc = y[0] <= y[-1]

    if x_asc:
        x0 = int(np.searchsorted(x, minx, side="left"))
        x1 = int(np.searchsorted(x, maxx, side="right"))
    else:
        xr = x[::-1]
        x0 = len(x) - int(np.searchsorted(xr, maxx, side="right"))
        x1 = len(x) - int(np.searchsorted(xr, minx, side="left"))

    if y_asc:
        y0 = int(np.searchsorted(y, miny, side="left"))
        y1 = int(np.searchsorted(y, maxy, side="right"))
    else:
        yr = y[::-1]
        y0 = len(y) - int(np.searchsorted(yr, maxy, side="right"))
        y1 = len(y) - int(np.searchsorted(yr, miny, side="left"))

    x0 = max(0, min(len(x), x0))
    x1 = max(0, min(len(x), x1))
    y0 = max(0, min(len(y), y0))
    y1 = max(0, min(len(y), y1))

    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("BBox produced an empty slice. Make sure it is in the same CRS/units as xCoordinates/yCoordinates.")

    return slice(y0, y1), slice(x0, x1)


def _get_s3_credentials(asf_s3_creds_url: str) -> dict:
    from maap.maap import MAAP

    maap = MAAP()
    creds = maap.aws.earthdata_s3_credentials(asf_s3_creds_url)
    required = {"accessKeyId", "secretAccessKey", "sessionToken"}
    missing = required.difference(creds)
    if missing:
        raise RuntimeError(f"Missing expected AWS credential fields: {sorted(missing)}")
    return creds


def open_file_like(
    access_mode: str,
    https_href: str,
    s3_href: str,
    asf_s3_creds_url: str,
):
    fsspec_params = {"cache_type": "blockcache", "block_size": 8 * 1024 * 1024}

    def open_https():
        if not https_href:
            raise RuntimeError("HTTPS href not available.")
        try:
            earthaccess.login(strategy="environment")
        except Exception:
            earthaccess.login()
        fs = earthaccess.get_fsspec_https_session()
        return fs.open(https_href, mode="rb", **fsspec_params), "https", https_href

    def open_s3():
        if not s3_href:
            raise RuntimeError("S3 href not available.")
        import s3fs

        creds = _get_s3_credentials(asf_s3_creds_url)
        fs = s3fs.S3FileSystem(
            anon=False,
            key=creds["accessKeyId"],
            secret=creds["secretAccessKey"],
            token=creds["sessionToken"],
        )
        return fs.open(s3_href, mode="rb", **fsspec_params), "s3", s3_href

    chosen_mode = access_mode
    if chosen_mode == "auto":
        chosen_mode = "s3" if s3_href else "https"

    if chosen_mode == "https":
        try:
            return open_https()
        except Exception:
            if s3_href:
                return open_s3()
            raise
    return open_s3()


def build_dataset(
    h5f: h5py.File,
    group: str,
    x_path: str,
    y_path: str,
    var_names: List[str],
    bbox: Optional[Tuple[float, float, float, float]],
) -> xr.Dataset:
    x = h5f[x_path][()]
    y = h5f[y_path][()]

    if bbox is not None:
        yslice, xslice = bbox_to_slices(x, y, bbox)
        x_sub = np.asarray(x)[xslice]
        y_sub = np.asarray(y)[yslice]
    else:
        yslice, xslice = slice(None), slice(None)
        x_sub = np.asarray(x)
        y_sub = np.asarray(y)

    data_vars = {}
    for var_name in var_names:
        dpath = f"{group}/{var_name}"
        if dpath not in h5f:
            raise KeyError(f"Dataset not found: {dpath}")
        arr = h5f[dpath][yslice, xslice]
        data_vars[var_name] = (("y", "x"), np.asarray(arr))

    attrs = {}
    proj_path = f"{group}/projection"
    if proj_path in h5f:
        proj = h5f[proj_path]
        for key in ("epsg_code", "spatial_ref", "grid_mapping_name"):
            if key in proj.attrs:
                value = proj.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode()
                attrs[key] = value

    return xr.Dataset(
        data_vars=data_vars,
        coords={"x": ("x", x_sub), "y": ("y", y_sub)},
        attrs=attrs,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    var_names = _split_csv(args.vars)
    if not var_names:
        raise ValueError("No variables provided in --vars")

    bbox = parse_bbox(args.bbox)
    https_href, s3_href = resolve_granule_hrefs(args)
    file_obj, chosen_mode, chosen_href = open_file_like(
        args.access_mode,
        https_href,
        s3_href,
        args.asf_s3_creds_url,
    )

    with file_obj as fp:
        with h5py.File(
            fp,
            "r",
            rdcc_nbytes=4 * 1024 * 1024,
            page_buf_size=16 * 1024 * 1024,
        ) as h5f:
            ds = build_dataset(h5f, args.group, args.x_path, args.y_path, var_names, bbox)

    ds.attrs.update(
        {
            "source_href": chosen_href,
            "access_mode": chosen_mode,
            "group": args.group,
            "vars": ",".join(var_names),
            "bbox": args.bbox,
            "bbox_crs": args.bbox_crs,
            "note": "bbox is assumed to be in the same CRS/units as xCoordinates/yCoordinates.",
        }
    )

    out_path = os.path.abspath(os.path.join(args.out_dir, args.out_name))
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    ds.to_zarr(out_path, mode="w", consolidated=True)

    manifest = {
        "out_zarr": out_path,
        "source_href": chosen_href,
        "access_mode": chosen_mode,
        "group": args.group,
        "vars": var_names,
        "bbox": bbox,
        "bbox_crs": args.bbox_crs,
        "x_path": args.x_path,
        "y_path": args.y_path,
        "zarr_consolidated": True,
    }
    manifest_path = os.path.abspath(os.path.join(args.out_dir, "manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print("WROTE_ZARR:", out_path)
    print("WROTE_MANIFEST:", manifest_path)
    print("SOURCE_MODE:", chosen_mode)
    print("SOURCE_HREF:", chosen_href)
    print("OUTPUT_DIR:", os.path.abspath(args.out_dir))
    print("OUTPUT_CONTENTS:", sorted(os.listdir(args.out_dir)))


if __name__ == "__main__":
    main()
