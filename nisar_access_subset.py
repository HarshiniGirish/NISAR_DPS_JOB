#!/usr/bin/env python3
"""
Subset selected variables from a NISAR GCOV granule and write a Zarr store.

Behavior:
- access_mode=s3     -> MAAP/ASF temporary S3 credentials
- access_mode=https  -> HTTPS only if non-interactive Earthaccess creds exist
- access_mode=auto   -> prefer s3 in DPS, then https if explicitly available

Also:
- reprojects bbox to dataset CRS when bbox_crs is provided
- checks bbox overlap against dataset extent before slicing
- prints clearer debug information in job logs
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Tuple

import earthaccess
import h5py
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

DEFAULT_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
DEFAULT_X = f"{DEFAULT_GROUP}/xCoordinates"
DEFAULT_Y = f"{DEFAULT_GROUP}/yCoordinates"
DEFAULT_ASF_S3_CREDS_URL = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _normalize_blank(value: Optional[str]) -> str:
    value = (value or "").strip()
    return "" if value in {"none", "None", "null", "NULL", '""', "''"} else value


def _normalize_cli_args(argv: List[str]) -> List[str]:
    value_options = {
        "--access_mode",
        "--https_href",
        "--s3_href",
        "--short_name",
        "--count",
        "--granule_index",
        "--asf_s3_creds_url",
        "--group",
        "--vars",
        "--x_path",
        "--y_path",
        "--bbox",
        "--bbox_crs",
        "--out_dir",
        "--out_name",
    }

    normalized: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in value_options and i + 1 < len(argv):
            normalized.append(f"{arg}={argv[i + 1]}")
            i += 2
            continue
        normalized.append(arg)
        i += 1
    return normalized


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
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("USER_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or "output",
    )
    parser.add_argument("--out_name", default="nisar_subset.zarr")

    raw_argv = sys.argv[1:]
    normalized_argv = _normalize_cli_args(raw_argv)

    print("RAW_ARGV:", raw_argv)
    print("NORMALIZED_ARGV:", normalized_argv)

    args = parser.parse_args(normalized_argv)

    for attr in ("https_href", "s3_href", "bbox", "bbox_crs", "out_dir", "out_name"):
        setattr(args, attr, _normalize_blank(getattr(args, attr, "")))

    if not args.out_name:
        args.out_name = "nisar_subset.zarr"

    return args


def _earthaccess_available_noninteractive() -> bool:
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return True
    netrc_path = os.environ.get("NETRC") or os.path.expanduser("~/.netrc")
    return os.path.exists(netrc_path)


def _login_earthaccess_noninteractive():
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return earthaccess.login(strategy="environment")
    netrc_path = os.environ.get("NETRC") or os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        return earthaccess.login()
    raise RuntimeError(
        "No non-interactive Earthaccess credentials available. "
        "In DPS, use --access_mode s3 unless EDL creds are injected."
    )


def _warn_if_mismatched_hrefs(https_href: str, s3_href: str) -> None:
    if not https_href or not s3_href:
        return
    https_name = os.path.basename(https_href)
    s3_name = os.path.basename(s3_href)
    if https_name != s3_name:
        print("WARNING_MISMATCHED_HREFS:")
        print("  HTTPS file:", https_name)
        print("  S3 file   :", s3_name)
        print("  These hrefs appear to point to different granules.")


def resolve_granule_hrefs(args: argparse.Namespace) -> Tuple[str, str]:
    https_href = args.https_href
    s3_href = args.s3_href

    if https_href or s3_href:
        _warn_if_mismatched_hrefs(https_href, s3_href)
        return https_href, s3_href

    auth = _login_earthaccess_noninteractive()
    _ = auth  # keep for clarity

    results = earthaccess.search_data(
        short_name=args.short_name,
        count=args.count,
        cloud_hosted=True,
    )
    if not results:
        raise RuntimeError("No granules found. Provide --https_href and/or --s3_href explicitly.")

    if args.granule_index >= len(results):
        raise IndexError(
            f"--granule_index {args.granule_index} is out of range for {len(results)} result(s)."
        )

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


def transform_bbox_if_needed(
    bbox: Optional[Tuple[float, float, float, float]],
    bbox_crs: str,
    ds_crs_epsg,
) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None

    if not bbox_crs or ds_crs_epsg in (None, "", 0):
        return bbox

    src = CRS.from_user_input(bbox_crs)
    dst = CRS.from_user_input(f"EPSG:{int(ds_crs_epsg)}")

    if src == dst:
        return bbox

    minx, miny, maxx, maxy = bbox
    transformer = Transformer.from_crs(src, dst, always_xy=True)

    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    tx, ty = transformer.transform(xs, ys)

    return (min(tx), min(ty), max(tx), max(ty))


def bbox_overlaps_extent(
    bbox: Tuple[float, float, float, float],
    x: np.ndarray,
    y: np.ndarray,
) -> bool:
    minx, miny, maxx, maxy = bbox
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    x_overlap = not (maxx < xmin or minx > xmax)
    y_overlap = not (maxy < ymin or miny > ymax)
    return x_overlap and y_overlap


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
        raise RuntimeError(
            "BBox produced an empty slice after overlap check. "
            "Make sure it is in the same CRS/units as xCoordinates/yCoordinates."
        )

    return slice(y0, y1), slice(x0, x1)


def _get_s3_credentials(asf_s3_creds_url: str) -> dict:
    env_key = os.environ.get("AWS_ACCESS_KEY_ID")
    env_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    env_token = os.environ.get("AWS_SESSION_TOKEN")

    if env_key and env_secret and env_token:
        print("USING_S3_CREDS_SOURCE: environment")
        return {
            "accessKeyId": env_key,
            "secretAccessKey": env_secret,
            "sessionToken": env_token,
        }

    from maap.maap import MAAP

    maap = MAAP()
    creds = maap.aws.earthdata_s3_credentials(asf_s3_creds_url)
    print("USING_S3_CREDS_SOURCE: maap")
    return creds


def _download_https_to_tempfile(https_href: str) -> Tuple[str, str, str]:
    if not https_href:
        raise RuntimeError("HTTPS href not available.")

    auth = _login_earthaccess_noninteractive()
    session = auth.get_session()

    with session.get(https_href, stream=True, allow_redirects=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            temp_path = tmp.name

    print("OPENING_SOURCE_MODE: https")
    print("DOWNLOADED_TEMP_FILE:", temp_path)
    return temp_path, "https", https_href


def _download_s3_to_tempfile(s3_href: str, asf_s3_creds_url: str) -> Tuple[str, str, str]:
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        temp_path = tmp.name

    fs.get(s3_href, temp_path)

    print("OPENING_SOURCE_MODE: s3")
    print("DOWNLOADED_TEMP_FILE:", temp_path)
    return temp_path, "s3", s3_href


def open_file_like(
    access_mode: str,
    https_href: str,
    s3_href: str,
    asf_s3_creds_url: str,
) -> Tuple[str, str, str]:
    if access_mode == "https":
        return _download_https_to_tempfile(https_href)

    if access_mode == "s3":
        return _download_s3_to_tempfile(s3_href, asf_s3_creds_url)

    if s3_href:
        try:
            return _download_s3_to_tempfile(s3_href, asf_s3_creds_url)
        except Exception as exc:
            print(f"S3_OPEN_FAILED: {exc}")

    if https_href and _earthaccess_available_noninteractive():
        return _download_https_to_tempfile(https_href)

    raise RuntimeError(
        "Could not open granule. In DPS, prefer S3 or inject non-interactive EDL credentials for HTTPS."
    )


def build_dataset(
    h5f: h5py.File,
    group: str,
    x_path: str,
    y_path: str,
    var_names: List[str],
    bbox: Optional[Tuple[float, float, float, float]],
    bbox_crs: str = "",
) -> xr.Dataset:
    x = h5f[x_path][()]
    y = h5f[y_path][()]

    attrs = {}
    ds_epsg = None
    proj_path = f"{group}/projection"
    if proj_path in h5f:
        proj = h5f[proj_path]
        for key in ("epsg_code", "spatial_ref", "grid_mapping_name"):
            if key in proj.attrs:
                value = proj.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode()
                attrs[key] = value
        ds_epsg = attrs.get("epsg_code")

    print("DATASET_X_MINMAX:", float(np.min(x)), float(np.max(x)))
    print("DATASET_Y_MINMAX:", float(np.min(y)), float(np.max(y)))
    print("DATASET_EPSG:", ds_epsg)
    print("INPUT_BBOX:", bbox)
    print("INPUT_BBOX_CRS:", bbox_crs)

    bbox_in_ds_crs = transform_bbox_if_needed(bbox, bbox_crs, ds_epsg)
    print("BBOX_IN_DATASET_CRS:", bbox_in_ds_crs)

    if bbox_in_ds_crs is not None:
        if not bbox_overlaps_extent(bbox_in_ds_crs, x, y):
            raise RuntimeError(
                "BBox does not overlap granule extent. "
                f"Dataset EPSG={ds_epsg}, "
                f"x_range=({float(np.min(x))}, {float(np.max(x))}), "
                f"y_range=({float(np.min(y))}, {float(np.max(y))}), "
                f"input_bbox={bbox}, input_bbox_crs={bbox_crs}, "
                f"bbox_in_dataset_crs={bbox_in_ds_crs}"
            )

        yslice, xslice = bbox_to_slices(x, y, bbox_in_ds_crs)
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

    local_path, chosen_mode, chosen_href = open_file_like(
        args.access_mode,
        https_href,
        s3_href,
        args.asf_s3_creds_url,
    )

    try:
        with h5py.File(
            local_path,
            "r",
            rdcc_nbytes=4 * 1024 * 1024,
            page_buf_size=16 * 1024 * 1024,
        ) as h5f:
            ds = build_dataset(
                h5f,
                args.group,
                args.x_path,
                args.y_path,
                var_names,
                bbox,
                args.bbox_crs,
            )
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    ds.attrs.update(
        {
            "source_href": chosen_href,
            "access_mode": chosen_mode,
            "group": args.group,
            "vars": ",".join(var_names),
            "bbox": args.bbox,
            "bbox_crs": args.bbox_crs,
            "note": "bbox is transformed to dataset CRS when bbox_crs is provided.",
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
