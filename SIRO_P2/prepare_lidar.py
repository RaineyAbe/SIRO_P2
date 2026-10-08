#!/usr/bin/env python

"""
Aggregate high-resolution LiDAR snow depth rasters onto common coarse grids.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Usage:
    python -m SIRO_P2.prepare_lidar \\
    --lidar_dir /path/to/SNEX_MCS_Lidar \\
    --out_dir /path/to/output
"""

import argparse
import os
import re
from glob import glob
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Aggregate LiDAR snow depth rasters onto common grids and save as netCDF.")
    parser.add_argument("--lidar_dir", required=True, type=str, help="Path to directory containing LiDAR snow depth files.")
    parser.add_argument("--pattern", default="*_SD_*.tif", type=str, help="Glob pattern for LiDAR snow depth files in lidar_dir.")
    parser.add_argument("--out_dir", required=True, type=str, help="Path where the output netCDF files will be saved.")
    parser.add_argument("--resolutions", default=[100, 2000], type=float, nargs="+", help="Target grid resolutions in meters.")
    parser.add_argument("--min_coverage", default=0.5, type=float, help="Minimum fraction of valid fine pixels for a cell to be kept.")
    parser.add_argument("--min_depth", default=0.0, type=float, help="Fine-resolution depths below this value [m] are set to NaN.")
    parser.add_argument("--max_depth", default=10.0, type=float, help="Fine-resolution depths above this value [m] are set to NaN.")
    return parser.parse_args()


def get_date(path):
    """Flight date (YYYYMMDD) from a LiDAR filename, as datetime64."""
    match = re.search(r"_(\d{8})_", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not find a YYYYMMDD date in {os.path.basename(path)}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def define_grid(fine_profile, resolutions):
    """
    Common target grid bounds covering the LiDAR extent, snapped outward to multiples of
    the coarsest resolution so that every target grid nests exactly within the next.
    Also checks that each target cell contains a whole number of fine pixels.
    """
    fine_res = fine_profile["transform"].a
    bounds = rio.transform.array_bounds(fine_profile["height"], fine_profile["width"], fine_profile["transform"])
    coarsest = max(resolutions)
    left, bottom = np.floor(bounds[0] / coarsest) * coarsest, np.floor(bounds[1] / coarsest) * coarsest
    right, top = np.ceil(bounds[2] / coarsest) * coarsest, np.ceil(bounds[3] / coarsest) * coarsest

    for res in resolutions:
        factor = res / fine_res
        col_offset = (bounds[0] - left) / fine_res
        row_offset = (top - bounds[3]) / fine_res
        if not (np.isclose(factor, round(factor)) and np.isclose(col_offset, round(col_offset))
                and np.isclose(row_offset, round(row_offset))):
            raise ValueError(f"The {fine_res} m LiDAR grid does not nest within a {res} m grid; "
                             "exact block averaging is not possible.")
    return left, bottom, right, top


def aggregate_lidar(path, grid_bounds, resolution, min_depth, max_depth):
    """
    Sum and count valid fine-resolution snow depths within each cell of a target grid.
    Reads one target row at a time to limit memory use. Returns (sum, count, n_fine_per_cell).
    """
    left, bottom, right, top = grid_bounds
    ny, nx = int(round((top - bottom) / resolution)), int(round((right - left) / resolution))

    with rio.open(path) as src:
        fine_res = src.transform.a
        factor = int(round(resolution / fine_res))                        # fine pixels per cell side
        col_offset = int(round((src.bounds.left - left) / fine_res))      # fine columns before the raster
        row_offset = int(round((top - src.bounds.top) / fine_res))        # fine rows above the raster
        n_fine_cols = nx * factor

        depth_sum = np.zeros((ny, nx), dtype="float64")
        depth_count = np.zeros((ny, nx), dtype="int64")
        for i in range(ny):
            # Fine rows covered by target row i, limited to the raster
            r0 = max(i * factor - row_offset, 0)
            r1 = min((i + 1) * factor - row_offset, src.height)
            if r1 <= r0:
                continue
            fine = src.read(1, window=rio.windows.Window(0, r0, src.width, r1 - r0), masked=True).filled(np.nan)
            valid = np.isfinite(fine) & (fine >= min_depth) & (fine <= max_depth)

            # Pad columns so the row spans the full target grid, then sum within each cell
            padded_depth = np.zeros((r1 - r0, n_fine_cols), dtype="float64")
            padded_valid = np.zeros((r1 - r0, n_fine_cols), dtype=bool)
            padded_depth[:, col_offset:col_offset + src.width] = np.where(valid, fine, 0)
            padded_valid[:, col_offset:col_offset + src.width] = valid
            depth_sum[i] = padded_depth.reshape(r1 - r0, nx, factor).sum(axis=(0, 2))
            depth_count[i] = padded_valid.reshape(r1 - r0, nx, factor).sum(axis=(0, 2))

    return depth_sum, depth_count, factor ** 2


def to_dataset(depth_sum, depth_count, n_fine, grid_bounds, resolution, dates, crs, min_coverage):
    """
    Build a (time, y, x) dataset of mean snow depth and valid fraction for one resolution.
    """
    left, bottom, right, top = grid_bounds
    x = np.arange(left + resolution / 2, right, resolution)
    y = np.arange(top - resolution / 2, bottom, -resolution)

    valid_fraction = depth_count / n_fine
    with np.errstate(invalid="ignore", divide="ignore"):
        snow_depth = depth_sum / depth_count
    snow_depth[valid_fraction < min_coverage] = np.nan
    snow_depth[snow_depth < 0] = 0

    ds = xr.Dataset(
        {
            "snow_depth": (
                ("time", "y", "x"), snow_depth.astype("float32"), 
                {"long_name": "Mean LiDAR snow depth", "units": "m"}
                ),
            "valid_fraction": (
                ("time", "y", "x"), valid_fraction.astype("float32"), 
                {"long_name": "Fraction of fine-resolution pixels with valid snow depth", "units": "1"}
                ),
        },
        coords={"time": dates, "y": y, "x": x},
    )
    ds = ds.rio.write_crs(crs)
    ds.attrs.update({
        "description": f"LiDAR snow depth averaged onto a {resolution:g} m grid",
        "resolution_m": resolution,
        "min_coverage": min_coverage,
    })
    return ds


def main():
    args = get_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    resolutions = sorted(args.resolutions)

    # Get all the LiDAR files, sorted by date
    lidar_files = sorted(glob(os.path.join(args.lidar_dir, args.pattern)), key=get_date)
    if not lidar_files:
        raise FileNotFoundError(f"No files matching '{args.pattern}' in {args.lidar_dir}")
    dates = [get_date(f) for f in lidar_files]
    print(f"Found {len(lidar_files)} LiDAR files from {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")

    # Create a common grid covering all files
    with rio.open(lidar_files[0]) as src:
        fine_profile, crs = src.profile, src.crs
    for f in lidar_files[1:]:
        with rio.open(f) as src:
            if src.crs != crs or src.transform != fine_profile["transform"] or src.shape != (fine_profile["height"], fine_profile["width"]):
                raise ValueError(f"{os.path.basename(f)} is not on the same grid as {os.path.basename(lidar_files[0])}")
    grid_bounds = define_grid(fine_profile, resolutions)
    print(f"Target grid bounds (left, bottom, right, top): {grid_bounds}")

    # Aggregate each file to the finest target grid, then sum those cells up to coarser grids
    finest = resolutions[0]
    sums, counts = [], []
    for f in tqdm(lidar_files, desc=f"Aggregating to {finest:g} m"):
        depth_sum, depth_count, n_fine = aggregate_lidar(f, grid_bounds, finest, args.min_depth, args.max_depth)
        sums.append(depth_sum)
        counts.append(depth_count)
    sums, counts = np.stack(sums), np.stack(counts)

    for res in resolutions:
        block = int(round(res / finest))
        nt, ny, nx = sums.shape
        res_sum = sums.reshape(nt, ny // block, block, nx // block, block).sum(axis=(2, 4))
        res_count = counts.reshape(nt, ny // block, block, nx // block, block).sum(axis=(2, 4))
        ds = to_dataset(res_sum, res_count, n_fine * block ** 2, grid_bounds, res, dates, crs, args.min_coverage)
        ds.attrs.update({
            "source_files": ", ".join(os.path.basename(f) for f in lidar_files),
            "fine_depth_filter_m": f"[{args.min_depth}, {args.max_depth}]",
        })
        out_file = os.path.join(args.out_dir, f"lidar_snow_depth_{res:g}m.nc")
        encoding = {v: {**ds[v].encoding, "zlib": True, "complevel": 4} for v in ds.data_vars}  # keeps grid_mapping
        ds.to_netcdf(out_file, encoding=encoding)
        n_valid = int(np.isfinite(ds.snow_depth).sum(dim=["x", "y"]).median())
        print(f"Saved {out_file} ({ds.sizes['y']} x {ds.sizes['x']} cells, median {n_valid} valid cells per date)")


if __name__ == "__main__":
    main()
