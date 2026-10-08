#!/usr/bin/env python

"""
Cross-check SIRO_P2/compare_lidar.py against the SIRO Phase 1 (P1) outputs for Mores Creek Summit (MCS).

    1. Stack the P1 per-date LiDAR rasters (dates/<YYYYMMDD>/lidar/) into netCDFs in the same format
       prepare_lidar.py writes, so both versions compare against identical LiDAR grids.
    2. Compile the raw model outputs in MCSModeling/ModelOutputs with prepare_model_outputs() (kept in
       --out_dir/model_outputs and reused on later runs), then run compare_lidar() on them.
    3. Compare every new raster with its P1 counterpart (dates/<YYYYMMDD>/outputs/task<N>/rasters/),
       the clip statistics with P1's MCS_stats.csv/basin_stats.csv, and the metrics with the
       make_figures.ipynb regression cell (recomputed from the P1 rasters) and with the saved
       figures/lidar_regression_results.csv.

Writes crosscheck_rasters.csv, crosscheck_stats.csv, and crosscheck_metrics.csv to --out_dir.

Usage:
    python crosscheck_compare_lidar_P1.py \\
    --p1_dir /Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling \\
    --out_dir /Users/rdcrlrka/Research/SIRO/SIRO_P2/crosscheck_MCS
"""

import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 (registers the .rio accessor)
import xarray as xr

from SIRO_P2.compare_lidar import LIDAR_FILES, compare_lidar
from SIRO_P2.prepare_models import prepare_model_outputs

# P1 file tags and the matching compare_lidar model names
P1_MODELS = {"HMS_EB": "HMS-EB", "HMS_TI": "HMS-TI", "iSnobal": "iSnobal", "SnowModel": "SnowModel"}
P1_LIDAR = {"100m": ("*_SD.tif", "LiDAR_MCS_clip.tif", "LiDAR"), "2000m": ("*_SD_2000m*.tif", "LiDAR_2000_MCS_clip.tif", "LiDAR-2000")}
# New output suffix -> P1 output suffix
RASTER_TYPES = {
    "basin_clip": "basin_clip",
    "lidar_domain_clip": "MCS_clip",
    "lidar_resample": "lidar_resample",
    "lidar_diff": "lidar_diff",
}
# Display names used in figures/lidar_regression_results.csv
DISPLAY_NAMES = {"HMS-EB": "HEC-HMS EB", "HMS-TI": "HEC-HMS TI", "iSnobal": "iSnobal", "SnowModel": "SnowModel"}
TASK_CONDITIONS = {1: "Baseline", 2: "Assimilation"}
MAX_LIDAR_DEPTH = 5.0
ATOL = 1e-5  # m


def get_parser():
    parser = argparse.ArgumentParser(description="Cross-check compare_lidar.py against the P1 MCS outputs.")
    parser.add_argument("--p1_dir", required=True, type=str, help="SIRO_P1/MCSModeling directory.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory for the new outputs and reports.")
    parser.add_argument(
        "--skip_run", action="store_true",
        help="Only compare; reuse compare_lidar outputs already in --out_dir."
    )
    return parser.parse_args()


def read(path):
    """
    (data with NaN for nodata, profile)
    """
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = src.profile
    data[~np.isfinite(data)] = np.nan
    data[data == -9999] = np.nan
    return data, profile


def build_lidar_netcdfs(dates_dir, lidar_dir):
    """
    Stack the P1 per-date LiDAR rasters into lidar_snow_depth_<res>.nc (prepare_lidar.py format).
    """
    os.makedirs(lidar_dir, exist_ok=True)
    date_dirs = sorted(d for d in glob(os.path.join(dates_dir, "*")) if re.fullmatch(r"\d{8}", os.path.basename(d)))
    for res, (pattern, _, _) in P1_LIDAR.items():
        arrays, dates, ref = [], [], None
        for date_dir in date_dirs:
            matches = glob(os.path.join(date_dir, "lidar", pattern))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected 1 file matching {pattern} in {date_dir}/lidar, found {matches}")
            with rasterio.open(matches[0]) as src:
                data = src.read(1, masked=True).astype("float32").filled(np.nan)
                grid = (src.crs, src.transform, src.shape)
            if ref is None:
                ref = grid
            elif grid != ref:
                raise ValueError(f"{matches[0]} is not on the same grid as the other {res} LiDAR rasters")
            arrays.append(data)
            dates.append(pd.to_datetime(os.path.basename(date_dir), format="%Y%m%d"))
        crs, transform, (ny, nx) = ref
        x = transform.c + transform.a * (np.arange(nx) + 0.5)
        y = transform.f + transform.e * (np.arange(ny) + 0.5)
        ds = xr.Dataset(
            {"snow_depth": (("time", "y", "x"), np.stack(arrays), {"long_name": "Mean LiDAR snow depth", "units": "m"})},
            coords={"time": dates, "y": y, "x": x},
        ).rio.write_crs(crs)
        out_file = os.path.join(lidar_dir, LIDAR_FILES[res])
        ds.to_netcdf(out_file)
        print(f"Wrote {out_file} ({len(dates)} dates, {ny} x {nx} cells)")


def compare_raster(new_path, old_path):
    row = {"new_exists": os.path.exists(new_path), "p1_exists": os.path.exists(old_path)}
    if not (row["new_exists"] and row["p1_exists"]):
        return row
    new, new_prof = read(new_path)
    old, old_prof = read(old_path)
    row["same_shape"] = new.shape == old.shape
    row["same_transform"] = bool(np.allclose(tuple(new_prof["transform"]), tuple(old_prof["transform"]), atol=1e-6))
    row["n_valid_new"] = int(np.isfinite(new).sum())
    row["n_valid_p1"] = int(np.isfinite(old).sum())
    if row["same_shape"]:
        row["same_valid_mask"] = bool((np.isfinite(new) == np.isfinite(old)).all())
        both = np.isfinite(new) & np.isfinite(old)
        row["max_abs_diff"] = float(np.abs(new[both] - old[both]).max()) if both.any() else np.nan
    row["match"] = bool(
        row["same_shape"] and row["same_transform"] and row.get("same_valid_mask", False)
        and (np.isnan(row.get("max_abs_diff", np.nan)) or row["max_abs_diff"] <= ATOL)
    )
    return row


def p1_metrics(lidar_path, resample_path):
    """
    The make_figures.ipynb regression cell, applied to P1 rasters.
    """
    from scipy.stats import linregress
    lidar, _ = read(lidar_path)
    lidar[lidar >= MAX_LIDAR_DEPTH] = np.nan
    model, _ = read(resample_path)
    mask = np.isfinite(lidar) & np.isfinite(model)
    x, y = lidar[mask], model[mask]
    if len(x) < 2:
        return {"n_pixels": len(x)}
    reg = linregress(x, y)
    return {
        "n_pixels": len(x), 
        "r_squared": reg.rvalue ** 2, 
        "RMSE_m": np.sqrt(np.mean((y - x) ** 2)), 
        "pvalue": reg.pvalue
        }


def main():
    args = get_parser()
    dates_dir = os.path.join(args.p1_dir, "dates")
    lidar_dir = os.path.join(args.out_dir, "lidar_inputs")
    cmp_dir = os.path.join(args.out_dir, "lidar")
    date_strs = sorted(os.path.basename(d) for d in glob(os.path.join(dates_dir, "*")) if re.fullmatch(r"\d{8}", os.path.basename(d)))
    outline_dir = os.path.join(dates_dir, date_strs[0], "MCS_outline")

    if not args.skip_run:
        build_lidar_netcdfs(dates_dir, lidar_dir)
        models_dir = os.path.join(args.out_dir, "model_outputs")
        prepare_model_outputs(
            os.path.join(args.p1_dir, "ModelOutputs"), models_dir, start_date="2022-10-01", end_date="2025-06-30"
        )
        compare_lidar(
            lidar_dir, models_dir,
            os.path.join(outline_dir, "MCS_outline.shp"), os.path.join(outline_dir, "basin_outline.shp"),
            cmp_dir, max_lidar_depth=MAX_LIDAR_DEPTH
        )

    # --- Rasters ---
    rows = []
    for date in date_strs:
        p1_rasters = os.path.join(dates_dir, date, "outputs", "task1", "rasters")
        for res, (_, p1_name, _) in P1_LIDAR.items():
            new = os.path.join(cmp_dir, "rasters", date, f"LiDAR_{res}_lidar_domain_clip.tif")
            rows.append({"date": date, "task": None, "model": f"LiDAR-{res}", "raster": "lidar_domain_clip",
                         **compare_raster(new, os.path.join(p1_rasters, p1_name))})
        for task in TASK_CONDITIONS:
            p1_rasters = os.path.join(dates_dir, date, "outputs", f"task{task}", "rasters")
            for tag, model in P1_MODELS.items():
                for new_type, p1_type in RASTER_TYPES.items():
                    new = os.path.join(cmp_dir, "rasters", date, f"Task{task}", f"{model}_{new_type}.tif")
                    old = os.path.join(p1_rasters, f"{tag}_{p1_type}.tif")
                    rows.append({"date": date, "task": task, "model": model, "raster": new_type, **compare_raster(new, old)})
    rasters = pd.DataFrame(rows)
    rasters.to_csv(os.path.join(args.out_dir, "crosscheck_rasters.csv"), index=False)

    # --- Clip statistics ---
    new_stats = pd.read_csv(os.path.join(cmp_dir, "lidar_comparison_clip_stats.csv"), dtype={"date": str})
    stat_rows = []
    for date in date_strs:
        for task in TASK_CONDITIONS:
            figs = os.path.join(dates_dir, date, "outputs", f"task{task}", "figs")
            for region, p1_file in [("basin", "basin_stats.csv"), ("lidar_domain", "MCS_stats.csv")]:
                path = os.path.join(figs, p1_file)
                if not os.path.exists(path):
                    continue
                for _, old in pd.read_csv(path).iterrows():
                    name = {"LiDAR": "LiDAR-100m", "LiDAR-2000": "LiDAR-2000m"}.get(old["model"], old["model"])
                    sel = new_stats[(new_stats.date == date) & (new_stats.region == region) & (new_stats.model == name)]
                    if name.startswith("LiDAR"):
                        sel = sel.head(1)
                    else:
                        sel = sel[sel.task == task]
                    new = sel.iloc[0] if len(sel) else {}
                    row = {"date": date, "task": task, "region": region, "model": name}
                    for k in ["n_valid", "min", "mean", "max", "zeros"]:
                        row[f"{k}_p1"], row[f"{k}_new"] = old.get(k), new.get(k, np.nan) if len(sel) else np.nan
                    row["match"] = bool(
                        len(sel) and row["n_valid_p1"] == row["n_valid_new"]
                        and np.isclose(row["mean_p1"], row["mean_new"], atol=ATOL)
                    )
                    stat_rows.append(row)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(os.path.join(args.out_dir, "crosscheck_stats.csv"), index=False)

    # --- Metrics ---
    new_metrics = pd.read_csv(os.path.join(cmp_dir, "lidar_comparison_metrics.csv"), dtype={"date": str})
    saved_file = os.path.join(args.p1_dir, "figures", "lidar_regression_results.csv")
    saved = pd.read_csv(saved_file, header=[0, 1], index_col=[0, 1]) if os.path.exists(saved_file) else None
    metric_rows = []
    for date in date_strs:
        for task, condition in TASK_CONDITIONS.items():
            p1_rasters = os.path.join(dates_dir, date, "outputs", f"task{task}", "rasters")
            for tag, model in P1_MODELS.items():
                lidar_name = "LiDAR_2000_MCS_clip.tif" if tag.startswith("HMS") else "LiDAR_MCS_clip.tif"
                old = p1_metrics(
                    os.path.join(dates_dir, date, "outputs", "task1", "rasters", lidar_name),
                    os.path.join(p1_rasters, f"{tag}_lidar_resample.tif")
                )
                sel = new_metrics[(new_metrics.date == date) & (new_metrics.task == task) & (new_metrics.model == model)]
                new = sel.iloc[0] if len(sel) else None
                row = {"date": date, "task": task, "model": model}
                for k in ["n_pixels", "r_squared", "RMSE_m", "pvalue"]:
                    row[f"{k}_p1"] = old.get(k, np.nan)
                    row[f"{k}_new"] = new[k] if new is not None else np.nan
                if saved is not None:
                    key = (f"{date[:4]}-{date[4:6]}-{date[6:]}", DISPLAY_NAMES[model])
                    if key in saved.index:
                        row["r_squared_p1_saved"] = saved.loc[key, (condition, "r-squared")]
                        row["RMSE_m_p1_saved"] = saved.loc[key, (condition, "RMSE")]
                row["model_source"] = new["model_source"] if new is not None else None
                row["match"] = bool(
                    new is not None and row["n_pixels_p1"] == row["n_pixels_new"]
                    and np.isclose(row["r_squared_p1"], row["r_squared_new"], atol=1e-6)
                    and np.isclose(row["RMSE_m_p1"], row["RMSE_m_new"], atol=1e-6)
                )
                metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(args.out_dir, "crosscheck_metrics.csv"), index=False)

    # --- Summary ---
    pd.set_option("display.width", 250)
    print("\n----- CROSS-CHECK SUMMARY -----")
    print(f"Rasters: {int(rasters['match'].fillna(False).sum())} of {len(rasters)} match P1")
    print(f"Clip stats: {int(stats['match'].sum())} of {len(stats)} match P1")
    print(f"Metrics: {int(metrics['match'].sum())} of {len(metrics)} match P1 (recomputed from P1 rasters)")
    bad = rasters[~rasters["match"].fillna(False).astype(bool)]
    if len(bad):
        print("\nRasters that differ:")
        cols = [c for c in ["date", "task", "model", "raster", "new_exists", "p1_exists", "same_shape", "same_transform",
                            "same_valid_mask", "n_valid_new", "n_valid_p1", "max_abs_diff"] if c in bad]
        print(bad[cols].to_string(index=False))
    bad = metrics[~metrics["match"]]
    if len(bad):
        print("\nMetrics that differ:")
        print(bad.drop(columns="match").round(4).to_string(index=False))
    print(f"\nReports written to {args.out_dir}")


if __name__ == "__main__":
    main()
