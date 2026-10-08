#!/usr/bin/env python

"""
Compare modeled snow depth with airborne LiDAR snow depth on each LiDAR date.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Steps for each LiDAR date, task, and model:
    1. Load the model snow depth [m] for the LiDAR date from the prepare_model_outputs.py netCDFs.
    2. Clip the model to the basin (all_touched=True) and, separately, to the LiDAR domain
       (all_touched=False).
    3. Resample the LiDAR-domain clip onto the LiDAR grid with nearest neighbour: the 2000 m LiDAR for
       the HMS models, the 100 m LiDAR for iSnobal and SnowModel. A LiDAR pixel is therefore only
       compared if its matched model cell's centre is inside the LiDAR domain too.
    4. Write model - LiDAR differences and compute error metrics (bias, MAE, RMSE, R^2, NSE, ...) over
       pixels valid in both, excluding LiDAR depths >= --max_lidar_depth (default: 5 m).

Inputs:
    lidar_dir   lidar_snow_depth_100m.nc and lidar_snow_depth_2000m.nc from prepare_lidar.py
    models_dir  <model>.nc files from prepare_model_outputs.py

Outputs, in out_dir:
    lidar_comparison_metrics.csv                            one row per date, task, and model
    lidar_comparison_clip_stats.csv                         n_valid/min/mean/max/zeros of every clip
    rasters/<YYYYMMDD>/LiDAR_<res>_lidar_domain_clip.tif
    rasters/<YYYYMMDD>/Task<N>/<model>_basin_clip.tif
    rasters/<YYYYMMDD>/Task<N>/<model>_lidar_domain_clip.tif
    rasters/<YYYYMMDD>/Task<N>/<model>_lidar_resample.tif   model on the LiDAR grid
    rasters/<YYYYMMDD>/Task<N>/<model>_lidar_diff.tif       model - LiDAR
All rasters are snow depth in meters with NaN as nodata. Load them by name with get_lidar_comparison_outputs().

Usage:
    python -m SIRO_P2.compare_lidar \\
    --lidar_dir /path/to/lidar \\
    --models_dir /path/to/model_outputs \\
    --lidar_aoi_file /path/to/lidar_domain_outline.shp \\
    --basin_aoi_file /path/to/basin_outline.shp \\
    --out_dir /path/to/lidar
"""

import argparse
import os
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rioxarray
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from scipy.stats import linregress

from .prepare_models import TASKS, open_model_outputs

# ----- SETTINGS -----
# LiDAR grid each model is compared on
MODEL_LIDAR_RES = {
    "HMS-EB": "2000m",
    "HMS-TI": "2000m",
    "iSnobal": "100m",
    "SnowModel": "100m",
}
# LiDAR snow depth files within lidar_dir (written by prepare_lidar.py)
LIDAR_FILES = {"100m": "lidar_snow_depth_100m.nc", "2000m": "lidar_snow_depth_2000m.nc"}
MAX_LIDAR_DEPTH = 5.0  # m; LiDAR depths at or above this are excluded from the metrics


def get_parser():
    parser = argparse.ArgumentParser(description="Compare modeled snow depth with LiDAR snow depth.")
    parser.add_argument("--lidar_dir", required=True, type=str, help="Directory with the prepare_lidar.py netCDFs.")
    parser.add_argument("--models_dir", required=True, type=str, help="Directory of prepare_model_outputs.py netCDFs.")
    parser.add_argument("--lidar_aoi_file", required=True, type=str, help="Vector file of the LiDAR domain.")
    parser.add_argument("--basin_aoi_file", required=True, type=str, help="Vector file of the basin.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory where outputs will be saved.")
    parser.add_argument("--start_date", default=None, type=str, help="First LiDAR date to compare (default: all).")
    parser.add_argument("--end_date", default=None, type=str, help="Last LiDAR date to compare (default: all).")
    parser.add_argument("--tasks", default=list(TASKS), type=int, nargs="+", help="Tasks to compare (default: 1 2).")
    parser.add_argument(
        "--max_lidar_depth", default=MAX_LIDAR_DEPTH, type=float,
        help=f"Exclude LiDAR depths >= this value [m] from the metrics (default: {MAX_LIDAR_DEPTH:g}). "
             "Use a negative value to disable."
    )
    return parser.parse_args()


# ----- RASTER HELPERS -----
def to_array(da, default_crs=None):
    """
    (data, profile) for a 2D DataArray: float32 snow depth with NaN as nodata, north-up.
    """
    da = da.squeeze(drop=True)
    y_dim = da.rio.y_dim
    if da[y_dim].values[0] < da[y_dim].values[-1]:
        da = da.isel({y_dim: slice(None, None, -1)})
    crs = da.rio.crs
    if crs is None:
        warnings.warn(f"{da.name} has no CRS; assuming {default_crs}.")
        crs = default_crs
    data = da.values.astype("float32")
    data[~np.isfinite(data)] = np.nan
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "crs": crs,
        "transform": da.rio.transform(recalc=True),
        "height": data.shape[0],
        "width": data.shape[1],
        "compress": "lzw",
    }
    return data, profile


def write_raster(path, data, profile):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


def clip(data, profile, aoi, all_touched):
    """
    Clip and crop a raster to an AOI GeoDataFrame. Pixels outside the AOI are set to NaN.
    Returns None if the AOI does not overlap the raster.
    """
    shapes = aoi.to_crs(profile["crs"]).geometry
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        with mem.open() as src:
            try:
                out, transform = rasterio.mask.mask(
                    src, shapes, crop=True, all_touched=all_touched, nodata=np.nan
                )
            except ValueError as e:  # no overlap
                warnings.warn(str(e))
                return None
    profile = profile.copy()
    profile.update(height=out.shape[1], width=out.shape[2], transform=transform)
    return out[0], profile


def resample_to_grid(data, profile, dst_profile):
    """
    Nearest-neighbour resample of a raster onto another grid.
    """
    out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=data, destination=out,
        src_transform=profile["transform"], src_crs=profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
    )
    return out


# ----- LIDAR -----
def load_lidar(lidar_dir, start_date=None, end_date=None):
    """
    {resolution: (time, y, x) snow depth DataArray} from the prepare_lidar.py netCDFs, with time as
    calendar dates, limited to start_date/end_date.
    """
    lidar = {}
    for res, name in LIDAR_FILES.items():
        path = os.path.join(lidar_dir, name)
        if not os.path.exists(path):
            warnings.warn(f"LiDAR file not found, skipping models compared at {res}: {path}")
            continue
        with xr.open_dataset(path, decode_coords="all") as ds:
            da = ds["snow_depth"].load()
        da = da.assign_coords(time=pd.DatetimeIndex(da.time.values).normalize())
        if start_date:
            da = da.sel(time=da.time >= pd.to_datetime(start_date))
        if end_date:
            da = da.sel(time=da.time <= pd.to_datetime(end_date))
        lidar[res] = da
    if not lidar:
        raise FileNotFoundError(f"No LiDAR files ({', '.join(LIDAR_FILES.values())}) found in {lidar_dir}")
    return lidar


# ----- METRICS -----
def calculate_metrics(model, lidar):
    """
    Error metrics for model vs. LiDAR over pixels valid in both (LiDAR as the reference).
    """
    valid = np.isfinite(model) & np.isfinite(lidar)
    m, obs = model[valid].astype("float64"), lidar[valid].astype("float64")
    metrics = {"n_pixels": int(valid.sum())}
    if metrics["n_pixels"] < 2:
        return metrics

    err = m - obs
    reg = linregress(obs, m)
    metrics.update({
        "lidar_mean_m": obs.mean(),
        "model_mean_m": m.mean(),
        "bias_m": err.mean(),
        "MAE_m": np.abs(err).mean(),
        "RMSE_m": np.sqrt((err ** 2).mean()),
        "r_squared": reg.rvalue ** 2,
        "slope": reg.slope,
        "intercept": reg.intercept,
        "pvalue": reg.pvalue,
        "stderr": reg.stderr,
        "NSE": 1 - (err ** 2).sum() / ((obs - obs.mean()) ** 2).sum(),  # Nash-Sutcliffe efficiency
    })
    return metrics


def raster_stats(data):
    valid = data[np.isfinite(data)]
    stats = {"n_valid": valid.size}
    if valid.size:
        stats.update({"min": valid.min(), "mean": valid.mean(), "max": valid.max(), "zeros": int((valid == 0).sum())})
    return stats


# ----- MAIN WORKFLOW -----
def compare_lidar(
    lidar_dir, models_dir, lidar_aoi_file, basin_aoi_file, out_dir,
    start_date=None, end_date=None, tasks=None, max_lidar_depth=MAX_LIDAR_DEPTH
):
    """
    Compare modeled and LiDAR snow depth on every LiDAR date (see module docstring).
    Returns the metrics DataFrame.
    """
    tasks = {t: TASKS.get(t, f"Task{t}") for t in (tasks or TASKS)}
    lidar_aoi = gpd.read_file(lidar_aoi_file)
    basin_aoi = gpd.read_file(basin_aoi_file)
    lidar = load_lidar(lidar_dir, start_date, end_date)
    dates = sorted(set().union(*(set(da.time.values) for da in lidar.values())))
    dates = pd.DatetimeIndex(dates)
    if dates.empty:
        raise ValueError("No LiDAR dates within the requested date range.")
    print(f"LiDAR dates: {', '.join(f'{d:%Y-%m-%d}' for d in dates)}")
    raster_dir = os.path.join(out_dir, "rasters")

    # Prepared model snow depth (read lazily, one date at a time)
    models = {
        m: ds["snow_depth"] for m, ds in open_model_outputs(models_dir, "snow_depth").items() 
        if m in MODEL_LIDAR_RES
        }
    for model, da in models.items():
        print(f"{model}: {sum(d in da.time.values for d in dates)} of {len(dates)} LiDAR dates, tasks {list(da.task.values)}")

    metric_rows, stat_rows = [], []
    for date in dates:
        date_str = f"{date:%Y%m%d}"
        print(f"\nProcessing {date:%Y-%m-%d}...")

        # LiDAR, clipped to the LiDAR domain
        lidar_clips = {}
        for res, da in lidar.items():
            if date not in da.time.values:
                continue
            clipped = clip(*to_array(da.sel(time=date)), lidar_aoi, all_touched=False)
            if clipped is None:
                continue
            lidar_clips[res] = clipped
            write_raster(os.path.join(raster_dir, date_str, f"LiDAR_{res}_lidar_domain_clip.tif"), *clipped)
            stat_rows.append({
                "date": date_str, "task": None, "region": "lidar_domain", "model": f"LiDAR-{res}",
                **raster_stats(clipped[0])
            })
        crs = next(iter(lidar_clips.values()))[1]["crs"] if lidar_clips else None

        for task, condition in tasks.items():
            task_dir = os.path.join(raster_dir, date_str, f"Task{task}")
            for model, model_da in models.items():
                res = MODEL_LIDAR_RES[model]
                if res not in lidar_clips:
                    warnings.warn(f"{date_str} Task {task} {model}: no {res} LiDAR on this date, skipping.")
                    continue
                if date not in model_da.time.values or task not in model_da.task.values:
                    warnings.warn(f"{date_str} Task {task} {model}: no model output on this date, skipping.")
                    continue
                data, profile = to_array(model_da.sel(time=date, task=task).load(), default_crs=crs)
                if np.isnan(data).all():
                    warnings.warn(f"{date_str} Task {task} {model}: model output is all NaN on this date, skipping.")
                    continue

                # Basin and LiDAR-domain clips, each made separately from the full model raster
                basin_clip = clip(data, profile, basin_aoi, all_touched=True)
                domain_clip = clip(data, profile, lidar_aoi, all_touched=False)
                if domain_clip is None:
                    warnings.warn(f"{date_str} Task {task} {model}: model does not overlap the LiDAR domain.")
                    continue
                if basin_clip is not None:
                    write_raster(os.path.join(task_dir, f"{model}_basin_clip.tif"), *basin_clip)
                    stat_rows.append({
                        "date": date_str, "task": task, "region": "basin", "model": model,
                        **raster_stats(basin_clip[0])
                    })
                write_raster(os.path.join(task_dir, f"{model}_lidar_domain_clip.tif"), *domain_clip)
                stat_rows.append({
                    "date": date_str, "task": task, "region": "lidar_domain", "model": model,
                    **raster_stats(domain_clip[0])
                })

                # Model on the LiDAR grid, and model - LiDAR
                lidar_data, lidar_profile = lidar_clips[res]
                resampled = resample_to_grid(*domain_clip, lidar_profile)
                invalid = np.isnan(resampled) | np.isnan(lidar_data)
                resampled[invalid] = np.nan
                diff = np.where(invalid, np.nan, resampled - lidar_data)
                write_raster(os.path.join(task_dir, f"{model}_lidar_resample.tif"), resampled, lidar_profile)
                write_raster(os.path.join(task_dir, f"{model}_lidar_diff.tif"), diff, lidar_profile)

                # Metrics, excluding deep LiDAR pixels
                lidar_for_metrics = lidar_data.copy()
                if max_lidar_depth is not None and max_lidar_depth >= 0:
                    lidar_for_metrics[lidar_for_metrics >= max_lidar_depth] = np.nan
                metric_rows.append({
                    "date": date_str,
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "lidar_resolution": res,
                    **calculate_metrics(resampled, lidar_for_metrics),
                    "model_source": f"{model}.nc",
                })

    metrics = pd.DataFrame(metric_rows)
    metrics_file = os.path.join(out_dir, "lidar_comparison_metrics.csv")
    metrics.to_csv(metrics_file, index=False)
    stats_file = os.path.join(out_dir, "lidar_comparison_clip_stats.csv")
    pd.DataFrame(stat_rows).to_csv(stats_file, index=False)
    print(f"\nMetrics for {len(metrics)} date/task/model combinations saved to: {metrics_file}")
    print(f"Clip statistics saved to: {stats_file}")
    if not metrics.empty:
        print(
            metrics[["date", "condition", "model", "n_pixels", "bias_m", "RMSE_m", "r_squared"]]
            .round(3).to_string(index=False)
        )
    return metrics


# ----- LOADING OUTPUTS -----
def get_lidar_comparison_outputs(out_dir, task):
    """
    Rasters written by compare_lidar() for one task, looked up by exact filename.

    Returns:
        dict: {date: {"lidar": {res: path}, "basin_clip": {model: path}, "lidar_domain_clip": {model: path},
                      "resample": {model: path}, "diff": {model: path}}}
              Missing files are reported and left out.
    """
    raster_dir = os.path.join(out_dir, "rasters")
    outputs = {}
    for date_str in sorted(d for d in os.listdir(raster_dir) if re.fullmatch(r"\d{8}", d)):
        date_dir = os.path.join(raster_dir, date_str)
        task_dir = os.path.join(date_dir, f"Task{task}")
        expected = {
            "lidar": {res: os.path.join(date_dir, f"LiDAR_{res}_lidar_domain_clip.tif") for res in LIDAR_FILES},
            "basin_clip": {m: os.path.join(task_dir, f"{m}_basin_clip.tif") for m in MODEL_LIDAR_RES},
            "lidar_domain_clip": {m: os.path.join(task_dir, f"{m}_lidar_domain_clip.tif") for m in MODEL_LIDAR_RES},
            "resample": {m: os.path.join(task_dir, f"{m}_lidar_resample.tif") for m in MODEL_LIDAR_RES},
            "diff": {m: os.path.join(task_dir, f"{m}_lidar_diff.tif") for m in MODEL_LIDAR_RES},
        }
        found = {k: {name: p for name, p in v.items() if os.path.exists(p)} for k, v in expected.items()}
        missing = [p for v in expected.values() for p in v.values() if not os.path.exists(p)]
        if missing:
            print(f"{date_str} task {task}: {len(missing)} expected outputs missing, e.g. {os.path.basename(missing[0])}")
        outputs[date_str] = found
    return outputs


def main():
    args = get_parser()
    compare_lidar(
        args.lidar_dir, args.models_dir, args.lidar_aoi_file, args.basin_aoi_file, args.out_dir,
        start_date=args.start_date, end_date=args.end_date, tasks=args.tasks,
        max_lidar_depth=args.max_lidar_depth if args.max_lidar_depth >= 0 else None
    )


if __name__ == "__main__":
    main()
