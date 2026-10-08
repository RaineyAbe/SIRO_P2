#!/usr/bin/env python

"""
Compile raw model outputs into one netCDF per model, so unit conversions and date handling are done once.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

For each model in MODELS, every daily snow depth and SWE snapshot for each task is read from model_dir,
converted to meters, and written on the model's native grid to:

    <out_dir>/<model>.nc     variables snow_depth and SWE [m], dimensions (time, task, y, x)

Time is the calendar date of each snapshot, treating each snapshot as the snow state at the end of that
day (HMS: 00:00 the next day or 24:00 the same day; iSnobal: 23:00; SnowModel: 12:00). Dates missing for
one task or variable are NaN. The CRS is stored in the spatial_ref variable, so the files open with
rioxarray (xr.open_dataset(path, decode_coords="all")). Use open_model_outputs() to load them.

Usage:
    python -m SIRO_P2.prepare_models \\
    --model_dir /path/to/raw_model_outputs \\
    --out_dir /path/to/model_outputs \\
    --start_date 2022-10-01 \\
    --end_date 2025-06-30
"""

import argparse
import os
import re
import warnings
from functools import lru_cache
from glob import glob

import dask
import dask.array as darr
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

# ----- SETTINGS -----
# Raw model outputs within model_dir, per model and variable:
#   files     glob pattern (use {task} for the task number, or a {task: pattern} dict)
#   units     units of the raw values (for netCDFs, the variable's "units" attribute takes precedence)
#   variable  variable name, for netCDF inputs
#   crs       optional CRS to assume if the files have none
MODELS = {
    "HMS-EB": {
        "snow_depth": {
            "files": "HMS_Task{task}/EB/snow_depth_tif/*.tif*",
            "units": "in"
            },
        "SWE": {
            "files": "HMS_Task{task}/EB/swe_tif/*.tif*",
            "units": "in"
            },
    },
    "HMS-TI": {
        "snow_depth": {
            "files": "HMS_Task{task}/TI/snow_depth_tif/*.tif*",
            "units": "in"
            },
        "SWE": {
            "files": "HMS_Task{task}/TI/swe_tif/*.tif*",
            "units": "in"
            },
    },
    "iSnobal": {
        "snow_depth": {
            "files": {
                1: "m3w_isnobal_task1/m3w_isnobal_task1_depth/wy*/*.tif",
                2: "m3w_isnobal_task_2_all_updated/m3w_isnobal_task2_depth/wy*/*.tif",
            },
            "units": "m",
        },
        "SWE": {
            "files": {
                1: "m3w_isnobal_task1/m3w_isnobal_task1_SWE/wy*/*.tif",
                2: "m3w_isnobal_task_2_all_updated/m3w_isnobal_task2_SWE/wy*/*.tif",
            },
            "units": "mm",
        },
    },
    "SnowModel": {
        "snow_depth": {
            "files": "SnowModel/SnowModel_WY*_Task{task}.nc",
            "variable": "SD",
            "units": "m"
            },
        "SWE": {
            "files": "SnowModel/SnowModel_WY*_Task{task}.nc",
            "variable": "SWE",
            "units": "mm"
            },
    },
}
TASKS = {1: "Baseline", 2: "Assimilation"}
VARIABLES = {
    "snow_depth": "Snow depth",
    "SWE": "Snow water equivalent"
    }
TO_METERS = {
    "in": 0.0254,
    "inches": 0.0254,
    "mm": 0.001,
    "millimeters": 0.001,
    "m": 1.0,
    "meters": 1.0,
}
TIME_CHUNK = 32         # days per chunk in the output files
SPATIAL_CHUNK = 256     # pixels per chunk side in the output files


def get_parser():
    parser = argparse.ArgumentParser(description="Compile raw model outputs into one netCDF per model.")
    parser.add_argument("--model_dir", required=True, type=str, help="Directory of raw model outputs (see MODELS).")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory where the model netCDFs will be saved.")
    parser.add_argument("--start_date", default=None, type=str, help="First date to include (default: all).")
    parser.add_argument("--end_date", default=None, type=str, help="Last date to include (default: all).")
    parser.add_argument("--models", default=list(MODELS), type=str, nargs="+", choices=list(MODELS), help="Models to prepare (default: all).")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild model netCDFs that already exist.")
    return parser.parse_args()


# ----- SNAPSHOT DATES -----
def snapshot_date(timestamp):
    """
    Calendar date of a model snapshot, treating each snapshot as the snow state at the end of that day.
    """
    return (pd.Timestamp(timestamp) - pd.Timedelta(minutes=1)).normalize()


def file_timestamp(path):
    """
    Last date-time in a filename (e.g., the end of an HMS interval). Handles 24:00 and end times written
    without a year, as in HMS-TI files (TI_snow_depth_2025_05_01T0000_05_01T2400.tif -> 2025-05-02 00:00).
    """
    name = os.path.basename(path)
    stamps = re.findall(r"(?:(\d{4})[-_])?(\d{2})[-_](\d{2})T(\d{2})[-_:]?(\d{2})", name)
    year, last = None, None
    for y, mo, d, h, mi in stamps:
        if y:
            year = int(y)
        elif year is None:
            continue
        elif last is not None and int(mo) < last.month:  # year rolled over within the interval
            year += 1
        last = pd.Timestamp(year, int(mo), int(d)) + pd.Timedelta(hours=int(h), minutes=int(mi))
    if last is None:
        raise ValueError(f"No date-time found in {name}")
    return last


def find_snapshots(model_dir, spec, task, start_date=None, end_date=None):
    """
    {date: (path, time step or None, units)} for one model variable and task.
    """
    pattern = spec["files"][task] if isinstance(spec["files"], dict) else spec["files"].format(task=task)
    paths = sorted(p for p in glob(os.path.join(model_dir, pattern)) if p.endswith((".tif", ".tiff", ".nc")))
    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None

    snapshots = {}
    def add(date, path, t, units):
        if (start is not None and date < start) or (end is not None and date > end):
            return
        if date in snapshots:
            warnings.warn(
                f"More than one snapshot for {date:%Y-%m-%d} ({os.path.basename(snapshots[date][0])}, "
                f"{os.path.basename(path)}); using the first."
            )
            return
        snapshots[date] = (path, t, units)

    for path in paths:
        if path.endswith(".nc"):
            with xr.open_dataset(path) as ds:
                if spec["variable"] not in ds:
                    warnings.warn(f"{os.path.basename(path)} has no variable '{spec['variable']}', skipping.")
                    continue
                units = ds[spec["variable"]].attrs.get("units", spec["units"])
                if units != spec["units"]:
                    warnings.warn(
                        f"{os.path.basename(path)}: {spec['variable']} units are '{units}', "
                        f"not '{spec['units']}'; using '{units}'."
                    )
                for t in ds.time.values:
                    add(snapshot_date(t), path, t, units)
        else:
            add(snapshot_date(file_timestamp(path)), path, None, spec["units"])

    for date, (path, _, units) in snapshots.items():
        if units not in TO_METERS:
            raise ValueError(f"Unknown units '{units}' for {path}")
    return snapshots


# ----- READING -----
@lru_cache(maxsize=8)
def _open_nc(path):
    return xr.open_dataset(path, decode_coords="all")


def read_snapshot(path, t=None, variable=None, default_crs=None):
    """
    One snapshot as a north-up (y, x) DataArray in its raw units, with NaN as nodata.
    """
    if t is None:
        da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    else:
        da = _open_nc(path)[variable].sel(time=t)
    da = da.squeeze(drop=True)
    y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
    if (y_dim, x_dim) != ("y", "x"):
        da = da.rename({y_dim: "y", x_dim: "x"})
    if da.y.values[0] < da.y.values[-1]:
        da = da.isel(y=slice(None, None, -1))
    if da.rio.crs is None:
        if default_crs is None:
            raise ValueError(f"{os.path.basename(path)} has no CRS. Set 'crs' for this model in MODELS.")
        da = da.rio.write_crs(default_crs)
    return da.astype("float32").transpose("y", "x")


def _read_values(path, t, variable, scale, default_crs, y, x):
    """
    Snapshot values [m], checked against the reference grid.
    """
    da = read_snapshot(path, t, variable, default_crs)
    if da.shape != (len(y), len(x)) or not (np.allclose(da.y, y) and np.allclose(da.x, x)):
        raise ValueError(f"{os.path.basename(path)} is not on the same grid as the other files for this model.")
    values = da.values * np.float32(scale)
    values[~np.isfinite(values)] = np.nan
    return values


# ----- MAIN WORKFLOW -----
def prepare_model(model, model_dir, out_file, start_date=None, end_date=None, tasks=None):
    """
    Compile one model's snapshots into out_file. Returns out_file, or None if no files were found.
    """
    spec = MODELS[model]
    tasks = list(tasks or TASKS)
    snapshots = {}
    for var, var_spec in spec.items():
        for task in tasks:
            snaps = find_snapshots(model_dir, var_spec, task, start_date, end_date)
            print(f"{model} {var} Task {task}: {len(snaps)} snapshots")
            if snaps:
                snapshots[(var, task)] = snaps
    if not snapshots:
        warnings.warn(f"{model}: no files found in {model_dir}, skipping.")
        return None

    dates = pd.DatetimeIndex(sorted(set().union(*snapshots.values())))
    tasks = [t for t in tasks if any(task == t for _, task in snapshots)]
    variables = [v for v in spec if any(var == v for var, _ in snapshots)]

    # Reference grid from the first snapshot
    first_var, first_task = next(iter(snapshots))
    path, t, _ = next(iter(snapshots[(first_var, first_task)].values()))
    default_crs = next((s.get("crs") for s in spec.values() if s.get("crs")), None)
    ref = read_snapshot(path, t, spec[first_var].get("variable"), default_crs)
    y, x, crs = ref.y.values, ref.x.values, ref.rio.crs
    shape = (len(y), len(x))
    print(f"{model}: {len(dates)} dates from {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}, tasks {tasks}, "
          f"{shape[0]} x {shape[1]} cells ({abs(ref.rio.resolution()[0]):g} m), CRS {crs}")

    # Lazy (time, task, y, x) arrays, one delayed read per snapshot
    data_vars = {}
    for var in variables:
        by_time = []
        for date in dates:
            by_task = []
            for task in tasks:
                snap = snapshots.get((var, task), {}).get(date)
                if snap is None:
                    by_task.append(darr.full(shape, np.nan, dtype="float32"))
                    continue
                path, t, units = snap
                values = dask.delayed(_read_values)(
                    path, t, spec[var].get("variable"), TO_METERS[units], default_crs, y, x
                )
                by_task.append(darr.from_delayed(values, shape, dtype="float32"))
            by_time.append(darr.stack(by_task))
        stacked = darr.stack(by_time).rechunk((TIME_CHUNK, 1, -1, -1))
        data_vars[var] = (
            ("time", "task", "y", "x"), stacked,
            {"long_name": VARIABLES[var], "units": "m"}
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": dates,
            "task": tasks,
            "y": y,
            "x": x
            }
        )
    ds = ds.rio.write_crs(crs)
    ds["task"].attrs["conditions"] = ", ".join(f"{t} = {TASKS.get(t, '')}" for t in tasks)
    ds["time"].attrs["description"] = "Calendar date of each snapshot (snow state at the end of the day)"
    ds.attrs.update({
        "title": f"{model} snow depth and SWE",
        "source": f"Compiled from {model_dir} by prepare_model_outputs.py",
        "institution": "USACE-ERDC-CRREL, Snow-Informed Reservoir Operations (SIRO)",
    })
    chunks = (min(TIME_CHUNK, len(dates)), 1, min(SPATIAL_CHUNK, shape[0]), min(SPATIAL_CHUNK, shape[1]))
    for var in variables:
        ds[var].attrs.pop("grid_mapping", None)
        ds[var].encoding.update({
            "zlib": True, 
            "complevel": 4, 
            "chunksizes": chunks, 
            "_FillValue": np.float32(np.nan),
            "grid_mapping": "spatial_ref",  # keeps the CRS readable with decode_coords="all"
        })

    tmp_file = out_file + ".tmp"
    with dask.config.set(scheduler="synchronous"):  # netCDF/HDF5 reads are not thread safe
        ds.to_netcdf(tmp_file, engine="netcdf4")
    os.replace(tmp_file, out_file)
    _open_nc.cache_clear()
    print(f"{model}: saved {out_file}")
    return out_file


def prepare_model_outputs(model_dir, out_dir, start_date=None, end_date=None, models=None, overwrite=False):
    """
    Compile every model in MODELS (or the given models) into <out_dir>/<model>.nc.
    Existing files are kept unless overwrite is True.
    """
    os.makedirs(out_dir, exist_ok=True)
    for model in models or MODELS:
        out_file = os.path.join(out_dir, f"{model}.nc")
        if os.path.exists(out_file) and not overwrite:
            print(f"{model}: {out_file} already exists, skipping (use --overwrite to rebuild).")
            continue
        prepare_model(model, model_dir, out_file, start_date, end_date)


# ----- LOADING -----
def open_model_outputs(models_dir, variable=None):
    """
    {model: Dataset} for the netCDFs written by prepare_model_outputs(), in MODELS order. If variable is
    given, only models that have it are returned. Raises an error if none are found.
    """
    outputs = {}
    for model in MODELS:
        path = os.path.join(models_dir, f"{model}.nc")
        if not os.path.exists(path):
            continue
        ds = xr.open_dataset(path, decode_coords="all")
        if variable is None or variable in ds:
            outputs[model] = ds
    if not outputs:
        raise FileNotFoundError(
            f"No prepared model outputs{f' with {variable}' if variable else ''} in {models_dir}. "
            "Run prepare_model_outputs.py (pipeline step 0) first."
        )
    return outputs


def main():
    args = get_parser()
    prepare_model_outputs(
        args.model_dir,
        args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        models=args.models,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
