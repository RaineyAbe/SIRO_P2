#!/usr/bin/env python

"""
Compare modeled snow-covered area (SCA) with SPIReS fractional snow-covered area (fSCA).

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Steps:
    1. Compile the daily SPIReS fSCA files into one time series. 
       Full-tile files that have not been clipped yet are clipped here.
    2. For each model, task, and SWE threshold, convert modeled SWE [m] (from the prepare_model_outputs.py
       netCDFs) to binary SCA (SWE > threshold)
       and put it on the SPIReS grid. 100 m models are averaged into SPIReS pixels (giving fSCA),
       2000 m models use the nearest model cell (binary SCA).
    3. Compare with SPIReS: fSCA error metrics for the 100 m models, partial-credit error for the 2000 m
       models, both basin-wide and binned by elevation and aspect (if a DEM is given).
    4. Compare melt-out dates (last day with fSCA >= 10% in each water year) at pixels SPIReS observed as
       snow covered.

Outputs, in --out_dir
    SPIReS_fSCA.nc                              SPIReS fSCA time series [%]
    fSCA_regridded_<model>.nc                   modeled fSCA on the SPIReS grid (task, SWE_threshold_m, time, y, x)
    compiled_performance_metrics.nc             basin-wide metrics (model, task, SWE_threshold_m, time)
    compiled_metrics_binned_elev_aspect.nc      metrics binned by elevation and aspect (requires --dem_file)
    compiled_melt_out_metrics.nc                melt-out date metrics (model, task, SWE_threshold_m, water_year)
    compiled_melt_out_binned_elev_aspect.nc     melt-out metrics binned by elevation and aspect (requires --dem_file)
    melt_out_dates_gridded.nc                   per-pixel melt-out day of water year

Usage:
    python -m SIRO_P2.compare_fSCA \\
    --spires_dir /path/to/SCA/SPIReS \\
    --models_dir /path/to/model_outputs \\
    --aoi_file /path/to/watershed_outline.gpkg \\
    --out_dir /path/to/SCA \\
    --dem_file /path/to/DEM.tif
"""

import argparse
import os
import re
import warnings
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import xrspatial
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from tqdm import tqdm

from .download_fSCA import clip_image_to_aoi, clipped_path
from .prepare_models import open_model_outputs

# ----- SETTINGS -----
# fine = True: 100 m models, averaged into SPIReS pixels as fSCA; False: 2000 m models, nearest cell as binary SCA
MODELS = {
    "HMS-EB": {"fine": False},
    "HMS-TI": {"fine": False},
    "iSnobal": {"fine": True},
    "SnowModel": {"fine": True},
}
TASKS = [1, 2]
SWE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]  # m; a pixel is snow covered if SWE > threshold
MELT_OUT_FSCA_THRESH = 0.10     # fSCA at or above which a pixel counts as snow covered for melt-out dates
WATER_YEAR_START_MONTH = 10     # inclusive
ELEV_BIN_WIDTH = 100            # m
ASPECT_BINS = np.arange(0, 361, 45)


def get_parser():
    parser = argparse.ArgumentParser(description="Compare modeled SCA with SPIReS fSCA.")
    parser.add_argument("--spires_dir", required=True, type=str, help="Directory of downloaded SPIReS files.")
    parser.add_argument("--models_dir", required=True, type=str, help="Directory of prepare_model_outputs.py netCDFs.")
    parser.add_argument("--aoi_file", required=True, type=str, help="Vector file of the area of interest.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory where outputs will be saved.")
    parser.add_argument("--start_date", default=None, type=str, help="First date to compare (default: all SPIReS dates).")
    parser.add_argument("--end_date", default=None, type=str, help="Last date to compare (default: all SPIReS dates).")
    parser.add_argument("--dem_file", default=None, type=str,
                        help="DEM GeoTIFF for elevation/aspect-binned metrics (skipped if not given).")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Recompute the regridded model fSCA files even if they exist.")
    return parser.parse_args()


# ----- SPIReS -----
def load_spires(spires_dir, aoi, start_date=None, end_date=None):
    """
    SPIReS fSCA (0-1) as a (time, y, x) DataArray on the clipped SPIReS grid, with time as calendar dates.
    Full-tile files without a clipped version are clipped first.
    """
    raw_files = [f for f in glob(os.path.join(spires_dir, "SPIRES*.nc")) if not f.endswith("_clip.nc")]
    to_clip = [f for f in raw_files if not os.path.exists(clipped_path(f))]
    for f in tqdm(to_clip, desc="Clipping SPIReS files to the AOI"):
        clip_image_to_aoi(f, aoi)

    files = sorted(glob(os.path.join(spires_dir, "SPIRES*_clip.nc")))
    dates = pd.to_datetime([re.search(r"_(\d{8})_", os.path.basename(f)).group(1) for f in files], format="%Y%m%d")
    keep = np.ones(len(files), dtype=bool)
    if start_date:
        keep &= dates >= pd.to_datetime(start_date)
    if end_date:
        keep &= dates <= pd.to_datetime(end_date)
    files = [f for f, k in zip(files, keep) if k]
    if not files:
        raise FileNotFoundError(f"No SPIReS files found in {spires_dir} for the requested dates.")

    fsca_list = []
    for f in tqdm(files, desc="Loading SPIReS"):
        with xr.open_dataset(f, decode_coords="all", mask_and_scale=True) as ds:
            fsca_list.append((ds["snow_fraction"].squeeze("time", drop=True) / 100).load()
                             .expand_dims(time=[pd.to_datetime(ds.time.values[0]).normalize()]))
    fsca = xr.concat(fsca_list, dim="time").rename("fSCA")
    crs = xr.open_dataset(files[0], decode_coords="all").rio.crs
    return fsca.rio.write_crs(crs)


# ----- Models -----
def swe_to_fsca_on_grid(swe_m, thresholds, spires_grid, fine, default_crs):
    """
    Binary SCA (SWE > threshold) for each threshold, put on the SPIReS grid. Returns (threshold, y, x):
    fraction of snow-covered model cells in each SPIReS pixel for fine models, nearest cell otherwise.
    """
    if swe_m.rio.crs is None:
        swe_m = swe_m.rio.write_crs(default_crs)
    sca = xr.concat([xr.where(np.isnan(swe_m), np.nan, (swe_m > t).astype("float32")) for t in thresholds],
                    dim=pd.Index(thresholds, name="SWE_threshold_m"))
    sca = sca.rio.write_crs(swe_m.rio.crs).rio.write_nodata(np.nan)
    resampling = Resampling.average if fine else Resampling.nearest
    return sca.rio.reproject_match(spires_grid, resampling=resampling).clip(0, 1)


def build_model_fsca(model, model_swe, spires_fsca, aoi_mask, out_file, overwrite=False):
    """
    Modeled fSCA on the SPIReS grid and dates, (task, SWE_threshold_m, time, y, x), from the prepared
    (time, task, y, x) SWE [m]. Loaded from out_file if it already covers the SPIReS dates and thresholds
    (use overwrite=True after rebuilding the prepared model outputs).
    """
    if os.path.exists(out_file) and not overwrite:
        cached = xr.open_dataset(out_file)
        if (set(spires_fsca.time.values) <= set(cached.time.values)
                and list(cached.SWE_threshold_m.values) == SWE_THRESHOLDS):
            print(f"{model}: loading regridded fSCA from {out_file}")
            return cached["fSCA"].sel(time=spires_fsca.time).load()
        cached.close()

    spires_grid = spires_fsca.isel(time=0)
    dates = pd.DatetimeIndex(spires_fsca.time.values)
    by_task = []
    model_dates = [d for d in dates if d in model_swe.time.values]
    for task in TASKS:
        if task not in model_swe.task.values or not model_dates:
            print(f"{model} Task {task}: no model SWE for the SPIReS dates, skipping.")
            continue
        fsca = np.full((len(SWE_THRESHOLDS), len(dates)) + spires_grid.shape, np.nan, dtype="float32")
        for date in tqdm(model_dates, desc=f"{model} Task {task}"):
            swe = model_swe.sel(time=date, task=task).load()
            if swe.isnull().all():
                continue
            on_grid = swe_to_fsca_on_grid(
                swe, SWE_THRESHOLDS, spires_grid, MODELS[model]["fine"], default_crs=spires_fsca.rio.crs
            )
            fsca[:, dates.get_loc(date)] = np.where(aoi_mask, on_grid.values, np.nan)
        by_task.append(xr.DataArray(
            fsca[np.newaxis], dims=("task", "SWE_threshold_m", "time", "y", "x"),
            coords={"task": [task], "SWE_threshold_m": SWE_THRESHOLDS, "time": dates,
                    "y": spires_grid.y, "x": spires_grid.x}, name="fSCA"))
    if not by_task:
        return None

    model_fsca = xr.concat(by_task, dim="task").rio.write_crs(spires_fsca.rio.crs)
    model_fsca.attrs.update({"long_name": f"{model} fractional snow-covered area on the SPIReS grid", "units": "1"})
    model_fsca.to_dataset().to_netcdf(out_file, encoding={"fSCA": {"zlib": True, "complevel": 4}})
    print(f"{model}: regridded fSCA saved to {out_file}")
    return model_fsca


# ----- Metrics -----
def fsca_metrics(model_fsca, ref_fsca):
    """
    Basin-wide fSCA error metrics (0-100 scale) for 100 m models, as time series.
    """
    valid = model_fsca.notnull() & ref_fsca.notnull()
    model_fsca, ref_fsca = model_fsca.where(valid), ref_fsca.where(valid)
    error = (model_fsca - ref_fsca) * 100
    return xr.Dataset({
        "RMSE": (error ** 2).mean(dim=["x", "y"]) ** 0.5,
        "MAE": abs(error).mean(dim=["x", "y"]),
        "bias": error.mean(dim=["x", "y"]),
        "fSCA_average_error": (model_fsca.mean(dim=["x", "y"]) - ref_fsca.mean(dim=["x", "y"])) * 100,
        "correlation": xr.corr(model_fsca, ref_fsca, dim=["x", "y"]),
    })


def partial_credit_error(model_sca, ref_fsca):
    """
    Partial-credit error (0-100) for binary SCA from 2000 m models: 1 - fSCA where the model has snow,
    fSCA where it does not.
    """
    valid = model_sca.notnull() & ref_fsca.notnull()
    errors = xr.where(model_sca == 1, 1 - ref_fsca, ref_fsca).where(valid)
    return xr.Dataset({"partial_credit_error": errors.mean(dim=["x", "y"]) * 100})


def terrain_bin_index(dem, aspect, elev_bins):
    """
    Flat elevation/aspect bin index for each pixel (-1 outside the bins).
    """
    e = np.digitize(dem.values, elev_bins) - 1
    a = np.digitize(aspect.values, ASPECT_BINS) - 1
    n_e, n_a = len(elev_bins) - 1, len(ASPECT_BINS) - 1
    ok = np.isfinite(dem.values) & np.isfinite(aspect.values) & (e >= 0) & (e < n_e) & (a >= 0) & (a < n_a)
    return np.where(ok, e * n_a + a, -1), n_e, n_a


def binned_metrics(model, ref, bin_index, n_e, n_a, binary):
    """
    Metrics within each elevation/aspect bin.
    """
    lead = model.shape[:-2]
    model = model.reshape(lead + (-1,))
    ref = ref.reshape(lead + (-1,))
    flat = bin_index.ravel()
    names = ["partial_credit_error", "n_pixels"] if binary else ["RMSE", "MAE", "correlation", "n_pixels"]
    out = {k: np.full(lead + (n_e * n_a,), np.nan, dtype="float32") for k in names}
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in np.unique(flat[flat >= 0]):
            m, r = model[..., flat == b], ref[..., flat == b]
            valid = np.isfinite(m) & np.isfinite(r)
            m, r = np.where(valid, m, np.nan), np.where(valid, r, np.nan)
            n = valid.sum(axis=-1)
            out["n_pixels"][..., b] = np.where(n > 0, n, np.nan)
            if binary:
                out["partial_credit_error"][..., b] = np.nanmean(np.where(m == 1, 1 - r, r), axis=-1) * 100
            else:
                diff = m - r
                out["RMSE"][..., b] = np.sqrt(np.nanmean(diff ** 2, axis=-1)) * 100
                out["MAE"][..., b] = np.nanmean(np.abs(diff), axis=-1) * 100
                dm = m - np.nanmean(m, axis=-1, keepdims=True)
                dr = r - np.nanmean(r, axis=-1, keepdims=True)
                corr = np.nansum(dm * dr, axis=-1) / np.sqrt(np.nansum(dm ** 2, axis=-1) * np.nansum(dr ** 2, axis=-1))
                out["correlation"][..., b] = np.where(n > 1, corr, np.nan)
    return {k: v.reshape(lead + (n_e, n_a)) for k, v in out.items()}


def water_year(dates):
    dates = pd.DatetimeIndex(dates)
    return np.where(dates.month >= WATER_YEAR_START_MONTH, dates.year + 1, dates.year)


def melt_out_dowy(fsca_wy):
    """
    Per-pixel melt-out day of water year (1 = Oct 1): the last date with fSCA >= MELT_OUT_FSCA_THRESH.
    NaN where never snow covered, or still snow covered on the last date (melt-out not observed).
    """
    dates = pd.DatetimeIndex(fsca_wy.time.values)
    wy_start = pd.Timestamp(int(water_year(dates[:1])[0]) - 1, WATER_YEAR_START_MONTH, 1)
    dowy = xr.DataArray((dates - wy_start).days + 1, dims="time", coords={"time": fsca_wy.time})
    is_snow = fsca_wy >= MELT_OUT_FSCA_THRESH
    return dowy.where(is_snow).max(dim="time").where(~is_snow.isel(time=-1))


def melt_out_metrics(model_dowy, ref_dowy):
    err = (model_dowy - ref_dowy).values
    valid = np.isfinite(err)
    if valid.sum() < 2:
        return dict(melt_out_bias_days=np.nan, melt_out_MAE_days=np.nan, melt_out_RMSE_days=np.nan,
                    melt_out_correlation=np.nan, melt_out_n_pixels=int(valid.sum()))
    m, r = model_dowy.values[valid], ref_dowy.values[valid]
    corr = np.corrcoef(m, r)[0, 1] if np.std(m) > 0 and np.std(r) > 0 else np.nan
    return dict(melt_out_bias_days=err[valid].mean(), melt_out_MAE_days=np.abs(err[valid]).mean(),
                melt_out_RMSE_days=np.sqrt((err[valid] ** 2).mean()), melt_out_correlation=corr,
                melt_out_n_pixels=int(valid.sum()))


# ----- MAIN WORKFLOW -----
def compare_fsca(spires_dir, models_dir, aoi_file, out_dir, start_date=None, end_date=None,
                dem_file=None, overwrite=False):
    """
    Run the full SCA comparison and save all outputs to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    aoi = gpd.read_file(aoi_file)

    # Pre-process SPIReS
    spires_fsca = load_spires(spires_dir, aoi, start_date, end_date)
    crs = spires_fsca.rio.crs
    spires_grid = spires_fsca.isel(time=0)
    aoi_mask = ~geometry_mask(
        aoi.to_crs(crs).geometry, 
        out_shape=spires_grid.shape,
        transform=spires_grid.rio.transform()
        )
    spires_fsca = spires_fsca.where(aoi_mask)
    dates = pd.DatetimeIndex(spires_fsca.time.values)
    wys = water_year(dates)
    print(f"SPIReS: {len(dates)} dates from {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}, "
          f"{spires_grid.shape[0]} x {spires_grid.shape[1]} pixels ({abs(spires_grid.rio.resolution()[0]):.0f} m)")
    (spires_fsca * 100).rename("snow_fraction").assign_attrs(units="%").rio.write_crs(crs).to_dataset().to_netcdf(
        os.path.join(out_dir, "SPIReS_fSCA.nc"))

    # Terrain, if a DEM is given
    terrain = None
    if dem_file:
        dem = rxr.open_rasterio(dem_file, masked=True).squeeze("band", drop=True)
        dem = dem.rio.reproject_match(spires_grid, resampling=Resampling.bilinear).where(aoi_mask)
        aspect = xrspatial.aspect(dem)
        elev_bins = np.arange(ELEV_BIN_WIDTH * (np.nanmin(dem) // ELEV_BIN_WIDTH),
                              ELEV_BIN_WIDTH * (np.nanmax(dem) // ELEV_BIN_WIDTH + 2), ELEV_BIN_WIDTH)
        terrain = (*terrain_bin_index(dem, aspect, elev_bins), elev_bins)
    else:
        print("No DEM given: skipping elevation/aspect-binned metrics.")

    # SPIReS melt-out dates, and the pixels SPIReS saw as snow covered, for each water year
    spires_dowy = {wy: melt_out_dowy(spires_fsca.sel(time=wys == wy)) for wy in np.unique(wys)}

    models = [m for m in MODELS]
    model_outputs = open_model_outputs(models_dir, "SWE")
    coords = {"model": models, "task": TASKS, "SWE_threshold_m": SWE_THRESHOLDS}
    series_names = ["RMSE", "MAE", "correlation", "bias", "fSCA_average_error", "partial_credit_error"]
    series = {k: np.full((len(models), len(TASKS), len(SWE_THRESHOLDS), len(dates)), np.nan) for k in series_names}
    binned, melt, melt_binned = {}, {}, {}
    melt_names = ["melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days", "melt_out_correlation", "melt_out_n_pixels"]
    melt_grids = np.full((len(models), len(TASKS), len(SWE_THRESHOLDS), len(np.unique(wys))) + spires_grid.shape, np.nan)

    # 2-4. Each model
    for mi, model in enumerate(models):
        if model not in model_outputs:
            print(f"{model}: no prepared SWE in {models_dir}, skipping.")
            continue
        model_fsca = build_model_fsca(
            model, model_outputs[model]["SWE"], spires_fsca, aoi_mask,
            os.path.join(out_dir, f"fSCA_regridded_{model}.nc"), overwrite
        )
        if model_fsca is None:
            continue
        fine = MODELS[model]["fine"]
        for task in model_fsca.task.values:
            ti = TASKS.index(int(task))
            for si, thresh in enumerate(SWE_THRESHOLDS):
                mod = model_fsca.sel(task=task, SWE_threshold_m=thresh)
                scores = fsca_metrics(mod, spires_fsca) if fine else partial_credit_error(mod, spires_fsca)
                for name in scores.data_vars:
                    series[name][mi, ti, si] = scores[name].values
                if terrain:
                    bin_index, n_e, n_a, _ = terrain
                    binned[(mi, ti, si)] = binned_metrics(mod.values, spires_fsca.values, bin_index, n_e, n_a, binary=not fine)

                for wi, wy in enumerate(np.unique(wys)):
                    ref = spires_dowy[wy]
                    model_dowy = melt_out_dowy(mod.sel(time=wys == wy).where(ref.notnull())).where(ref.notnull())
                    melt[(mi, ti, si, wi)] = melt_out_metrics(model_dowy, ref)
                    melt_grids[mi, ti, si, wi] = model_dowy.values
                    if terrain:
                        melt_binned[(mi, ti, si, wi)] = binned_errors((model_dowy - ref).values, bin_index, n_e, n_a)

    # Save outputs
    water_years = np.unique(wys)
    metrics_ds = xr.Dataset(
        {k: (list(coords) + ["time"], v) for k, v in series.items() if np.isfinite(v).any()},
        coords={**coords, "time": dates}
        )
    metrics_ds.attrs.update({
        "description": "Performance metrics for all models vs. SPIReS fSCA.",
        "note_fine": "100 m models: RMSE, MAE, bias, fSCA_average_error, and correlation. Errors are on a 0-100 scale.",
        "note_coarse": "2000 m models: partial_credit_error (0-100).",
        "swe_units": "SWE_threshold_m is in meters of SWE for all models.",
    })
    metrics_ds.to_netcdf(os.path.join(out_dir, "compiled_performance_metrics.nc"))

    melt_ds = xr.Dataset(
        {k: (list(coords) + ["water_year"], _stack(melt, k, (len(models), len(TASKS), len(SWE_THRESHOLDS), len(water_years)))) 
         for k in melt_names}, 
         coords={**coords, "water_year": water_years}
         )
    melt_ds.attrs.update({
        "description": (
            f"Melt-out date comparison metrics (model - SPIReS, days), at pixels where SPIReS observed snow cover (fSCA >= {MELT_OUT_FSCA_THRESH}) during that water year."),
        "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH, 
        "water_year_start_month": WATER_YEAR_START_MONTH
        })
    melt_ds.to_netcdf(os.path.join(out_dir, "compiled_melt_out_metrics.nc"))

    grids_ds = xr.Dataset(
        {
            "model_melt_out_dowy": (list(coords) + ["water_year", "y", "x"], melt_grids),
            "SPIReS_melt_out_dowy": (["water_year", "y", "x"], np.stack([spires_dowy[wy].values for wy in water_years]))
            },
            coords={**coords, "water_year": water_years, "y": spires_grid.y, "x": spires_grid.x}
            )
    grids_ds.attrs.update({
        "description": "Per-pixel melt-out day of water year (1 = Oct 1), masked to pixels SPIReS observed as snow covered that water year.",
        "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH
        })
    grids_ds.rio.write_crs(crs).to_netcdf(os.path.join(out_dir, "melt_out_dates_gridded.nc"))

    if terrain:
        _, n_e, n_a, elev_bins = terrain
        bin_coords = {"elev_bin": np.arange(n_e), "aspect_bin": np.arange(n_a)}
        attrs = {"elev_bins": elev_bins.tolist(), "aspect_bins": ASPECT_BINS.tolist()}
        names = sorted({k for d in binned.values() for k in d})
        
        binned_ds = xr.Dataset(
            {k: (list(coords) + ["time", "elev_bin", "aspect_bin"],
                 _stack(binned, k, (len(models), len(TASKS), len(SWE_THRESHOLDS), len(dates), n_e, n_a))) for k in names
                 },
            coords={**coords, "time": dates, **bin_coords}, 
            attrs={"description": "Performance metrics binned by elevation and aspect.", **attrs}
            )
        binned_ds.to_netcdf(os.path.join(out_dir, "compiled_metrics_binned_elev_aspect.nc"))
        
        melt_binned_ds = xr.Dataset(
            {k: (list(coords) + ["water_year", "elev_bin", "aspect_bin"],
                 _stack(melt_binned, k, (len(models), len(TASKS), len(SWE_THRESHOLDS), len(water_years), n_e, n_a)))
                 for k in ["melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days", "melt_out_n_pixels"]},
            coords={**coords, "water_year": water_years, **bin_coords},
            attrs={
                "description": "Melt-out date errors (model - SPIReS, days) binned by elevation and aspect.",
                "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH, **attrs
                }
                )
        melt_binned_ds.to_netcdf(os.path.join(out_dir, "compiled_melt_out_binned_elev_aspect.nc"))

    print(f"SCA comparison outputs saved to {out_dir}")


def binned_errors(err, bin_index, n_e, n_a):
    """
    Bias, MAE, RMSE [days], and pixel count of a (y, x) melt-out error grid within each terrain bin.
    """
    out = {k: np.full(n_e * n_a, np.nan) for k in
           ["melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days", "melt_out_n_pixels"]}
    flat, e = bin_index.ravel(), err.ravel()
    for b in np.unique(flat[flat >= 0]):
        vals = e[(flat == b) & np.isfinite(e)]
        if vals.size:
            out["melt_out_bias_days"][b] = vals.mean()
            out["melt_out_MAE_days"][b] = np.abs(vals).mean()
            out["melt_out_RMSE_days"][b] = np.sqrt((vals ** 2).mean())
            out["melt_out_n_pixels"][b] = vals.size
    return {k: v.reshape(n_e, n_a) for k, v in out.items()}


def _stack(results, name, shape):
    """
    Fill an array of the given shape from {(index tuple): {name: value or array}}. Entries without the
    metric (e.g., partial_credit_error for 100 m models) are left as NaN.
    """
    data = np.full(shape, np.nan)
    for idx, values in results.items():
        if name in values:
            data[idx] = values[name]
    return data


def main():
    args = get_parser()
    compare_fsca(
        args.spires_dir, args.models_dir, args.aoi_file, args.out_dir, start_date=args.start_date,
        end_date=args.end_date, dem_file=args.dem_file, overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
