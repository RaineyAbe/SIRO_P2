#!/usr/bin/env python

"""
Compare modeled snow depth and SWE with SNOTEL and CDEC station observations.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Steps:
    1. Load the station list and data written by download_SNOTEL.py: daily data from SNOTEL and CDEC
       sensor stations, and measurement-date data from CDEC snow courses.
    2. Sample each prepared model (prepare_model_outputs.py) at the pixel containing each station
       (nearest pixel centre; stations outside a model's grid are skipped).
    3. For each station, model, task, and variable, compute NSE, KGE (with its r, alpha, and beta
       components), bias, MAE, and RMSE over the days with both observed and modeled values, for each
       water year and for all water years together. Days can be limited to --months. Metrics need at
       least MIN_DAYS paired days, so snow courses (a few measurements per year) usually only get
       metrics for the "all" period.

Outputs, in --out_dir:
    SNOTEL_comparison_metrics.csv   one row per station, model, task, variable, and period

Usage:
    python -m SIRO_P2.compare_SNOTEL \\
    --snotel_dir /path/to/SNOTEL \\
    --models_dir /path/to/model_outputs \\
    --out_dir /path/to/SNOTEL
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import rioxarray 
from pyproj import Transformer

from .prepare_models import TASKS, open_model_outputs

# ----- SETTINGS -----
# Model variable -> SNOTEL column (both in meters)
VARIABLES = {
    "snow_depth": "snow_depth_m",
    "SWE": "SWE_m"
    }
WATER_YEAR_START_MONTH = 10
MIN_DAYS = 10  # minimum number of paired days to compute metrics


def get_parser():
    parser = argparse.ArgumentParser(description="Compare modeled snow depth and SWE with SNOTEL observations.")
    parser.add_argument("--snotel_dir", required=True, type=str, help="Directory written by download_SNOTEL.py.")
    parser.add_argument("--models_dir", required=True, type=str, help="Directory of prepare_model_outputs.py netCDFs.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory where the metrics CSV will be saved.")
    parser.add_argument("--start_date", default=None, type=str, help="First date to compare (default: all).")
    parser.add_argument("--end_date", default=None, type=str, help="Last date to compare (default: all).")
    parser.add_argument("--months", default=None, type=int, nargs="+", help="Only compare these months (default: all).")
    return parser.parse_args()


# ----- SNOTEL -----
def load_stations(snotel_dir):
    """
    Station table (id, name, lon, lat, ...) from SNOTEL_stations.csv.
    """
    path = os.path.join(snotel_dir, "SNOTEL_stations.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run download_SNOTEL.py (pipeline step 4) first.")
    stations = pd.read_csv(path)
    if not {"lon", "lat"} <= set(stations.columns):
        raise ValueError(f"{path} has no lon/lat columns. Rerun download_SNOTEL.py to add station locations.")
    return stations


def load_station_data(snotel_dir, station_id, data_file=None):
    """
    Station data [m] indexed by calendar date, or None if the file is missing. data_file is the file name
    from the station table (default: the SNOTEL daily file name).
    """
    if not isinstance(data_file, str):
        data_file = f"SNOTEL_{station_id.replace(':', '_')}_daily.csv"
    path = os.path.join(snotel_dir, data_file)
    if not os.path.exists(path):
        warnings.warn(f"No data file for station {station_id}: {path}")
        return None
    data = pd.read_csv(path, index_col="date")
    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None).normalize()
    return data[~data.index.duplicated()]


# ----- MODELS -----
def sample_at_station(ds, lon, lat):
    """
    The model pixel containing a station: (time, task) Dataset of all variables, plus the pixel centre
    and station coordinates in the model CRS. Returns None if the station is outside the model grid.
    """
    x, y = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True).transform(lon, lat)
    res_x, res_y = (abs(r) for r in ds.rio.resolution())
    if not (ds.x.min() - res_x / 2 <= x <= ds.x.max() + res_x / 2 and ds.y.min() - res_y / 2 <= y <= ds.y.max() + res_y / 2):
        return None
    pixel = ds[[v for v in VARIABLES if v in ds]].sel(x=x, y=y, method="nearest").load()
    return pixel, (float(pixel.x), float(pixel.y)), (x, y)


# ----- METRICS -----
def calculate_metrics(sim, obs):
    """
    NSE, KGE and its components (r, alpha = sim/obs standard deviation ratio, beta = sim/obs mean ratio),
    bias, MAE, and RMSE over days with both values.
    """
    valid = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[valid], obs[valid]
    metrics = {"n_days": int(valid.sum())}
    if metrics["n_days"] < MIN_DAYS:
        return metrics

    err = sim - obs
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(sim, obs)[0, 1] if sim.std() > 0 and obs.std() > 0 else np.nan
        alpha = sim.std() / obs.std() if obs.std() > 0 else np.nan
        beta = sim.mean() / obs.mean() if obs.mean() > 0 else np.nan
        nse = 1 - (err ** 2).sum() / ((obs - obs.mean()) ** 2).sum() if obs.std() > 0 else np.nan
    metrics.update({
        "obs_mean_m": obs.mean(),
        "model_mean_m": sim.mean(),
        "bias_m": err.mean(),
        "MAE_m": np.abs(err).mean(),
        "RMSE_m": np.sqrt((err ** 2).mean()),
        "NSE": nse,
        "KGE": 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
        "KGE_r": r,
        "KGE_alpha": alpha,
        "KGE_beta": beta,
    })
    return metrics


def water_year(dates):
    dates = pd.DatetimeIndex(dates)
    return np.where(dates.month >= WATER_YEAR_START_MONTH, dates.year + 1, dates.year)


# ----- MAIN WORKFLOW -----
def compare_snotel(snotel_dir, models_dir, out_dir, start_date=None, end_date=None, months=None):
    """
    Compare every prepared model with every SNOTEL station and save SNOTEL_comparison_metrics.csv.
    Returns the metrics DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    stations = load_stations(snotel_dir)
    if stations.empty:
        print("No stations in SNOTEL_stations.csv; skipping the station comparison.")
        return pd.DataFrame()
    models = open_model_outputs(models_dir)
    print(f"Stations: {', '.join(f'{s.name} ({s.id})' for s in stations.itertuples())}")
    print(f"Models: {', '.join(models)}")

    rows = []
    for station in stations.itertuples():
        obs_data = load_station_data(snotel_dir, station.id, getattr(station, "data_file", None))
        if obs_data is None:
            continue

        for model, ds in models.items():
            sampled = sample_at_station(ds, station.lon, station.lat)
            if sampled is None:
                warnings.warn(f"{station.name} ({station.id}) is outside the {model} grid, skipping.")
                continue
            pixel, (pixel_x, pixel_y), (station_x, station_y) = sampled

            # Days to compare
            dates = pd.DatetimeIndex(pixel.time.values)
            keep = np.ones(len(dates), dtype=bool)
            if start_date:
                keep &= dates >= pd.to_datetime(start_date)
            if end_date:
                keep &= dates <= pd.to_datetime(end_date)
            if months:
                keep &= dates.month.isin(months)
            dates = dates[keep]
            wys = water_year(dates)
            periods = [(str(wy), wys == wy) for wy in np.unique(wys)] + [("all", np.ones(len(dates), dtype=bool))]

            for var, column in VARIABLES.items():
                if var not in pixel or column not in obs_data:
                    continue
                obs = obs_data[column].reindex(dates).to_numpy(dtype="float64")
                for task in pixel.task.values:
                    sim = pixel[var].sel(task=task, time=dates).to_numpy().astype("float64")
                    for period, in_period in periods:
                        rows.append({
                            "network": getattr(station, "network", "SNOTEL"),
                            "station_type": getattr(station, "station_type", "sensor"),
                            "station_id": station.id,
                            "station_name": station.name,
                            "model": model,
                            "task": int(task),
                            "condition": TASKS.get(int(task), ""),
                            "variable": var,
                            "period": period,
                            **calculate_metrics(sim[in_period], obs[in_period]),
                            "start_date": f"{dates[in_period].min():%Y-%m-%d}" if in_period.any() else None,
                            "end_date": f"{dates[in_period].max():%Y-%m-%d}" if in_period.any() else None,
                            "station_x": station_x,
                            "station_y": station_y,
                            "pixel_x": pixel_x,
                            "pixel_y": pixel_y,
                            "station_to_pixel_center_m": np.hypot(station_x - pixel_x, station_y - pixel_y),
                        })

    metrics = pd.DataFrame(rows)
    out_file = os.path.join(out_dir, "SNOTEL_comparison_metrics.csv")
    metrics.to_csv(out_file, index=False)
    print(f"\nSNOTEL comparison metrics for {len(metrics)} station/model/task/variable/period combinations saved to: {out_file}")
    if not metrics.empty:
        summary = metrics[(metrics.period == "all") & (metrics.variable == "snow_depth")]
        print(summary[["network", "station_name", "model", "condition", "n_days", "NSE", "KGE", "bias_m"]].round(3).to_string(index=False))
    return metrics


def main():
    args = get_parser()
    compare_snotel(
        args.snotel_dir,
        args.models_dir,
        args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        months=args.months
    )


if __name__ == "__main__":
    main()
