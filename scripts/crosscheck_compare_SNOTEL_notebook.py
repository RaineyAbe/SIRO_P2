#!/usr/bin/env python

"""
Cross-check SIRO_P2/compare_SNOTEL.py against the KGE/NSE tables in analysis_notebooks/KGE_netCDF.ipynb.

Recomputes snow depth KGE, NSE, and beta with compare_SNOTEL.py's functions, but on the notebook's date
windows (Oct 2 - Jun 30 for WY23 and WY24, Oct 2 - May 30 for WY25), and prints them next to the
notebook's saved results.

Usage:
    python crosscheck_compare_SNOTEL_notebook.py \\
    --snotel_dir /Users/rdcrlrka/Research/SIRO/SIRO_P2/crosscheck_MCS/SNOTEL \\
    --models_dir /Users/rdcrlrka/Research/SIRO/SIRO_P2/crosscheck_MCS/model_outputs
"""

import argparse

import numpy as np
import pandas as pd

from SIRO_P2.compare_SNOTEL import calculate_metrics, load_station_data, load_stations, sample_at_station
from SIRO_P2.prepare_models import open_model_outputs

WINDOWS = [("2022-10-02", "2023-06-30"), ("2023-10-02", "2024-06-30"), ("2024-10-02", "2025-05-30")]
# Saved notebook outputs (cells 12 and 15): (KGE, NSE, beta) for Task 1 and Task 2
NOTEBOOK = {
    "637:ID:SNTL": {
        "HMS-TI": ((0.763, 0.916, 0.835), (0.517, 0.675, 0.628)),
        "HMS-EB": ((0.439, 0.586, 0.595), (0.518, 0.676, 0.680)),
        "SnowModel": ((0.429, 0.586, 0.612), (0.521, 0.681, 0.670)),
        "iSnobal": ((0.726, 0.861, 0.803), (0.813, 0.917, 0.847)),
    },
    "978:ID:SNTL": {
        "HMS-TI": ((0.264, 0.138, 0.430), (0.125, -0.175, 0.330)),
        "HMS-EB": ((0.224, 0.184, 0.419), (0.286, 0.317, 0.469)),
        "SnowModel": ((0.663, 0.831, 0.790), (0.804, 0.904, 0.893)),
        "iSnobal": ((0.738, 0.877, 0.824), (0.833, 0.930, 0.875)),
    },
}


def get_parser():
    parser = argparse.ArgumentParser(description="Cross-check compare_SNOTEL.py against KGE_netCDF.ipynb.")
    parser.add_argument("--snotel_dir", required=True, type=str, help="Directory written by download_SNOTEL.py.")
    parser.add_argument("--models_dir", required=True, type=str, help="Directory of prepare_models.py netCDFs.")
    return parser.parse_args()


def main():
    args = get_parser()
    stations = load_stations(args.snotel_dir)
    models = open_model_outputs(args.models_dir, "snow_depth")
    rows = []
    for station in stations.itertuples():
        if station.id not in NOTEBOOK:
            continue
        obs_data = load_station_data(args.snotel_dir, station.id)
        for model, ds in models.items():
            pixel = sample_at_station(ds, station.lon, station.lat)[0]["snow_depth"]
            dates = pd.DatetimeIndex(pixel.time.values)
            in_windows = np.zeros(len(dates), dtype=bool)
            for start, end in WINDOWS:
                in_windows |= (dates >= start) & (dates <= end)
            dates = dates[in_windows]
            obs = obs_data["snow_depth_m"].reindex(dates).to_numpy(dtype="float64")
            for task in (1, 2):
                sim = pixel.sel(task=task, time=dates).to_numpy().astype("float64")
                m = calculate_metrics(sim, obs)
                kge, nse, beta = NOTEBOOK[station.id][model][task - 1]
                rows.append({
                    "station": station.id, "model": model, "task": task, "n_days": m["n_days"],
                    "KGE": m.get("KGE"), "KGE_nb": kge, "NSE": m.get("NSE"), "NSE_nb": nse,
                    "beta": m.get("KGE_beta"), "beta_nb": beta,
                })
    df = pd.DataFrame(rows)
    for k in ["KGE", "NSE", "beta"]:
        df[f"{k}_diff"] = df[k] - df[f"{k}_nb"]
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    worst = df[[c for c in df if c.endswith("_diff")]].abs().max()
    print(f"\nLargest absolute differences from the notebook (it reports 3 decimals):\n{worst.round(4).to_string()}")


if __name__ == "__main__":
    main()
