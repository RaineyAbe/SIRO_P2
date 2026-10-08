#!/usr/bin/env python

"""
SIRO model intercomparison pipeline.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Runs the selected steps (--steps) for an area of interest and time span:
    0. Compile raw model outputs into one netCDF per model (needs --model_dir)
    1. Compare modeled and LiDAR snow depth
    2. Download SPIReS fSCA
    3. Compare modeled and SPIReS SCA
    4. Download SNOTEL snow depth and SWE
    5. Compare modeled and SNOTEL snow depth and SWE

Steps 1, 3, and 5 read the step 0 netCDFs from OUT_DIR/model_outputs/ (or --models_dir).

Outputs are organized as:
  OUT_DIR/
  ├── pipeline_settings.json    settings used for the most recent run
  ├── lidar/                    LiDAR comparison metrics and rasters (see compare_lidar.py)
  ├── model_outputs/            Prepped model outputs (see prepare_models.py)
  ├── SCA/                      fSCA comparison metrics (see compare_fSCA.py)
  │   └── SPIReS/               downloaded SPIReS fSCA files
  └── SNOTEL/                   downloaded SNOTEL station data and comparison metrics (see download_SNOTEL.py and compare_SNOTEL.py)

Usage:
    python SIRO_P2_pipeline.py \\
    --aoi_file /path/to/MCS_outline.gpkg \\
    --start_date "2022-10-01" \\
    --end_date "2025-06-30" \\
    --steps 2 4 \\
    --out_dir /path/to/output

Step 1 also needs --lidar_aoi_file and the prepare_lidar.py netCDFs (lidar_snow_depth_<res>m.nc), which
are read from OUT_DIR/lidar/ unless --lidar_dir is given.
"""

import argparse
import json
import os
import pandas as pd
from SIRO_P2.prepare_models import prepare_model_outputs
from SIRO_P2.compare_lidar import compare_lidar
from SIRO_P2.download_fSCA import download_spires
from SIRO_P2.compare_fSCA import compare_fsca
from SIRO_P2.download_SNOTEL import download_snotel
from SIRO_P2.compare_SNOTEL import compare_snotel

STEPS = {
    0: "prepare_models",
    1: "compare_lidar",
    2: "download_fsca",
    3: "compare_fsca",
    4: "download_snotel",
    5: "compare_snotel",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Run the SIRO model intercomparison pipeline.")
    parser.add_argument("--aoi_file", required=True, type=str, help="Vector file (e.g., .gpkg, .shp) of the area of interest.")
    parser.add_argument("--lidar_aoi_file", default=None, type=str, help="Vector file of the LiDAR domain. Required for compare_lidar.")
    parser.add_argument("--lidar_dir", default=None, type=str, help="Directory of the prepare_lidar.py netCDFs (default: OUT_DIR/lidar).")
    parser.add_argument("--max_lidar_depth", default=5.0, type=float, help="Exclude LiDAR depths >= this value [m] from the LiDAR metrics (default: 5). Use a negative value to disable.")
    parser.add_argument("--start_date", required=True, type=str, help="First date to include (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, type=str, help="Last date to include, inclusive (YYYY-MM-DD).")
    parser.add_argument("--months", default=[10, 11, 12, 1, 2, 3, 4, 5, 6], type=int, nargs="+", help="Only include these months.")
    parser.add_argument("--out_dir", required=True, type=str, help="Path where all pipeline outputs will be saved.")
    parser.add_argument("--model_dir", default=None, type=str, help="Path to the raw model outputs. Required for prepare_models.")
    parser.add_argument("--models_dir", default=None, type=str, help="Directory of prepared model netCDFs for the comparison steps (default: OUT_DIR/model_outputs).")
    parser.add_argument("--dem_file", default=None, type=str, help="DEM GeoTIFF for elevation/aspect-binned SCA metrics (skipped if not given).")
    parser.add_argument("--snotel_sites", default=None, type=str, nargs="+", help="SNOTEL station IDs (e.g., 637:ID:SNTL). Default: all stations within --snotel_buffer_km of the AOI.")
    parser.add_argument("--snotel_buffer_km", default=1.0, type=float, help="Include SNOTEL stations within this distance of the AOI [km] (default: 1).")
    parser.add_argument("--steps", default=list(STEPS), type=int, nargs="+", choices=list(STEPS), help="Pipeline steps to run: " + ", ".join(f"{k} = {v}" for k, v in STEPS.items()) + " (default: all).")
    parser.add_argument("--dry_run", default=False, type=bool, help="For download steps, list what would be downloaded without downloading.")
    parser.add_argument("--clip_fsca_to_aoi", action="store_true", help="Clip fSCA files to the AOI after downloading.")
    parser.add_argument("--remove_unclipped_fsca", action="store_true", help="Remove unclipped fSCA files after clipping.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing files.")
    args = parser.parse_args()

    # Check inputs before running anything
    # AOI
    if not os.path.exists(args.aoi_file):
        parser.error(f"AOI file not found: {args.aoi_file}")
    # Date range
    try:
        start, end = pd.to_datetime(args.start_date), pd.to_datetime(args.end_date)
    except ValueError as e:
        parser.error(f"Could not parse dates: {e}")
    if end < start:
        parser.error("--end_date must be on or after --start_date.")
    if any(m < 1 or m > 12 for m in args.months):
        parser.error("--months must be between 1 and 12.")
    # Raw model outputs needed to prepare the model netCDFs
    if 0 in args.steps and args.model_dir is None:
        parser.error("--model_dir is required for prepare_models.")
    # Lidar data needed for lidar comparison
    if 1 in args.steps:
        if args.lidar_aoi_file is None:
            parser.error("--lidar_aoi_file is required for compare_lidar.")
        if not os.path.exists(args.lidar_aoi_file):
            parser.error(f"LiDAR AOI file not found: {args.lidar_aoi_file}")
    # DEM needed for fSCA comparison
    if 3 in args.steps:
        if not args.dem_file:
            parser.error("--dem_file is required for compare_fSCA.")
        elif not os.path.exists(args.dem_file):
            parser.error(f"DEM file could not be found and is required for compare_fSCA. Check input --dem_file: {args.dem_file}")

    return args


def main():
    args = get_parser()

    # --- I/O ---
    dirs = {
        "models": args.models_dir or os.path.join(args.out_dir, "model_outputs"),
        "lidar": os.path.join(args.out_dir, "lidar"),
        "SCA": os.path.join(args.out_dir, "SCA"),
        "SPIReS": os.path.join(args.out_dir, "SCA", "SPIReS"),
        "SNOTEL": os.path.join(args.out_dir, "SNOTEL"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Save the settings for this run
    with open(os.path.join(args.out_dir, "pipeline_settings.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Running steps: {', '.join(f'{k} ({v})' for k, v in STEPS.items() if k in args.steps)}")

    # --- Prep model outputs ---
    if 0 in args.steps:
        print("\n--- Prepping model outputs ---")
        prepare_model_outputs(
            args.model_dir, dirs["models"], start_date=args.start_date, end_date=args.end_date,
            overwrite=args.overwrite
        )

    # --- LiDAR ---
    if 1 in args.steps:
        print("\n--- Comparing modeled and LiDAR snow depth ---")
        compare_lidar(
            args.lidar_dir or dirs["lidar"], dirs["models"], args.lidar_aoi_file, args.aoi_file, dirs["lidar"],
            start_date=args.start_date, end_date=args.end_date,
            max_lidar_depth=args.max_lidar_depth if args.max_lidar_depth >= 0 else None
        )

    # --- SCA ---
    if 2 in args.steps:
        print("\n--- Downloading SPIReS fSCA ---")
        download_spires(
            dirs["SPIReS"], args.start_date, args.end_date, aoi_file=args.aoi_file,
            months=args.months, dry_run=args.dry_run, overwrite=args.overwrite, 
            clip_to_aoi=args.clip_fsca_to_aoi, remove_unclipped=args.remove_unclipped_fsca
            )
        
    if 3 in args.steps:
        print("\n--- Comparing modeled and SPIReS fSCA ---")
        compare_fsca(
            dirs["SPIReS"], dirs["models"], args.aoi_file, dirs["SCA"], start_date=args.start_date,
            end_date=args.end_date, dem_file=args.dem_file, overwrite=args.overwrite
            )

    # --- SNOTEL ---
    if 4 in args.steps:
        print("\n--- Downloading SNOTEL snow depth ---")
        download_snotel(
            dirs["SNOTEL"], args.start_date, args.end_date, aoi_file=args.aoi_file,
            sites=args.snotel_sites, buffer_km=args.snotel_buffer_km
            )

    if 5 in args.steps:
        print("\n--- Comparing modeled and SNOTEL snow depth ---")
        compare_snotel(
            dirs["SNOTEL"], dirs["models"], dirs["SNOTEL"], start_date=args.start_date,
            end_date=args.end_date, months=args.months
        )


if __name__ == "__main__":
    main()
