#!/usr/bin/env python

"""
Download daily SNOTEL snow depth and SWE for stations in and near an area of interest.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Stations are found from the AOI file, where NRCS SNOTEL stations within --buffer_km of the AOI polygon are included. 

Outputs, in --out_dir:
    SNOTEL_stations.csv         station ID, name, location, elevation, and distance from the AOI
    SNOTEL_<ID>_daily.csv       daily snow depth and SWE [m] for each station (e.g., SNOTEL_637_ID_SNTL_daily.csv)

Usage:
    # With area of interest vector file:
    python -m SIRO_P2.download_SNOTEL \
        --aoi_file /path/to/MCS_outline.gpkg \
        --start_date 2022-10-01 \
        --end_date 2025-06-30 \
        --out_dir /path/to/SNOTEL
  
    # With station IDs:
    python -m SIRO_P2.download_SNOTEL \
        --sites 637:ID:SNTL 978:ID:SNTL \
        --start_date 2022-10-01 \
        --end_date 2025-06-30 \
        --out_dir /path/to/SNOTEL
"""

import argparse
import os
import time
import geopandas as gpd
import pandas as pd
from metloom.pointdata import SnotelPointData
from shapely.geometry import Point

INCHES_TO_M = 0.0254
FEET_TO_M = 0.3048
N_RETRIES = 3           # attempts per station if the NRCS server returns an error
RETRY_WAIT_S = 30       # seconds to wait between attempts
VARIABLES = {           # metloom variable -> output column [m]
    "SNOWDEPTH": "snow_depth_m",
    "SWE": "SWE_m",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Download daily SNOTEL snow depth and SWE for an area of interest.")
    parser.add_argument("--out_dir", required=True, type=str, help="Path where downloaded files will be saved.")
    parser.add_argument("--start_date", required=True, type=str, help="First date to download (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, type=str, help="Last date to download, inclusive (YYYY-MM-DD).")
    parser.add_argument("--aoi_file", default=None, type=str, help="Vector file (e.g., .gpkg, .shp) of the area of interest, used to find stations.")
    parser.add_argument("--sites", default=None, type=str, nargs="+", help="SNOTEL station IDs (e.g., 637:ID:SNTL), instead of finding them from --aoi_file.")
    parser.add_argument("--buffer_km", default=1.0, type=float, help="Include stations within this distance of the AOI [km] (default: 1).")
    args = parser.parse_args()
    if (args.aoi_file is None) == (args.sites is None):
        parser.error("Specify exactly one of --aoi_file or --sites.")
    return args


def find_stations(aoi_file, buffer_km=1.0):
    """
    SNOTEL stations within buffer_km of the AOI, as a GeoDataFrame (EPSG:4326) with the
    station ID, name, elevation [m], and distance from the AOI [km] (0 = inside).
    """
    aoi = gpd.read_file(aoi_file)
    aoi_projected = aoi.to_crs(aoi.estimate_utm_crs())

    # Candidate stations in the AOI's bounding box, expanded by the buffer
    search_area = gpd.GeoDataFrame(geometry=aoi_projected.buffer(buffer_km * 1000), crs=aoi_projected.crs).to_crs(4326)
    candidates = SnotelPointData.points_from_geometry(
        search_area, [SnotelPointData.ALLOWED_VARIABLES.SNOWDEPTH], within_geometry=False
        )

    # Keep stations within buffer_km of the AOI polygon itself
    aoi_polygon = aoi_projected.union_all()
    rows = []
    for point in candidates:
        lon, lat, elev_ft = point.metadata.x, point.metadata.y, point.metadata.z
        location = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(aoi_projected.crs).iloc[0]
        distance_km = location.distance(aoi_polygon) / 1000
        if distance_km <= buffer_km:
            rows.append({
                "id": point.id, 
                "name": point.name, 
                "elevation_m": round(elev_ft * FEET_TO_M, 1),
                "distance_from_aoi_km": round(distance_km, 2), 
                "geometry": Point(lon, lat)
                })
    if not rows:  # no stations: an empty table with the same columns (GeoDataFrame needs a geometry column)
        return gpd.GeoDataFrame(
            columns=["id", "name", "elevation_m", "distance_from_aoi_km", "geometry"], 
            geometry="geometry", 
            crs="EPSG:4326"
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def station_locations(sites):
    """
    GeoDataFrame (EPSG:4326) of the given station IDs with their elevation [m], so that stations given by
    ID are located the same way as stations found from the AOI.
    """
    rows = []
    for site in sites:
        point = SnotelPointData(site, site)
        lon, lat, elev_ft = point.metadata.x, point.metadata.y, point.metadata.z
        rows.append({
            "id": site,
            "name": site,
            "elevation_m": round(elev_ft * FEET_TO_M, 1),
            "geometry": Point(lon, lat),
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def get_station_data(site_id, start_date, end_date, name=None):
    """
    Daily snow depth and SWE [m] for one station, indexed by local date.
    """
    point = SnotelPointData(site_id, name or site_id)
    variables = [getattr(point.ALLOWED_VARIABLES, v) for v in VARIABLES]
    df = point.get_daily_data(pd.to_datetime(start_date), pd.to_datetime(end_date), variables)
    if df is None or df.empty:
        return None

    df = df.reset_index()
    out = pd.DataFrame({"date": df["datetime"]})
    for variable, column in VARIABLES.items():
        if variable not in df:
            out[column] = float("nan")
            continue
        units = df[f"{variable}_units"].dropna().unique()
        if len(units) and set(units) != {"in"}:
            raise ValueError(f"Unexpected {variable} units for {site_id}: {units}")
        out[column] = (df[variable] * INCHES_TO_M).round(4)
    return out.set_index("date")


def download_snotel(out_dir, start_date, end_date, aoi_file=None, sites=None, buffer_km=1.0):
    """
    Download daily SNOTEL snow depth and SWE between start_date and end_date (inclusive) for
    stations within buffer_km of aoi_file (or the given station IDs). Returns the stations table.
    """
    os.makedirs(out_dir, exist_ok=True)

    if sites:
        stations = station_locations(sites)
    else:
        stations = find_stations(aoi_file, buffer_km)
        if stations.empty:
            print(f"No SNOTEL stations found within {buffer_km} km of the AOI. Try a larger --buffer_km.")
            # Write the empty station table so the SNOTEL comparison knows there is nothing to compare
            pd.DataFrame(columns=["id", "name", "elevation_m", "distance_from_aoi_km", "lon", "lat"]).to_csv(
                os.path.join(out_dir, "SNOTEL_stations.csv"), index=False
                )
            return stations
    print(f"SNOTEL stations: {', '.join(f'{r.name} ({r.id})' for r in stations.itertuples())}")

    stations_table = pd.DataFrame(stations.drop(columns="geometry", errors="ignore"))
    if "geometry" in stations:
        stations_table["lon"], stations_table["lat"] = stations.geometry.x, stations.geometry.y
    stations_table.to_csv(os.path.join(out_dir, "SNOTEL_stations.csv"), index=False)

    failed = []
    for station in stations.itertuples():
        # The NRCS server occasionally returns temporary errors, so retry before giving up
        for attempt in range(1, N_RETRIES + 1):
            try:
                data = get_station_data(station.id, start_date, end_date, name=station.name)
                break
            except Exception as e:
                print(f"{station.name} ({station.id}): request failed (attempt {attempt}/{N_RETRIES}): "
                      f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
                if attempt < N_RETRIES:
                    time.sleep(RETRY_WAIT_S)
        else:
            failed.append(station.id)
            continue

        if data is None:
            print(f"{station.name} ({station.id}): no data for {start_date} to {end_date}")
            continue
        out_file = os.path.join(out_dir, f"SNOTEL_{station.id.replace(':', '_')}_daily.csv")
        data.to_csv(out_file)
        print(f"{station.name} ({station.id}): {data['snow_depth_m'].notna().sum()} days of snow depth -> {out_file}")

    if failed:
        print(f"WARNING: could not download {', '.join(failed)}. Rerun this step later.")
    return stations


def main():
    args = get_parser()
    download_snotel(
        args.out_dir, args.start_date, args.end_date, 
        aoi_file=args.aoi_file,
        sites=args.sites, 
        buffer_km=args.buffer_km
        )


if __name__ == "__main__":
    main()
