#!/usr/bin/env python

"""
Download SNOTEL and CDEC snow depth and SWE for stations in and near an area of interest.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Stations within --buffer_km of the AOI polygon are found for each network (--networks):
    SNOTEL  NRCS SNOTEL stations, daily data
    CDEC    California Data Exchange Center snow sensor stations (daily data) and snow courses (monthly
            manual measurements, dated by the day they were measured)

Outputs, in --out_dir:
    SNOTEL_stations.csv             all stations: network, station type, ID, name, location, elevation,
                                    distance from the AOI, and the station's data file
    SNOTEL_<ID>_daily.csv           daily snow depth and SWE [m] (e.g., SNOTEL_637_ID_SNTL_daily.csv)
    CDEC_<ID>_daily.csv             daily snow depth and SWE [m] at CDEC sensor stations
    CDEC_<ID>_snow_course.csv       snow course snow depth and SWE [m] on each measurement date

Usage:
    # With area of interest vector file:
    python -m SIRO_P2.download_SNOTEL \\
        --aoi_file /path/to/watershed_outline.shp \\
        --start_date 2022-10-01 \\
        --end_date 2025-06-30 \\
        --out_dir /path/to/SNOTEL

    # With station IDs (SNOTEL IDs contain ":", e.g., 637:ID:SNTL; CDEC IDs are 3 letters, e.g., TNY):
    python -m SIRO_P2.download_SNOTEL \\
        --sites 637:ID:SNTL 978:ID:SNTL \\
        --start_date 2022-10-01 \\
        --end_date 2025-06-30 \\
        --out_dir /path/to/SNOTEL
"""

import argparse
import os
import time

import geopandas as gpd
import pandas as pd
from metloom.pointdata import CDECPointData, SnotelPointData
from shapely.geometry import Point

FEET_TO_M = 0.3048
TO_METERS = {           # station units (lowercase) -> meters
    "in": 0.0254,
    "inches": 0.0254,
}
N_RETRIES = 3           # attempts per station if the server returns an error
RETRY_WAIT_S = 30       # seconds to wait between attempts
VARIABLES = {           # metloom variable -> output column [m]
    "SNOWDEPTH": "snow_depth_m",
    "SWE": "SWE_m",
}
NETWORKS = {
    "SNOTEL": SnotelPointData,
    "CDEC": CDECPointData,
}
STATION_COLUMNS = ["network", "station_type", "id", "name", "elevation_m", "distance_from_aoi_km", "data_file", "geometry"]
STATIONS_FILE = "SNOTEL_stations.csv"


def get_parser():
    parser = argparse.ArgumentParser(description="Download SNOTEL and CDEC snow depth and SWE for an area of interest.")
    parser.add_argument("--out_dir", required=True, type=str, help="Path where downloaded files will be saved.")
    parser.add_argument("--start_date", required=True, type=str, help="First date to download (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, type=str, help="Last date to download, inclusive (YYYY-MM-DD).")
    parser.add_argument("--aoi_file", default=None, type=str, help="Vector file (e.g., .gpkg, .shp) of the area of interest, used to find stations.")
    parser.add_argument("--sites", default=None, type=str, nargs="+", help="Station IDs (SNOTEL, e.g., 637:ID:SNTL; CDEC, e.g., TNY), instead of finding them from --aoi_file.")
    parser.add_argument("--networks", default=list(NETWORKS), type=str, nargs="+", choices=list(NETWORKS), help="Station networks to search with --aoi_file (default: SNOTEL CDEC).")
    parser.add_argument("--buffer_km", default=1.0, type=float, help="Include stations within this distance of the AOI [km] (default: 1).")
    args = parser.parse_args()
    if (args.aoi_file is None) == (args.sites is None):
        parser.error("Specify exactly one of --aoi_file or --sites.")
    return args


# ----- STATIONS -----
def with_retries(func, label):
    """
    func() with up to N_RETRIES attempts, as the NRCS and CDEC servers occasionally drop connections.
    Returns None if every attempt fails.
    """
    for attempt in range(1, N_RETRIES + 1):
        try:
            return func()
        except Exception as e:
            print(f"{label}: request failed (attempt {attempt}/{N_RETRIES}): "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ''}")
            if attempt < N_RETRIES:
                time.sleep(RETRY_WAIT_S * attempt)
    return None


def data_file(network, station_type, station_id):
    suffix = "snow_course" if station_type == "snow_course" else "daily"
    return f"{network}_{station_id.replace(':', '_')}_{suffix}.csv"


def station_row(network, station_type, point, distance_km=None):
    lon, lat, elev_ft = point.metadata.x, point.metadata.y, point.metadata.z
    return {
        "network": network,
        "station_type": station_type,
        "id": point.id,
        "name": point.name,
        "elevation_m": round(elev_ft * FEET_TO_M, 1) if elev_ft is not None else None,
        "distance_from_aoi_km": round(distance_km, 2) if distance_km is not None else None,
        "data_file": data_file(network, station_type, point.id),
        "geometry": Point(lon, lat),
    }


def to_geodataframe(rows):
    """
    Station GeoDataFrame (EPSG:4326); empty but with the usual columns if there are no stations.
    """
    if not rows:
        return gpd.GeoDataFrame(columns=STATION_COLUMNS, geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, columns=STATION_COLUMNS, geometry="geometry", crs="EPSG:4326")


def find_stations(aoi_file, buffer_km=1.0, networks=None):
    """
    Stations within buffer_km of the AOI for each network, as a GeoDataFrame (EPSG:4326) with the
    network, station type ("sensor" or "snow_course"), ID, name, elevation [m], distance from the AOI [km]
    (0 = inside), and data file name.
    """
    aoi = gpd.read_file(aoi_file)
    aoi_projected = aoi.to_crs(aoi.estimate_utm_crs())
    aoi_polygon = aoi_projected.union_all()

    # Search area: the AOI buffered by buffer_km, as one polygon (metloom only uses the first row's bounds).
    # within_geometry=True drops stations outside this polygon before metloom checks each station's
    # metadata, which for CDEC is one request per station; searching the whole bounding box made enough
    # requests that the CDEC server started resetting connections.
    search_area = gpd.GeoDataFrame(geometry=[aoi_polygon.buffer(buffer_km * 1000)], crs=aoi_projected.crs).to_crs(4326)

    rows, failed = [], []
    for network in networks or NETWORKS:
        cls = NETWORKS[network]
        variables = [getattr(cls.ALLOWED_VARIABLES, v) for v in VARIABLES]
        # CDEC returns either snow courses or sensor stations per search, so search for both
        searches = [("sensor", False), ("snow_course", True)] if network == "CDEC" else [("sensor", None)]
        for station_type, snow_courses in searches:
            kwargs = {"within_geometry": True}
            if snow_courses is not None:
                kwargs["snow_courses"] = snow_courses
            label = f"{network} {station_type.replace('_', ' ')} station search"
            candidates = with_retries(lambda: cls.points_from_geometry(search_area, variables, **kwargs), label)
            if candidates is None:
                failed.append(label)
                continue

            # Keep stations within buffer_km of the AOI polygon itself
            for point in candidates:
                location = gpd.GeoSeries([Point(point.metadata.x, point.metadata.y)], crs=4326).to_crs(aoi_projected.crs).iloc[0]
                distance_km = location.distance(aoi_polygon) / 1000
                if distance_km <= buffer_km:
                    rows.append(station_row(network, station_type, point, distance_km))
    if failed:
        print(f"WARNING: {', '.join(failed)} failed after {N_RETRIES} attempts, so those stations are missing. "
              "Rerun this step later.")
    return to_geodataframe(rows)


def station_locations(sites):
    """
    Stations for the given IDs: IDs containing ":" are SNOTEL (e.g., 637:ID:SNTL), others CDEC sensor
    stations (e.g., TNY).
    """
    rows = []
    for site in sites:
        network = "SNOTEL" if ":" in site else "CDEC"
        rows.append(station_row(network, "sensor", NETWORKS[network](site, site)))
    return to_geodataframe(rows)


# ----- DATA -----
def local_date(times):
    """
    Calendar date in Pacific Standard Time (CDEC's fixed UTC-8) of datetimes, as naive midnight timestamps.
    """
    times = pd.to_datetime(times)
    if times.dt.tz is not None:
        times = times.dt.tz_convert("Etc/GMT+8").dt.tz_localize(None)
    return times.dt.normalize()


def get_station_data(site_id, start_date, end_date, name=None, network="SNOTEL", station_type="sensor"):
    """
    Snow depth and SWE [m] for one station, indexed by date: daily for sensor stations, the measurement
    dates for snow courses.
    """
    point = NETWORKS[network](site_id, name or site_id)
    variables = [getattr(point.ALLOWED_VARIABLES, v) for v in VARIABLES]
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if station_type == "snow_course":
        df = point.get_snow_course_data(start, end, variables)
    else:
        df = point.get_daily_data(start, end, variables)
    if df is None or df.empty:
        return None

    df = df.reset_index()
    if station_type == "snow_course":
        # Local (Pacific Standard Time) calendar date the course was measured, else the reported date
        dates = local_date(df["datetime"])
        if "measurementDate" in df:
            dates = local_date(df["measurementDate"]).fillna(dates)
    else:
        dates = df["datetime"]
    out = pd.DataFrame({"date": dates})
    for variable, column in VARIABLES.items():
        name_in_df = getattr(point.ALLOWED_VARIABLES, variable).name
        if name_in_df not in df:
            out[column] = float("nan")
            continue
        units = {str(u).lower() for u in df[f"{name_in_df}_units"].dropna().unique()}
        if len(units) != 1 or not units <= set(TO_METERS):
            raise ValueError(f"Unexpected {variable} units for {network} {site_id}: {units}")
        out[column] = (df[name_in_df] * TO_METERS[units.pop()]).round(4)
    return out.set_index("date")


def download_snotel(out_dir, start_date, end_date, aoi_file=None, sites=None, buffer_km=1.0, networks=None):
    """
    Download SNOTEL and CDEC snow depth and SWE between start_date and end_date (inclusive) for stations
    within buffer_km of aoi_file (or the given station IDs). Returns the stations table.
    """
    os.makedirs(out_dir, exist_ok=True)
    networks = networks or list(NETWORKS)
    stations_file = os.path.join(out_dir, STATIONS_FILE)

    stations = station_locations(sites) if sites else find_stations(aoi_file, buffer_km, networks)
    stations_table = pd.DataFrame(stations.drop(columns="geometry"))
    stations_table["lon"] = stations.geometry.x if not stations.empty else []
    stations_table["lat"] = stations.geometry.y if not stations.empty else []
    stations_table.to_csv(stations_file, index=False)  # written even if empty, for the comparison step
    if stations.empty:
        print(f"No {' or '.join(networks)} stations found within {buffer_km} km of the AOI. Try a larger --buffer_km.")
        return stations
    for (network, station_type), group in stations.groupby(["network", "station_type"]):
        print(f"{network} {station_type.replace('_', ' ')} stations: {', '.join(f'{r.name} ({r.id})' for r in group.itertuples())}")

    failed = []
    for station in stations.itertuples():
        label = f"{station.network} {station.name} ({station.id})"
        result = with_retries(  # (data,) on success, so "no data" (None) is distinct from a failed request
            lambda: (get_station_data(
                station.id, start_date, end_date, name=station.name,
                network=station.network, station_type=station.station_type
                ),),
            label
            )
        if result is None:
            failed.append(f"{station.network} {station.id}")
            continue
        data = result[0]
        if data is None:
            print(f"{label}: no data for {start_date} to {end_date}")
            continue
        out_file = os.path.join(out_dir, station.data_file)
        data.to_csv(out_file)
        unit = "measurements" if station.station_type == "snow_course" else "days"
        print(f"{label}: {data['snow_depth_m'].notna().sum()} {unit} of snow depth, "
              f"{data['SWE_m'].notna().sum()} of SWE -> {out_file}")

    if failed:
        print(f"WARNING: could not download {', '.join(failed)}. Rerun this step later.")
    return stations


def main():
    args = get_parser()
    download_snotel(
        args.out_dir,
        args.start_date,
        args.end_date,
        aoi_file=args.aoi_file,
        sites=args.sites,
        buffer_km=args.buffer_km,
        networks=args.networks
        )


if __name__ == "__main__":
    main()
