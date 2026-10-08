#!/usr/bin/env python

"""
Create modeling domains and download static inputs.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL

Datasets downloaded:
    - USGS 3DEP 1/3 arc second DEMs
    - LANDFIRE: 2025 existing vegetation type (EVT) and existing vegetation height (EVH)

For each site in SITE_NAMES this script:
    1. Loads <BASE_DIR>/<site>/AOIs/<site>_watershed_outline.shp
    2. Buffers it by BUFFER meters and builds the bounding box of the buffered outline.
       Both domains are written to <BASE_DIR>/<site>/AOIs/
    3. Downloads the 3DEP 1/3 arc-second DEM (via py3dep), reprojects it to HOR_CRS
       at DEM_RES, and writes a bounding-box version and a buffer-masked version.
    4. Downloads LANDFIRE LF2025 EVT and EVH (via the LFPS v2 API), splits the
       layers, and writes a bounding-box version and a buffer-masked version of each.

Outputs go to <BASE_DIR>/<site>/static_inputs/.
"""

import glob
import os
import shutil
import time
import zipfile
import geopandas as gpd
import numpy as np
import py3dep
import rasterio
import requests
import rioxarray
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from shapely.geometry import box, mapping

# SETTINGS
BASE_DIR = "/Users/rdcrlrka/Research/SIRO/SIRO_P2/study_sites"
SITE_NAMES = ["east_taylor", "kings"]
BUFFER = 10e3           # meters to buffer the input watershed outlines
HOR_CRS = "EPSG:5070"   # output horizontal CRS: NAD83 / Conus Albers
VERT_CRS = "EPSG:5703"  # output vertical CRS: NAVD88 (3DEP is already NAVD88)

DEM_RES = 10            # output DEM grid spacing (m) in HOR_CRS is ~10 m
LF_RES = 30             # LANDFIRE output resolution (m) is 30 m
LF_ORIGIN = (-2362425.0, 3310005.0)  # LANDFIRE CONUS grid corner in EPSG:5070; domains snap to this grid
LF_LAYERS = ["LF2025_EVT", "LF2025_EVH"]
LFPS_EMAIL = "rainey.k.aberle@erdc.dren.mil" #os.environ.get("LFPS_EMAIL", "")  # required by the LFPS v2 API
OVERWRITE = True           # re-download / rewrite outputs that already exist

LFPS_URL = "https://lfps.usgs.gov/api/job"


# --- AOI preprocessing ---
def preprocess_aoi(
        aoi_path, out_dir, site, buffer=BUFFER, crs=HOR_CRS, snap=LF_RES,
        origin=LF_ORIGIN, overwrite=False
        ):
    """
    Buffer the watershed outline and build the bounding box of the buffered area.

    The bounding box is snapped outward onto the LANDFIRE 30 m grid (cell edges at
    `origin` + n * `snap`), so LANDFIRE cells fit the box exactly and the 10 m DEM cells
    nest 3x3 inside each LANDFIRE cell. Returns (buffer_gdf, bbox_gdf), both in `crs`,
    and writes each to `out_dir` as a shapefile.
    """
    # Define output files
    tag = f"buffer{buffer / 1e3:g}km"
    buffer_file = os.path.join(
        out_dir, 
        os.path.splitext(os.path.basename(aoi_path))[0] + f"_{tag}",
        os.path.splitext(os.path.basename(aoi_path))[0] + f"_{tag}.shp"
        )
    bbox_file = os.path.join(
        out_dir, 
        os.path.splitext(os.path.basename(aoi_path))[0] + f"_{tag}_bbox",
        os.path.splitext(os.path.basename(aoi_path))[0] + f"_{tag}_bbox.shp"
        )

    # Create buffered AOI
    if not os.path.exists(buffer_file) or overwrite:
        # load the AOI
        aoi = gpd.read_file(aoi_path).to_crs(crs)
        # merge all features into a single (multi)polygon, then buffer it
        buffered = aoi.geometry.union_all().buffer(buffer)
        # create geodataframe
        buffer_gdf = gpd.GeoDataFrame(
            {"site": [site], "buffer_m": [buffer]}, geometry=[buffered], crs=crs
            )
        # save (create the shapefile's folder first)
        os.makedirs(os.path.dirname(buffer_file), exist_ok=True)
        buffer_gdf.to_file(buffer_file)
        print("Buffered outline saved to:", buffer_file)
    else:
        print("Buffered outline already exists, skipping.")
        buffer_gdf = gpd.read_file(buffer_file)

    # Create bounding box for buffered AOI
    if not os.path.exists(bbox_file) or overwrite:
        # snap the bounding box edges outward onto the LANDFIRE grid
        xmin, ymin, xmax, ymax = buffer_gdf.total_bounds
        ox, oy = origin
        xmin = ox + np.floor((xmin - ox) / snap) * snap
        ymin = oy + np.floor((ymin - oy) / snap) * snap
        xmax = ox + np.ceil((xmax - ox) / snap) * snap
        ymax = oy + np.ceil((ymax - oy) / snap) * snap
        # create geodataframe
        bbox_gdf = gpd.GeoDataFrame(
            {"site": [site], "buffer_m": [buffer]}, geometry=[box(xmin, ymin, xmax, ymax)], crs=crs
            )
        # save (create the shapefile's folder first)
        os.makedirs(os.path.dirname(bbox_file), exist_ok=True)
        bbox_gdf.to_file(bbox_file)
        print("Buffered outline bounding box saved to:", bbox_file)
    else:
        print("Buffered outline bounding box already exists, skipping.")
        bbox_gdf = gpd.read_file(bbox_file)


    return buffer_gdf, bbox_gdf


def write_bbox_and_buffer(da, buffer_gdf, bbox_gdf, out_stem, tag):
    """
    Write two clipped versions of `da` on the same grid, named to match the shapefiles:
        <stem>_<tag>.tif      : clipped to the buffered outline (*_buffer10km.shp);
                                cells outside the outline are nodata
        <stem>_<tag>_bbox.tif : clipped to the bounding box (*_buffer10km_bbox.shp)
    """
    kwargs = dict(compress="LZW", tiled=True, BIGTIFF="IF_SAFER")

    # bounding box: the box edges sit on cell edges, so this keeps whole cells only
    bbox_da = da.rio.clip_box(*bbox_gdf.total_bounds)
    bbox_da.rio.to_raster(f"{out_stem}_{tag}_bbox.tif", **kwargs)

    # buffered outline: same grid as the bbox file, nodata outside the outline
    buffer_da = bbox_da.rio.clip(
        [mapping(g) for g in buffer_gdf.geometry], crs=buffer_gdf.crs,
        all_touched=True, drop=False
        )
    buffer_da.rio.to_raster(f"{out_stem}_{tag}.tif", **kwargs)
    print(f"Wrote {os.path.basename(out_stem)}_{tag}.tif and _{tag}_bbox.tif")


# --- Landfire download ---
def download_landfire(
        bbox_gdf, zip_path, layers=LF_LAYERS, email=LFPS_EMAIL,
        crs=HOR_CRS, res=LF_RES, poll=15, timeout=3600
        ):
    """
    Request `layers` from the LFPS v2 API for the bounding box and save the zip.

    LFPS takes a WGS84 bounding box ("xmin ymin xmax ymax") and returns a zip holding a
    multi-band GeoTIFF (one band per layer, in Layer_List order) in `crs`.
    """
    if not email:
        raise RuntimeError(
            "LFPS v2 requires an email: set LFPS_EMAIL in the script or `export LFPS_EMAIL=you@example.com`."
            )

    # Pad the WGS84 box slightly so the reprojected result fully covers the bbox
    w, s, e, n = bbox_gdf.to_crs("EPSG:4326").total_bounds
    pad = 0.01
    aoi = f"{w - pad:.5f} {s - pad:.5f} {e + pad:.5f} {n + pad:.5f}"

    # Design the request
    params = {
        "Email": email,
        "Layer_List": ";".join(layers),
        "Area_of_Interest": aoi,
        "Output_Projection": CRS.from_user_input(crs).to_epsg(),
    }
    if res != 30:  # 30 m is the LFPS default
        params["Resample_Resolution"] = res

    # Submit the request
    r = requests.get(f"{LFPS_URL}/submit", params=params, timeout=60)
    r.raise_for_status()
    job_id = r.json()["jobId"]
    print(f"LFPS job submitted: {job_id}")

    # Check on staus...
    t0 = time.time()
    while True:
        status = requests.get(
            f"{LFPS_URL}/status", params={"JobId": job_id}, timeout=60
            ).json()
        state = status.get("status", "")
        if state == "Succeeded":
            break
        if state in ("Failed", "Canceled", "Cancelled"):
            raise RuntimeError(f"LFPS job {job_id} {state}: {status.get('messages')}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"LFPS job {job_id} still '{state}' after {timeout} s")
        print(f"Status: {state} (queue position: {status.get('queuePosition')})")
        time.sleep(poll)

    with requests.get(status["outputFile"], stream=True, timeout=300) as dl:
        dl.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Downloaded {os.path.basename(zip_path)}")
    return zip_path


def process_landfire(
        zip_path, bbox_gdf, buffer_gdf, out_dir, site, tag,
        layers=LF_LAYERS, crs=HOR_CRS, cleanup=True
        ):
    """
    Split the LFPS multi-band GeoTIFF into one raster per band, named after the layer
    (e.g. LF2025_EVT), and crop each to the buffered outline and the bounding box.
    With `cleanup`, the downloaded zip and its extracted files are removed afterward.
    """
    # Unzip the download
    extract_dir = os.path.splitext(zip_path)[0]
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if len(tifs) != 1:
            raise RuntimeError(
                f"Expected one GeoTIFF in {os.path.basename(zip_path)}, found {tifs}"
                )
        zf.extractall(extract_dir)
    tif_path = os.path.join(extract_dir, tifs[0])

    # Match each layer to its band via the band description (e.g. LF2025_EVT_CONUS);
    # fall back to Layer_List order if a band has no description
    with rasterio.open(tif_path) as src:
        descriptions = list(src.descriptions)
    print(f"Bands in LANDFIRE file: {descriptions}")
    band_index = {}
    for i, layer in enumerate(layers):
        matches = [j for j, d in enumerate(descriptions) if d and d.startswith(layer)]
        band_index[layer] = matches[0] if matches else i

    lf = rioxarray.open_rasterio(tif_path, masked=False)
    if lf.rio.crs is None or CRS.from_user_input(lf.rio.crs) != CRS.from_user_input(crs):
        raise RuntimeError(f"LANDFIRE output CRS {lf.rio.crs} does not match {crs}")

    # Write each band to its own pair of files
    for layer, i in band_index.items():
        band = lf.isel(band=[i])  # kept on the LANDFIRE grid, no resampling
        if band.rio.nodata is None:
            band = band.rio.write_nodata(-9999)
        band.attrs.pop("long_name", None)
        band.attrs["layer"] = layer
        write_bbox_and_buffer(
            band, buffer_gdf, bbox_gdf,
            os.path.join(out_dir, f"{site}_{layer}_{res_tag(band)}"), tag
            )
    lf.close()

    # Remove the unprocessed download
    if cleanup:
        shutil.rmtree(extract_dir)
        os.remove(zip_path)
        print(f"Removed {os.path.basename(zip_path)} and {os.path.basename(extract_dir)}/")


def landfire_outputs_exist(out_dir, site, tag, layers=LF_LAYERS):
    """
    True if the final buffer and bbox rasters exist for every layer.
    """
    for layer in layers:
        for suffix in (f"{tag}.tif", f"{tag}_bbox.tif"):
            pattern = os.path.join(out_dir, f"{site}_{layer}_*m_{suffix}")
            if not glob.glob(pattern):
                return False
    return True


def res_tag(da):
    return f"{abs(da.rio.resolution()[0]):g}m"


# --- 3DEP download function ---
def download_3dep(bbox_gdf, crs=HOR_CRS, vert_crs=VERT_CRS, res=DEM_RES):
    """
    Download the 3DEP 1/3 arc-second (~10 m) seamless DEM over the bounding box and
    reproject it onto a `res` grid in `crs` that exactly covers the bounding box.
    """
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    # pull a slightly larger area in the DEM's native geographic CRS so the
    # reprojected grid has no empty edges
    pad = 5 * res
    dem = py3dep.static_3dep_dem(
        (xmin - pad, ymin - pad, xmax + pad, ymax + pad), crs=crs, resolution=10
        )
    dem = dem.squeeze(drop=True) if "band" in dem.dims and dem.sizes["band"] == 1 else dem

    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    dem = dem.rio.reproject(
        crs, transform=Affine(res, 0, xmin, 0, -res, ymax),
        shape=(height, width), resampling=Resampling.bilinear, nodata=np.nan
        )
    dem = dem.astype("float32")

    # tag the vertical datum (3DEP is NAVD88); fall back to horizontal-only if the
    # local PROJ cannot build the compound CRS
    try:
        compound = CRS.from_user_input(f"{crs}+{CRS.from_user_input(vert_crs).to_epsg()}")
        dem = dem.rio.write_crs(compound)
    except Exception as err:  # pragma: no cover
        print(f"Could not write compound CRS ({err}). Writing {crs} only")
    dem.attrs.update({"units": "meters", "vertical_datum": "NAVD88"})
    dem.name = "elevation"
    return dem


def main():
    # Iterate over sites
    for site in SITE_NAMES:
        print(f"\n--- {site} ---")

        # define inputs and outputs
        site_dir = os.path.join(BASE_DIR, site)
        aoi_dir = os.path.join(site_dir, "AOIs")
        out_dir = os.path.join(site_dir, "static_inputs")
        tag = f"buffer{BUFFER / 1e3:g}km"

        os.makedirs(out_dir, exist_ok=True)

        # load the AOI, then preprocess (buffer, create bounding box)
        aoi_path = os.path.join(aoi_dir, f"{site}_watershed_outline", f"{site}_watershed_outline.shp")
        buffer_gdf, bbox_gdf = preprocess_aoi(
            aoi_path, out_dir=aoi_dir, site=site, overwrite=OVERWRITE
            )

        # download: 3DEP DEM
        dem_stem = os.path.join(out_dir, f"{site}_3DEP_DEM_{DEM_RES}m")
        if OVERWRITE or not os.path.exists(f"{dem_stem}_{tag}_bbox.tif"):
            print("Downloading 3DEP DEM ...")
            dem = download_3dep(bbox_gdf)
            write_bbox_and_buffer(dem, buffer_gdf, bbox_gdf, dem_stem, tag)
        else:
            print("DEM already exists, skipping.")

        # download: LANDFIRE EVT/EVH
        zip_path = os.path.join(out_dir, f"{site}_LF2025_{tag}.zip")
        if OVERWRITE or not landfire_outputs_exist(out_dir, site, tag):
            # reuse a zip left over from an interrupted run
            if OVERWRITE or not os.path.exists(zip_path):
                print("Requesting LANDFIRE layers ...")
                download_landfire(bbox_gdf, zip_path)
            else:
                print(f"Using existing {os.path.basename(zip_path)}")
            process_landfire(zip_path, bbox_gdf, buffer_gdf, out_dir, site, tag)
        else:
            print("LANDFIRE layers already exist, skipping.")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
