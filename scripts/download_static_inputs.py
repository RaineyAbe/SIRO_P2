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
    3. Downloads the 3DEP 1/3 arc-second DEM (via py3dep), reprojects it to HOR_CRS, converts its
       heights from NAVD88 to OUT_VERT_CRS (default: WGS84 ellipsoidal heights; see dem_height_offset),
       at the output resolution, and writes a bounding-box version and a buffer-masked version.
    4. Downloads LANDFIRE LF2025 EVT and EVH (via the LFPS v2 API) at their native 30 m, splits
       the layers, resamples them to the output resolution if it differs, and writes a
       bounding-box version and a buffer-masked version of each.
    5. Converts the EVH height-class codes (already on the output grid, resampled with nearest)
       to vegetation height in meters (see evh_to_height) and writes them as
       <site>_<EVH layer>_height_<res>m_*.tif.

Output resolution (OUT_RES_M, or --out_res_m on the command line): 
- By default each dataset keeps its native resolution (DEM 10 m, LANDFIRE 30 m). 
- If set, every output is put on the same OUT_RES_M grid, resampled with the method in RESAMPLING 
  (nearest for the LANDFIRE class codes, average for the DEM). The bounding box is then snapped to 
  a grid that both 30 m LANDFIRE cells and OUT_RES_M cells fit exactly. 
- File names include the resolution (e.g. kings_3DEP_DEM_100m_*.tif) and, for the DEM, the vertical
  datum (e.g. kings_3DEP_DEM_100m_WGS84_*.tif).

Outputs go to <BASE_DIR>/<site>/static_inputs/.

Usage:
    python download_static_inputs.py                     # native resolutions
    python download_static_inputs.py --out_res_m 100     # everything on a 100 m grid
    python download_static_inputs.py --out_vert_crs NAVD88   # keep 3DEP's NAVD88 heights
"""

import argparse
import math
import os
import shutil
import time
import re
import zipfile
import geopandas as gpd
import numpy as np
import py3dep
import rasterio
import requests
import rioxarray
from affine import Affine
import pyproj
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from shapely.geometry import box, mapping

# SETTINGS
BASE_DIR = "/Users/rdcrlrka/Research/SIRO/SIRO_P2/study_sites"
SITE_NAMES = ["east_taylor", "kings"]
BUFFER = 10e3           # meters to buffer the input watershed outlines
HOR_CRS = "EPSG:5070"   # output horizontal CRS: NAD83 / Conus Albers
SRC_CRS_3DEP = "EPSG:4269+5703"  # 3DEP heights: NAD83 + NAVD88 orthometric heights
# Output DEM heights: "NAVD88" keeps 3DEP's orthometric heights; otherwise a geographic 3D CRS whose
# ellipsoidal heights the DEM is converted to (EPSG:4979 = WGS 84 ellipsoidal height)
OUT_VERT_CRS = "EPSG:4979"
GEOID_STEP_M = 1000     # spacing (m) of the grid the geoid offset is computed on, then interpolated

DEM_RES = 10            # native DEM grid spacing (m) in HOR_CRS is ~10 m
LF_RES = 30             # native LANDFIRE resolution (m) is 30 m
OUT_RES_M = 100         # output resolution (m) for all rasters. None keeps the native resolutions.
# Resampling method for each output when OUT_RES_M is set. EVT and EVH are both LANDFIRE class codes
# (EVH codes height classes, e.g. 101-199 = tree height bins), so both use nearest.
RESAMPLING = {
    "DEM": Resampling.average,
    "LF2024_EVT": Resampling.nearest,
    "LF2024_EVH": Resampling.nearest,
}
LF_ORIGIN = (-2362425.0, 3310005.0)  # LANDFIRE CONUS grid corner in EPSG:5070. Domains snap to this grid.
LF_LAYERS = ["LF2024_EVT", "LF2024_EVH"]
EVH_LAYER = "LF2024_EVH"   # the layer converted from height classes to height in meters
# Height [m] for the EVH codes that are not tree/shrub/herb height classes. Codes not listed here become nodata. 
# See the data dictionary for more info: https://www.landfire.gov/sites/default/files/DataDictionary/2024/LF24_EVHADD.pdf 
EVH_OTHER_HEIGHTS = {
    11: 0.0,    # open water
    12: 0.0,    # snow/ice
    21: 0.0,    # developed - open space
    22: 0.0,    # developed - low intensity
    23: 0.0,    # developed - medium intensity
    24: 0.0,    # developed - high intensity
    25: 0.0,    # developed - roads
    31: 0.0,    # barren
    32: 0.0,    # quarries, strip mines, gravel pits, well and wind pads
    100: 0.0,   # sparse vegetation canopy
}
LFPS_EMAIL = "rainey.k.aberle@erdc.dren.mil" #os.environ.get("LFPS_EMAIL", "")  # required by the LFPS v2 API
OVERWRITE = False           # re-download / rewrite outputs that already exist

LFPS_URL = "https://lfps.usgs.gov/api/job"


# --- AOI preprocessing ---
def preprocess_aoi(
        aoi_path, out_dir, site, buffer=BUFFER, crs=HOR_CRS, snap=LF_RES,
        origin=LF_ORIGIN, overwrite=False
        ):
    """
    Buffer the watershed outline and build the bounding box of the buffered area.

    The bounding box is snapped outward onto a grid with cell edges at `origin` + n * `snap`
    (see snap_size()), so LANDFIRE cells and output cells fit the box exactly.
    Returns (buffer_gdf, bbox_gdf), both in `crs`,
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

    # Create bounding box for buffered AOI (rebuilt if an existing one is not on the `snap` grid,
    # e.g. it was made for a different output resolution)
    if os.path.exists(bbox_file) and not overwrite:
        bbox_gdf = gpd.read_file(bbox_file)
        offsets = np.array(bbox_gdf.total_bounds) - np.array(origin * 2)
        if not np.allclose(offsets / snap, np.round(offsets / snap)):
            print(f"Existing bounding box is not on the {snap:g} m grid, rebuilding it.")
            overwrite = True
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


    return buffer_gdf, bbox_gdf


def snap_size(out_res=None):
    """
    Grid spacing (m) the bounding box is snapped to: the smallest multiple of the LANDFIRE and DEM
    resolutions and the output resolution, so each of those grids fits the box exactly.
    """
    resolutions = [LF_RES, DEM_RES] + ([out_res] if out_res else [])
    if any(float(r) != int(r) for r in resolutions):
        raise ValueError(f"Resolutions must be whole meters, got {resolutions}")
    return math.lcm(*(int(r) for r in resolutions))


def to_output_grid(da, bbox_gdf, res, resampling, crs=HOR_CRS):
    """
    Resample `da` onto the `res` grid that exactly covers the bounding box.
    """
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    return da.rio.reproject(
        crs, 
        transform=Affine(res, 0, xmin, 0, -res, ymax), 
        shape=(height, width),
        resampling=resampling, 
        nodata=da.rio.nodata
        )


def write_bbox_and_buffer(da, buffer_gdf, bbox_gdf, out_stem, tag):
    """
    Write two clipped versions of `da` on the same grid, named to match the shapefiles:
        <stem>_<tag>.tif      : Clipped to the buffered outline (*_buffer10km.shp). 
                                Cells outside the outline are nodata.
        <stem>_<tag>_bbox.tif : Clipped to the bounding box (*_buffer10km_bbox.shp)
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
        layers=LF_LAYERS, crs=HOR_CRS, out_res=None, cleanup=True
        ):
    """
    Split the LFPS multi-band GeoTIFF into one raster per band, named after the layer
    (e.g. LF2025_EVT), resample it to `out_res` (if given and not 30 m) with the layer's method
    in RESAMPLING, and crop each to the buffered outline and the bounding box.
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
        band = lf.isel(band=[i])
        if band.rio.nodata is None:
            band = band.rio.write_nodata(-9999)

        if out_res and out_res != LF_RES:
            resampling = RESAMPLING.get(layer, Resampling.nearest)
            print(f"Resampling {layer} from {LF_RES} m to {out_res:g} m ({resampling.name})")
            band = to_output_grid(band, bbox_gdf, out_res, resampling, crs=crs)
        band.attrs.pop("long_name", None)
        band.attrs["layer"] = layer
        write_bbox_and_buffer(
            band, buffer_gdf, bbox_gdf,
            os.path.join(out_dir, f"{site}_{layer}_{res_tag(band)}"), tag
            )

        # EVH: convert the (nearest-resampled) height classes to height [m] on the output grid
        if layer == EVH_LAYER:
            height = evh_height_raster(band)
            height.attrs["layer"] = f"{layer}_height"
            write_bbox_and_buffer(
                height, buffer_gdf, bbox_gdf,
                os.path.join(out_dir, f"{site}_{layer}_height_{res_tag(height)}"), tag
                )
    lf.close()

    # Remove the unprocessed download
    if cleanup:
        shutil.rmtree(extract_dir)
        os.remove(zip_path)
        print(f"Removed {os.path.basename(zip_path)} and {os.path.basename(extract_dir)}/")


def evh_to_height(codes, nodata=None):
    """
    Vegetation height [m] from LANDFIRE EVH codes (LANDFIRE EVH data dictionary):
        101-199  tree height  = code - 100        (1 m to >= 99 m)
        201-230  shrub height = (code - 200) / 10  (0.1 m to >= 3.0 m)
        301-310  herb height  = (code - 300) / 10  (0.1 m to >= 1.0 m)
    The top code in each range is a lower bound (e.g. 199 means at least 99 m). Other codes use
    EVH_OTHER_HEIGHTS, and codes not listed there (and nodata) become NaN.
    """
    codes = np.asarray(codes)
    height = np.full(codes.shape, np.nan, dtype="float32")
    tree = (codes >= 101) & (codes <= 199)
    shrub = (codes >= 201) & (codes <= 230)
    herb = (codes >= 301) & (codes <= 310)
    height[tree] = codes[tree] - 100
    height[shrub] = (codes[shrub] - 200) / 10
    height[herb] = (codes[herb] - 300) / 10
    for code, value in EVH_OTHER_HEIGHTS.items():
        height[codes == code] = value
    if nodata is not None:
        height[codes == nodata] = np.nan
    return height


def evh_height_raster(band):
    """
    Vegetation height [m] DataArray from an EVH code DataArray, on the same grid.
    """
    height = band.copy(data=evh_to_height(band.values, band.rio.nodata)).astype("float32")
    height.encoding = {}  # drop the integer code fill value carried over from the EVH band
    height = height.rio.write_nodata(np.nan, encoded=False)
    height.attrs = {"units": "m", "long_name": "Vegetation height from LANDFIRE EVH classes"}
    return height


def landfire_outputs_exist(out_dir, site, tag, res, layers=LF_LAYERS):
    """
    True if the final buffer and bbox rasters exist for every layer (and the EVH height) at
    resolution `res` (m).
    """
    names = list(layers) + ([f"{EVH_LAYER}_height"] if EVH_LAYER in layers else [])
    for name in names:
        for suffix in (f"{tag}.tif", f"{tag}_bbox.tif"):
            if not os.path.exists(os.path.join(out_dir, f"{site}_{name}_{res:g}m_{suffix}")):
                return False
    return True


def res_tag(da):
    return f"{abs(da.rio.resolution()[0]):g}m"


# --- 3DEP download function ---
def vert_tag(out_vert_crs):
    """
    Short name of the output vertical datum for file names, e.g. "NAVD88" or "WGS84".
    """
    if str(out_vert_crs).upper() in ("NAVD88", "EPSG:5703"):
        return "NAVD88"
    return re.sub(r"[^0-9A-Za-z]+", "", CRS.from_user_input(out_vert_crs).name)


def dem_height_offset(dem, crs, out_vert_crs, step=GEOID_STEP_M):
    """
    Height [m] to add to NAVD88 heights to get ellipsoidal heights in `out_vert_crs`, on the DEM grid.

    The NAVD88 -> ellipsoid offset (mainly the GEOID18 geoid height, about -10 to -35 m across the
    western US) varies smoothly, so it is computed with PROJ on a coarse grid (`step` m) covering the
    DEM and interpolated bilinearly to the DEM cells. PROJ downloads the geoid grid if needed, and an
    error is raised if it can only use a transformation without it.
    """
    pyproj.network.set_network_enabled(active=True)  # fetch the GEOID18 grid from the PROJ CDN if needed
    transformer = Transformer.from_crs(SRC_CRS_3DEP, out_vert_crs, always_xy=True, only_best=True)
    print(f"Vertical transformation: {transformer.description}")

    # Coarse grid that includes the DEM's first and last cell centres, so no extrapolation is needed
    x, y = dem.x.values, dem.y.values
    xs = np.linspace(x[0], x[-1], max(2, int(np.ceil(abs(x[-1] - x[0]) / step)) + 1))
    ys = np.linspace(y[0], y[-1], max(2, int(np.ceil(abs(y[-1] - y[0]) / step)) + 1))
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(crs, "EPSG:4269", always_xy=True).transform(xx, yy)
    _, _, h = transformer.transform(lon, lat, np.zeros_like(lon))  # ellipsoidal height of NAVD88 = 0
    h = np.asarray(h, dtype="float64")
    if not np.all(np.isfinite(h)) or np.nanmax(np.abs(h)) < 1:
        raise RuntimeError(
            "NAVD88 -> ellipsoid offset failed (missing geoid grid?). Check that PROJ can download "
            "grids (PROJ_NETWORK=ON) or install them with `projsync --source-id us_noaa`."
            )
    print(f"NAVD88 -> {vert_tag(out_vert_crs)} offset over the domain: {h.min():.2f} to {h.max():.2f} m")

    offset = xr.DataArray(h, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return offset.interp(x=x, y=y, method="linear").values.astype("float32")


def download_3dep(
        bbox_gdf, crs=HOR_CRS, out_vert_crs=OUT_VERT_CRS, res=DEM_RES, resampling=RESAMPLING["DEM"]
        ):
    """
    Download the 3DEP 1/3 arc-second (~10 m) seamless DEM over the bounding box, reproject it onto
    a `res` grid in `crs` that exactly covers the bounding box, and convert its NAVD88 heights to
    `out_vert_crs` ellipsoidal heights (unless out_vert_crs is "NAVD88").
    """
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    # pull a slightly larger area in the DEM's native geographic CRS so the
    # reprojected grid has no empty edges
    pad = 5 * max(res, DEM_RES)
    dem = py3dep.static_3dep_dem(
        (xmin - pad, ymin - pad, xmax + pad, ymax + pad), crs=crs, resolution=10
        )
    dem = dem.squeeze(drop=True) if "band" in dem.dims and dem.sizes["band"] == 1 else dem

    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    dem = dem.rio.reproject(
        crs, transform=Affine(res, 0, xmin, 0, -res, ymax),
        shape=(height, width), resampling=resampling, nodata=np.nan
        )
    dem = dem.astype("float32")

    if vert_tag(out_vert_crs) == "NAVD88":
        # tag the vertical datum; fall back to horizontal-only if the local PROJ cannot build the
        # compound CRS
        try:
            dem = dem.rio.write_crs(CRS.from_user_input(f"{crs}+5703"))
        except Exception as err:  # pragma: no cover
            print(f"Could not write compound CRS ({err}). Writing {crs} only")
        dem.attrs.update({"units": "meters", "vertical_datum": "NAVD88 (EPSG:5703)"})
    else:
        # Ellipsoidal heights cannot be part of a compound CRS, so the file keeps the horizontal CRS
        # and the vertical datum is recorded in the attributes
        dem = dem + dem_height_offset(dem, crs, out_vert_crs)
        dem = dem.astype("float32").rio.write_crs(crs).rio.write_nodata(np.nan, encoded=False)
        target = CRS.from_user_input(out_vert_crs)
        dem.attrs.update({
            "units": "meters",
            "vertical_datum": f"{target.name} ellipsoidal height ({out_vert_crs})",
            "source_vertical_datum": "NAVD88 (EPSG:5703), converted with PROJ",
        })
    dem.name = "elevation"
    return dem


def get_parser():
    parser = argparse.ArgumentParser(description="Create modeling domains and download static inputs (3DEP DEM, LANDFIRE).")
    parser.add_argument(
        "--out_res_m", default=OUT_RES_M, type=float,
        help="Output resolution (m) for all rasters (default: native, DEM 10 m and LANDFIRE 30 m)."
        )
    parser.add_argument(
        "--out_vert_crs", default=OUT_VERT_CRS, type=str,
        help=f"Output DEM heights: a geographic 3D CRS for ellipsoidal heights (default: {OUT_VERT_CRS}, "
             "WGS 84), or NAVD88 to keep 3DEP's orthometric heights."
        )
    return parser.parse_args()


def main():
    args = get_parser()
    out_res = args.out_res_m
    snap = snap_size(out_res)
    dem_res = out_res or DEM_RES
    lf_res = out_res or LF_RES
    print(f"Output resolution: DEM {dem_res:g} m, LANDFIRE {lf_res:g} m (bounding box snapped to {snap} m)")
    out_vert_crs = args.out_vert_crs
    print(f"DEM vertical datum: {vert_tag(out_vert_crs)}")

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
            aoi_path, out_dir=aoi_dir, site=site, snap=snap, overwrite=OVERWRITE
            )

        # download: 3DEP DEM
        dem_stem = os.path.join(out_dir, f"{site}_3DEP_DEM_{dem_res:g}m_{vert_tag(out_vert_crs)}")
        if OVERWRITE or not os.path.exists(f"{dem_stem}_{tag}_bbox.tif"):
            print("Downloading 3DEP DEM ...")
            dem = download_3dep(bbox_gdf, out_vert_crs=out_vert_crs, res=dem_res)
            write_bbox_and_buffer(dem, buffer_gdf, bbox_gdf, dem_stem, tag)
        else:
            print("DEM already exists, skipping.")

        # download: LANDFIRE EVT/EVH
        zip_path = os.path.join(out_dir, f"{site}_LF2025_{tag}.zip")
        if OVERWRITE or not landfire_outputs_exist(out_dir, site, tag, lf_res):
            # reuse a zip left over from an interrupted run
            if OVERWRITE or not os.path.exists(zip_path):
                print("Requesting LANDFIRE layers ...")
                download_landfire(bbox_gdf, zip_path)
            else:
                print(f"Using existing {os.path.basename(zip_path)}")
            process_landfire(zip_path, bbox_gdf, buffer_gdf, out_dir, site, tag, out_res=out_res)
        else:
            print("LANDFIRE layers already exist, skipping.")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
