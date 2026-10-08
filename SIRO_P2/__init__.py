"""
Snow-Informed Reservoir Operations (SIRO) model intercomparison, Phase 2.

USACE-ERDC-CRREL

Modules:
    prepare_models          compile raw model outputs into one netCDF per model
    prepare_lidar           aggregate LiDAR snow depth onto common 100 m and 2000 m grids
    download_fSCA           download SPIReS fSCA
    download_SNOTEL         download SNOTEL snow depth and SWE
    download_static_inputs  create modeling domains and download static inputs (3DEP DEM, LANDFIRE)
    compare_lidar           compare modeled and LiDAR snow depth
    compare_fSCA            compare modeled SCA and SPIReS fSCA
    compare_SNOTEL          compare modeled and SNOTEL snow depth and SWE
"""

__version__ = "0.1.0"
