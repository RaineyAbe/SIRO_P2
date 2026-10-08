# Snow Informed Reservoir Operations Model Intercomparison Experiment: Phase 2

## Installation

The core functions live in the `SIRO_P2` package. Create the conda environment, which also installs the
package in editable mode (code changes take effect without reinstalling):

```
conda env create -f environment.yml
conda activate siro_p2
```

Or, in an existing environment, from the repository root:

```
pip install -e .
```

Then import functions with, e.g., `from SIRO_P2.compare_lidar import compare_lidar`, or run a step from the
command line with `python -m SIRO_P2.compare_lidar --help` or the matching command (`siro-compare-lidar --help`).

## Structure

```
SIRO_P2/                         # repository root
├── environment.yml
├── pyproject.toml               # package metadata and dependencies (pip install -e .)
├── SIRO_P2/                     # core package
│   ├── prepare_models.py        # compile raw model outputs into one netCDF per model
│   ├── prepare_lidar.py         # aggregate LiDAR snow depth onto common 100 m and 2000 m grids
│   ├── download_fSCA.py         # download SPIReS fSCA
│   ├── download_SNOTEL.py       # download SNOTEL snow depth and SWE
│   ├── download_static_inputs.py  # modeling domains, 3DEP DEM, LANDFIRE
│   ├── compare_lidar.py
│   ├── compare_fSCA.py
│   └── compare_SNOTEL.py
├── scripts/                     # wrappers and other analysis scripts
│   ├── SIRO_P2_pipeline.py      # specify I/O, settings, steps to run
│   ├── run_SIRO_P2_pipeline.sh
│   ├── crosscheck_compare_lidar_P1.py
│   └── crosscheck_compare_SNOTEL_notebook.py
├── analysis_notebooks/
└── SNODAS/
```
