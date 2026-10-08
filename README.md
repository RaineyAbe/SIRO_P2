# Snow Informed Reservoir Operations (SIRO) Model Intercomparison Experiment: Phase 2

<p align="center">
  <img src="logos/SIRO_logo_main_transparentbg.png" alt="SIRO logo by Sydney Baratta" width="300">
</p>


## Correspondence

Rainey Aberle (<Rainey.K.Aberle@erdc.dren.mil>), Wyatt Reis, Shad O'Neel, Julie Parno, and Sydney Baratta

USACE-ERDC-CRREL

## Installation

The core functions live in the `SIRO_P2` package. Create the environment using [mamba](https://mamba.readthedocs.io/en/latest/) or [conda](https://docs.conda.io/en/latest/), which also installs the
package in editable mode (code changes take effect without reinstalling):

```
mamba env create -f environment.yml
mamba activate siro_p2
```

Or, in an existing environment, from the repository root:

```
pip install -e .
```

Then import functions with, e.g.: `from SIRO_P2.compare_lidar import compare_lidar`, or run a step from the
command line with `python -m SIRO_P2.compare_lidar --help` or the matching command (`siro-compare-lidar --help`).

## Usage

The core pipeline to compare snow model outputs to validation datasets lies in `scripts/SIRO_P2_pipeline.py`, which can be run with `sh run_SIRO_P2_pipeline.sh`. 

Steps in the pipeline:

1. **Prepare models**: Load and compile all model outputs, rectify units and timestamps, and save one netCDF per model.

2. **Compare lidar**: Sample model pre-processed outputs over the lidar domain for each lidar date and calculate comparison metrics for each date (e.g., RMSE, R$^2$, bias ratio).

3. **Download and compare fractional snow-covered area (fSCA)**: Download MODIS/Terra SPIReS fSCA product for the modeling time period, convert preprocessed model outputs to fSCA using a range of SWE thresholds, and calculate comparison metrics (e.g., watershed-wide fSCA time series, melt-out day biases).

4. **Download and compare SNOTEL**: Download SNOTEL data for all stations within the watershed, sample preprocessed model outputs at each SNOTEL station, and calculate comparison metrics (e.g., Nash-Sutcliffe Efficiency and Kling-Gupta Efficiency).