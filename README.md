## CellOracle pipeline

Last updated: 2025-11-26

## Overview

This repository contains a reproducible pipeline to build CellOracle gene regulatory network (GRN) models from single-cell RNA-seq data, run perturbation simulations, and visualize predicted cell-state shifts. The pipeline is centered on `RunCellOracle.py` and configured via `config.yaml`.

## Features

- Load AnnData (.h5ad) input
- Build or import base GRN (mouse/human)
- Perform KNN-based imputation and GRN fitting
- Simulate gene perturbations and estimate cell-state transitions
- Visualize results (UMAPs, quiver/vector fields, pseudotime plots)

## Quick links

- Configuration: `config.yaml`
- Runner script: `script.bash`
- Main pipeline: `RunCellOracle.py`

## Requirements

- Python 3.8+ (test with the project's environment)
- Key Python packages (examples):
	- scanpy
	- pandas
	- matplotlib
	- numpy
	- palantir
	- celloracle

Install packages into a conda environment or virtualenv. We recommend pinning versions used in your lab environment. Example (conda):

```bash
conda create -n celloracle python=3.9
conda activate celloracle
pip install scanpy pandas matplotlib numpy palantir celloracle
```

Note: `celloracle` has additional system and bioinformatics dependencies; follow its installation docs if you see import errors.

## Inputs

- An AnnData file (.h5ad) with at least:
	- `.obs` metadata including the column named in `celltype_colname`
	- `.layers['raw_count']` (raw counts matrix) used by the pipeline

- `config.yaml` controls runtime parameters (see section below).

## Configuration (`config.yaml`)

The main options used by `RunCellOracle.py` (example values from repository):

- `input_data`: path to the input .h5ad file (must be .h5ad)
- `celltype_colname`: column name in `.obs` that describes cell types/clusters
- `target_gene`: the gene to simulate perturbations for (e.g. "BCL6")
- `n_cells_downsample`: integer, downsample to this many cells for faster runs
- `base_GRN_model`: either `mouse` or `human` to load built-in priors
- `oracle_file_name`: filename prefix for saved Oracle object
- `grn_links_file_name`: filename prefix for GRN links file
- `min_mass`: minimum mass for selecting grid points in vector-field analyses
- `use_manual_start`: bool, whether to use a manual start cell for pseudotime
- `start_cell` / `start_celltype`: values used to choose start for palantir
- `dev_gradient_file_name`: filename prefix for developmental gradient output
- `save_folder`: folder where PNGs and outputs are saved

Edit these values before running, or create a copy of `config.yaml` per experiment.

## Running the pipeline

Two recommended ways to run:

1) Use the provided launcher script (recommended for reproducible runs):

```bash
cd CellOracle_pipeline
./script.bash
```

This will run `python RunCellOracle.py --config config.yaml` and save figures/outputs to the `save_folder` configured.

2) Run Python directly (for debugging or stepwise execution):

```bash
python RunCellOracle.py --config config.yaml
```

If you want to run only parts of the pipeline, open `RunCellOracle.py` and run selected sections in an interactive session or Jupyter notebook. The script is linear and well-commented for this purpose.

## Outputs

By default outputs are PNG figures and hdf5 celloracle objects written near the working directory or to `save_folder` as configured. Example outputs created by the script:

- `{save_folder}/celltype_condition_targetGene_umap.png`
- `{save_folder}/imputed_count_histogram.png`
- `{save_folder}/quiver_plot.png`
- `{oracle_file_name}.oracle` — saved Oracle object (HDF5)
- `{grn_links_file_name}_{unit}.celloracle.links` — GRN links for the specified unit
- `{dev_gradient_file_name}.gradient` — developmental gradient object (HDF5)

Inspect `RunCellOracle.py` to see additional filenames and plotting calls; many intermediate files are saved for downstream analyses.

## File structure (key files)

- `RunCellOracle.py` — main pipeline driver (preprocessing, GRN fit, simulation, visualization)
- `config.yaml` — pipeline configuration
- `script.bash` — small wrapper to call `RunCellOracle.py`
- `*.h5ad` — input AnnData files (example: `cardiomyocyte.h5ad`)