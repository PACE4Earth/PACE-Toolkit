# PACE-Toolkit Documentation

## 1. Overview

The PACE-Toolkit is a Python-based framework designed for calculating and visualizing physical consistency diagnostics (metrics) for weather prediction models, including machine-learning-based models. It is optimized for High-Performance Computing (HPC) environments, leveraging MPI and PyTorch for scalable, parallel processing of large datasets.

The core philosophy is to provide a configurable and extensible platform for researchers to evaluate model output against fundamental physical principles like geostrophic balance, hydrostatic balance, humidity–temperature consistency, and others.

High-level workflow:

```
Config → Dataloader → Evaluator → Metric Modules → Postprocessing & Plots
```

Main components:

1. **Config (`config.json`)** – Passed to scripts. Defines datasets, metrics to compute, time & spatial ranges, number of samples, and which visualizations to create.
2. **Dataloader (`dataset.py`)** – Loads datasets (e.g., NetCDF/ERA5/GraphCast), applies filtering, and prepares PyTorch tensors. Provides samples via `__getitem__`, representing unique base+lead time combinations with the required variables.
3. **Evaluator (`evaluator.py`)** – Orchestrates the workflow. Retrieves samples from the Dataloader, runs the selected metric modules, and manages output saving. Supports distributed execution across multiple workers/nodes.
4. **Metrics (modules in `metrics/`– imported in `evaluator.py`)** –  Contain the actual implementations of the physical consistency metrics (diagnostics), e.g., geostrophic balance, hydrostatic balance, humidity–temperature consistency.
5. **Postprocessing & Plotting (`postprocess.py`)** – Aggregates the metric outputs and generates visualizations (histograms, vertical profiles, spatial slices, etc.).

## 2. General Workflow

The toolkit operates in a two-stage process, typically managed by SLURM batch scripts.

1. **Evaluation Stage (`evaluator.py`)**

   * Reads weather model data (e.g., NetCDF files).
   * Calculates a suite of physical metrics in parallel using MPI/PyTorch distributed.
   * Saves the results to a Zarr data store.
   

2. **Post-processing Stage (`postprocess.py`)**

   * Reads the Zarr data generated during the evaluation stage.
   * Creates various visualizations, such as histograms and vertical profiles of the calculated metrics.

### How to Run

On a local machine (development/testing):

```bash
python pace/evaluator.py
python pace/postprocess.py
```

On an HPC system with SLURM:

```bash
srun python pace/evaluator.py
srun python pace/postprocess.py
```

The workflow is controlled by environment variables set within the `.slurm` script:

* `CONFIG_PATH`: **(Required)** Path to the main JSON configuration file that defines the run. Can be also specified as a CLI argument: `--config path/to/config` 
* `OUTPUT_DIR_PATH`: **(Required)** Directory where the output Zarr datasets will be saved.
* `PLOTS_DIR_PATH`: Directory for saving plots generated during post-processing.

Note: Paths can be also specified directly in the config


## 3. Directory Structure

```
pace/
├── configs/          # JSON configuration files for runs, metrics, and visualizations.
├── metrics/          # Individual modules for each physical metric calculation.
├── utils/            # Helper modules for data loading, I/O, plotting, etc.
├── evaluator.py      # Main script for the evaluation stage.
├── postprocess.py    # Main script for the post-processing stage.
```

## 4. Core Components

### 4.1. Scripts

#### `evaluator.py`

Main driver for metric computation. It orchestrates the entire parallel processing pipeline.

* **Initialization**: Sets up MPI/PyTorch Distributed for parallel execution.
* **Workload Distribution**: Rank 0 process builds a list of samples (unique base+lead times). Workload is split among all workers.
* **Metric Calculation**: Uses `MetricHandler` to run the configured metric modules.
* **Parallel I/O**: Employs `MPIZarrSaver` to handle concurrent writes to Zarr.

#### `postprocess.py`

Generates visualizations from evaluation results.

* **Data Loading**: Opens the Zarr datasets produced by `evaluator.py`.
* **Plot Generation**: Uses config instructions to create histograms, vertical profiles, maps, etc.

### 4.2. Key Objects and Classes

#### Configuration Files (`configs/`)

The toolkit’s behavior is defined by JSON configuration files.

* **Main Config (`config_*.json`)**

  * `datasets`: Defines model and optional reference dataset, with name and path.
  * `metrics`: Dict specifying which metric suites to run. Values for each metric can be specific outputs or `["all"]`.
  * `time` & `spatial`: Define temporal and spatial domains (date range, lat/lon bounds, pressure levels).
  * `visualization`: Flags for enabling/disabling plot types.
  * `sample_percent`: Defines what fraction of available timesteps to evaluate (useful for testing).

* **`variables_for_metrics.json`**: Defines required variables for each metric.

* **`aliases.json`**: Maps canonical variable names to alternative names in data files.

#### `UnifiedDataset` (`utils/dataset.py`)

Handles all dataset interactions.

* **File Discovery**: Scans data directories with parsers (e.g., `utils/parsers/netcdf.py`).
* **Variable Loading**: Identifies and loads required variables using `variables_for_metrics.json` + `aliases.json`.
* **Sample Generation**: Creates list of samples (unique base+lead times).
* **Data Provision**: `__getitem__` loads a single sample, applies slicing, and returns PyTorch tensors.
* **Distributed Support**: Integrated with `DistributedSampler` to split sample lists among workers.

#### `MetricHandler` (`metrics/metric_handler.py`)

Dispatcher for metric calculations.

* Initializes only requested metric modules based on config.
* Runs metrics per sample and collects outputs.
* Supports metrics with `evaluate()` for global aggregation at the end (e.g., correlation maps).

#### Metric Modules (`metrics/*.py`)

Each module implements a specific diagnostic (`torch.nn.Module`).

* **Structure**: Initialized with grid info; `forward()` performs calculation and returns results.
* **Output Keys**: Each module defines names for its outputs.
* **Accumulation**: Optional `evaluate()` for post-hoc reductions.

#### `MPIZarrSaver` (`utils/output_logger.py`)

Enables efficient parallel I/O.

* **Initialization**: Rank 0 creates Zarr file and pre-allocates arrays for all outputs.
* **Synchronization**: Uses `zarr.ProcessSynchronizer` for concurrency safety.
* **Distributed Writing**: Each worker writes directly into its assigned slice.

## 5. Running & Testing

* **Quick Test**: Set `"sample_percent": 1` in config to run on \~1% of random timesteps.
* **Full Run**: Set `"sample_percent": 100` to evaluate all timesteps (large computational cost).
* **Sampling**: Total samples = (# base\_times × # lead\_times).
* **Random Seed**: Set for reproducibility when using random sampling.

## 6. Extending PACE

### Adding a New Metric

1. Create new file in `metrics/` (e.g., `metrics/my_metric.py`).
2. Implement:

   * `forward()` → returns outputs.
   * `output_keys` → defines output variable names.
3. Register metric in config.
4. Add required variables in `variables_for_metrics.json`.
5. Register in `metric_handler.py`.

### Adding Plots

* Extend `utils/plot_utils/` with a new function.
* Add call in `postprocess.py` and in the `config` under `visualization` section.

### Adding Dataset Support

* Update or subclass `UnifiedDataset`.
* Add new parser if file structure differs (e.g., not NetCDF).

## 7. Documentation Sources

* **GitHub README** – Installation, overview, minimal usage.
* **Config Documentation** – Details on configuring datasets, metrics, and sampling.
* **Metric Documentation** – Explanations of each diagnostic.
* **General User/Developer Guide (This file)** – Workflow, running/testing instructions, and extension guidelines.
* **Inline Docstrings** – For parameter/function explanations in code.

## 8. Notes & Best Practices

* Keep configs under version control for reproducibility.
* Use `datetime` for `base_time` and `valid_time`; use `timedelta` for `lead_time`.
* GPU acceleration is optional; CPU workflows are sufficient for sampled evaluation.
* Use Zarr outputs for scalable postprocessing.
