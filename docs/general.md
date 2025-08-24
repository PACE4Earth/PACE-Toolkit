# PACE-Toolkit Documentation

## 1. Overview

The PACE-Toolkit is a Python-based framework designed for calculating and visualizing physical consistency metrics for weather and climate models. It is optimized for High-Performance Computing (HPC) environments, leveraging MPI and PyTorch for scalable, parallel processing of large datasets.

The core philosophy is to provide a configurable and extensible platform for researchers to evaluate model output against fundamental physical principles like geostrophic balance, hydrostatic equilibrium, and energy conservation.

## 2. General Workflow

The toolkit operates in a two-stage process, typically managed by SLURM batch scripts.

1.  **Evaluation Stage (`evaluator.py`)**: This is the main computational step.
    * Reads weather model data (e.g., NetCDF files).
    * Calculates a suite of physical metrics in parallel using MPI.
    * Saves the results to a scalable Zarr data store.
    * *Launched by scripts like `minimal_run.sbatch` or `run_cluster.sbatch`.*

2.  **Post-processing Stage (`postprocess.py`)**: This is the visualization step.
    * Reads the Zarr data generated during the evaluation stage.
    * Creates various visualizations, such as histograms and vertical profiles of the calculated metrics.
    * *Launched by scripts like `minimal_postprocess.sbatch`.*

### How to Run

A complete job, including both evaluation and post-processing, can be launched with a single command:

```bash
# This script runs both stages sequentially
sbatch minimal_all.sbatch
```

The workflow is controlled by environment variables set within the `.sbatch` script:

  * `DATASET_CONFIG_PATH`: **(Required)** Specifies the path to the main JSON configuration file that defines the entire run.
  * `OUTPUT_DIR_PATH`: **(Required)** Defines the directory where the output Zarr datasets will be saved.
  * `PLOTS_DIR_PATH`: Defines the directory for saving plots generated during post-processing.

## 3\. Directory Structure

The project is organized into the following key directories:

```
pace/
├── configs/          # JSON configuration files for runs, variables, and aliases.
├── metrics/          # Individual modules for each physical metric calculation.
├── utils/            # Helper modules for data loading, I/O, plotting, etc.
├── logs/             # Output and error logs from SLURM jobs.
├── evaluator.py      # Main script for the evaluation stage.
├── postprocess.py    # Main script for the post-processing stage.
└── *.sbatch          # SLURM scripts for submitting jobs to an HPC cluster.
```


## 4\. Core Components

### 4.1. Scripts

#### `evaluator.py`

This is the main driver for metric computation. It orchestrates the entire parallel processing pipeline.

  * **Initialization**: Sets up the MPI and PyTorch Distributed environments for parallel execution.
  * **Configuration**: Reads a central JSON configuration file to determine datasets, metrics, and the time/space domain.
  * **Workload Distribution**: The main process (rank 0) scans for data and builds a list of all samples (a unique combination of a base time and a lead time). This workload is then split and distributed evenly among all MPI processes.
  * **Metric Calculation**: Uses the `MetricHandler` to run the configured metric modules on its assigned data samples.
  * **Parallel I/O**: Employs the `MPIZarrSaver` to handle concurrent writes from all processes into a single, shared Zarr file, avoiding I/O bottlenecks.

#### `postprocess.py`

This script generates visualizations from the results of the evaluation stage.

  * **Data Loading**: Opens the Zarr datasets produced by `evaluator.py` using `xarray`.
  * **Plot Generation**: Reads the `visualization` section of the config file and calls the appropriate functions from `utils/plot_utils/` to generate and save figures.

#### `minimal_inspect.py`

A utility script for developers to perform ad-hoc analysis and visualization of the output Zarr files. It serves as a practical example of how to interact with the toolkit's output data using standard libraries like `xarray` and `matplotlib`.

### 4.2. Key Objects and Classes

#### Configuration Files (`configs/`)

The toolkit's behavior is defined by human-readable JSON files.

  * **Main Config (`config_*.json`)**: The central control file for a run.
      * `datasets`: Defines the `model` and an optional `reference` dataset, including their name and path on the filesystem.
      * `metrics`: A dictionary specifying which metric suites to run (e.g., `"geostrophic_balance"`). The value is a list of specific output fields to save, or simply `["all"]` to save everything the metric produces.
      * `time` & `spatial`: Objects that define the temporal and spatial domains for the analysis (e.g., date ranges, latitude/longitude bounds, pressure levels).
      * `visualization`: A dictionary with boolean flags to enable or disable different plot types in the `postprocess.py` script.
  * **`variables_for_metrics.json`**: This file defines the data dependency for each metric. It maps a metric suite (e.g., `"hydrostatic_balance"`) to the list of input physical variables it requires from the source data (e.g., `["geopotential", "temperature"]`).
  * **`aliases.json`**: This file provides flexibility in data loading by mapping the toolkit's canonical variable names (e.g., `"geopotential"`) to potential alternative names found in different data files (e.g., `"z"`, `"gh"`).

#### `UnifiedDataset` (`utils/dataset.py`)

This class is the cornerstone of the data loading pipeline. It abstracts away the complexity of finding, parsing, and preparing data for computation.

  * **File Discovery**: Scans data directories using parsers (e.g., `utils/parsers/netcdf.py`) to find relevant files within the configured time range.
  * **Variable Loading**: Uses `variables_for_metrics.json` and `aliases.json` to identify and load only the variables required for the requested metrics.
  * **Sample Generation**: Creates a list of all possible "samples" (a unique file, base time, and lead time).
  * **Data Provision**: The `__getitem__` method loads a single sample, applies spatial slicing, and returns a dictionary of PyTorch tensors ready for computation.

#### `MetricHandler` (`metrics/metric_handler.py`)

This class acts as a factory and dispatcher for all metric calculations.

  * It reads the `metrics` section of the configuration and initializes only the requested metric modules.
  * Its `forward` method takes a data sample and passes it to each initialized module, collecting the results into a single output dictionary.

#### Metric Modules (`metrics/*.py`)

Each file in this directory implements a specific physical calculation as a `torch.nn.Module`.

  * **Structure**: Modules are initialized with static grid information (e.g., grid spacing). The `forward` method accepts a dictionary of data tensors, performs its calculation, and returns the resulting metric(s).
  * **Accumulation**: For metrics that require statistics over the entire time series (e.g., `CorrelationMap`), an `evaluate()` method can be implemented. This method is called once at the end of the evaluation stage to perform a final aggregation (like an MPI reduction) and save the final result.

#### `MPIZarrSaver` (`utils/output_logger.py`)

This class enables efficient and safe parallel I/O to a single Zarr data store.

  * **Initialization**: The main process (rank 0) creates the Zarr file and pre-allocates empty arrays for all output variables, sized to hold the results from all samples across all processes.
  * **Synchronization**: Uses `zarr.ProcessSynchronizer` to manage concurrent file access and prevent race conditions.
  * **Distributed Writing**: Each process writes its results directly into its assigned slice of the pre-allocated Zarr arrays, identified by a global sample index. This "write-in-place" strategy is highly scalable as it avoids gathering all data on a single node.

<!-- end list -->

```
```