# PACE Project Configuration Documentation

This document describes the configuration file format for the **PACE** project's evaluation and visualization pipeline. It covers all available options, their meanings, default values, and how they interact with the `UnifiedDataset` data loader.

## Overview

The configuration file is a JSON object that specifies:

* Input datasets (model and reference)
* Output directories
* Metrics to compute
* Spatial and temporal subset parameters
* Visualization options

Example configuration:

```json
{
  "distributed": false,
  "datasets": {
    "model": {
      "name": "graphcast",
      "path": "${GRAPHCAST_DATA_PATH}"
    },
    "reference": {
      "name": "era5",
      "path": "${ERA5_DATA_PATH}"
    }
  },
  "outputs_dir": "${OUTPUTS_DIR_PATH}",
  "metrics": {
    "geostrophic_balance": ["geostrophic_wind_ratio"],
    "hydrostatic_balance": ["hydrostatic_rel_error", "hydrostatic_rmse"],
    "correlation": ["all"],
    "correlation_map": ["all"],
    "potential_vorticity": ["all"],
    "humidity_temperature": ["all"]
  },
  "spatial": {
    "pressure_levels": 3,
    "lat_range": [30, 50],
    "lon_range": [0, 30]
  },
  "time": {
    "start": "20210401",
    "end": "20210412",
    "num_lead_times": 40,
    "stride_hours": 6,
    "sample_percent": 20,
    "custom_times": {
      "enabled": true,
      "times": ["20210405_12", "20210406_03", "20210103_12"]
    }
  },
  "visualization": {
    "plots_dir": null,
    "histogram": false,
    "vertical_profile": false,
    "summary_stats": ["mean", "stdev", "min", "max"],
    "spatial_slice": {
      "enabled": true,
      "variable": "geostrophic_wind_ratio",
      "level": 500,
      "samples": 3
    }
  }
}
```

## Sections

### 0. `distributed`

Specifies whether the evaluation should run in distributed (parallel) mode (e.g., on an HPC system with SLURM).

| Key           | Type  | Description                                                                                   | Default |
| ------------- | ----- | --------------------------------------------------------------------------------------------- | ------- |
| `distributed` | bool  | If `true`, the evaluation uses distributed sampling and computation across multiple workers. | false   |

---
### 1. `datasets`

Specifies the **model** and **reference** datasets to be compared.

| Key    | Type   | Description                                                                  | Default      |
| ------ | ------ | ---------------------------------------------------------------------------- | ------------ |
| `name` | string | Human-readable name of the dataset (e.g., `graphcast`, `era5`).              | *(required)* |
| `path` | string | Path to the dataset files. Environment variables are expanded automatically. | *(optional)* |
---
Note: Metrics can be computed for one dataset only (just remove reference from the config).

### 2. `outputs_dir`

Directory where metric outputs (.zarr files) are stored.

* Type: `string`
* Default: *(.../pace/outputs/)*
* Environment variables are expanded.

---

### 3. `metrics`

Specifies which metrics to compute and which outputs to request for each metric.

* Keys correspond to metric module names (e.g., `geostrophic_balance`).
* Values are arrays of output names to compute (or `"all"` to compute all available outputs).

Default: *(required — must specify at least one metric)*

### List of currently supported metrics and their outputs keys:

```json
"metrics": {
    "geostrophic_balance": ["geostrophic_wind_ratio"],
    "hydrostatic_balance": ["hydrostatic_abs_error", "hydrostatic_rel_error", "hydrostatic_rmse"],
    "correlation": ["all"],
    "correlation_map": ["all"],
    "potential_vorticity": ["potential_vorticity"],
    "humidity_temperature": ["relative_humidity"]
  },
```

---

### 4. `spatial`

Defines the spatial subset of data to be loaded.

| Key               | Type                 | Description                                                                                                                               | Default             |
| ----------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `pressure_levels` | int \| list \| "all" | If integer: number of pressure levels from the top of the file. If list: exact pressure levels to select. If `"all"`: include all levels. | `"all"` |
| `lat_range`       | list \| "all"        | Latitude range `[min, max]` to select (expected from range [-90, 90]), or `"all"` for all available latitudes.                                                            | `"all"`             |
| `lon_range`       | list \| "all"        | Longitude range `[min, max]` to select (expected from range [0, 360]), or `"all"` for all available longitudes.                                                          | `"all"`             |

---

### 5. `time`

Defines the temporal subset of data to be used.

| Key                    | Type   | Description                                                       | Default             |
| ---------------------- | ------ | ----------------------------------------------------------------- | ------------------- |
| `start`                | string | Start date in `YYYYMMDD` format.                                  | *(required)*        |
| `end`                  | string | End date in `YYYYMMDD` format.                                    | *(required)*        |
| `num_lead_times`       | int    | Number of lead times to include from each base time.              | `1`                 |
| `stride_hours`         | int    | Time step between lead times in hours.                            | `6`                 |
| `sample_percent`       | float  | Percentage of valid times to sample randomly (select 100 to process all samples in specified time range).                     | `1`                 |
| `custom_times`         | object | Allows selecting specific valid times instead of random sampling. | disabled by default |
| `custom_times.enabled` | bool   | If `true`, `custom_times.times` list is used.                     | `false`             |
| `custom_times.times`   | list   | Specific times to include, in `YYYYMMDD_HH` format.               | `[]`                |

---

Note: `valid_times` are selected from time range `start : end - num_lead_times * stride_hours`, so that each `lead_time` has equal number of samples (in total).

Note: When in `custom_times` mode, adjust number of leadtimes per valid time with `num_lead_times` and `sample_percent`.

### 6. `visualization`

Configures postprocessing and plotting.

| Key                                | Type         | Description                                                            | Default                 |
| ---------------------------------- | ------------ | ---------------------------------------------------------------------- | ----------------------- |
| `plots_dir`                        | string\|null | Directory for plots. Environment variable supported.                 | `.../pace/plots/`                  |
| `histogram`                        | bool         | Whether to generate histograms.                                        | `false`                 |
| `vertical_profile`                 | bool         | Whether to generate vertical profiles.                                 | `false`                 |
| `summary_stats`                    | list         | List of summary statistics to compute for vertical profiles (`mean`, `stdev`, `min`, `max`). | `["mean"]`                    |
| `spatial_slice`                    | object       | Controls spatial slice plotting.                                       | disabled by default     |
| `spatial_slice.enabled`            | bool         | Whether to enable spatial slice plots.                                 | `false`                 |
| `spatial_slice.variable`           | string       | Variable to plot.                                                      | *(required if enabled)* |
| `spatial_slice.level`              | int          | Pressure level (hPa) for slice.                                        | *(required if enabled)* |
| `spatial_slice.samples`            | int          | Number of random samples to plot.                                      | *(required if enabled)* |

---

## Data Loader Interaction (`UnifiedDataset`)

The configuration is directly consumed by the `UnifiedDataset` class:

* **Spatial selection**: `lat_range`, `lon_range`, and `pressure_levels` are applied to each NetCDF file.
* **Temporal selection**: Files are filtered by `start`, `end`, and `num_lead_times`. If `custom_times.enabled` is `true`, times are matched to the closest available valid times.
* **Metric requirements**: For each metric in the `metrics` section, the loader checks if all required fields are present in the dataset (`variables_for_metrics.json` + `aliases.json`).
* **Sample selection**: Random sampling is applied if `custom_times` is not enabled, controlled by `sample_percent`.
* **Environment variables**: All paths in the config can use `${VAR}` syntax and are expanded at runtime.

### Expected NetCDF structure

* **Dimensions:** should include `time`, and optionally `level` or `pressure`. Only one `base_time` pre file is allowed.
* **Latitude/longitude variables:** either `latitude`/`longitude` or `lat`/`lon`.
* **Time handling:**

  * Preferably includes `base_time` (datetime) and `lead_time` (hours or timedelta).
  * If absent, `time` variable should be absolute datetimes or relative lead times.
* **Variable names:** may differ between datasets; resolved using `aliases.json` mapping.

### Returned sample format

The loader returns samples as dictionaries:

```python
{
  "<variable_name>": torch.Tensor(...),
  "base_time": datetime,
  "lead_time": timedelta
}
```

---
