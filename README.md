# 🌍 Physics-Aware Consistency Evaluator (PACE)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-in%20progress-lightgrey)](https://pace4earth.github.io/toolkit/)

**PACE toolkit** provides a set of diagnostics to evaluate the **physical consistency** of machine learning-based Earth system predictions. It helps verify whether models like **GraphCast** (global) and **CorrDiff** (regional downscaling) respect fundamental physical laws across space, time, and variables.

---

## 💡 Why PACE?

Machine learning forecasts can appear statistically accurate while violating basic physical relationships (e.g., pressure-wind imbalance, unphysical temperature fields). **PACE** provides a configurable framework to:

* Assess **multivariate physical consistency**
* Evaluate **spatial/temporal coherence**
* Compare ML outputs against physically grounded baselines
* Customize evaluations via a single JSON configuration file

---

## 🔍 What It Does

**Physical balance checks**: 
* ✅ **geostrophic balance**
* ✅ **hydrostatic balance**
* ✅ **humidity–temperature consistency**
* ✅ **potential vorticity**

**Spatial/temporal metrics**:
* ✅ **correlation**
* ✅ **correlation maps**

**Flexible configuration** via `config.json`

---

## ⚙️ Configuration System

PACE uses a single JSON configuration file to control all aspects of evaluation and visualization.

Example:

```json
{
  "datasets": {
    "model": { "name": "graphcast", "path": "${GRAPHCAST_DATA_PATH}" },
    "reference": { "name": "era5", "path": "${ERA5_DATA_PATH}" }
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
  "spatial": { "pressure_levels": 3, "lat_range": [30, 50], "lon_range": [0, 30] },
  "time": {
    "start": "20210401", "end": "20210412",
    "num_lead_times": 40, "stride_hours": 6, "sample_percent": 20,
    "custom_times": { "enabled": true, "times": ["20210405_12", "20210406_03"] }
  },
  "visualization": {
    "plots_dir": null,
    "histogram": false,
    "vertical_profile": false,
    "summary_stats": ["mean", "stdev", "min", "max"],
    "spatial_slice": {
      "enabled": true, "variable": "geostrophic_wind_ratio", "level": 500, "samples": 3
    }
  }
}
```

See full configuration documentation in [`config.md`](https://github.com/PACE4Earth/PACE-Toolkit/blob/main/docs/config/configuration.md).

---

## 📦 Installation 

```bash
git clone https://github.com/PACE4Earth/PACE-Toolkit.git
cd PACE-Toolkit
pip install -r requirements.txt
```

---

## 🚀 Running an Evaluation

```bash
python pace/evaluator.py
```

Parallel computing supported!

Output: `.zarr` files for each dataset specified in the config

---

## 🚀 Running Visualization

```bash
python pace/postprocess.py
```

### Currently supported plots:
* ✅ **histograms**
* ✅ **vertical profiles**
* ✅ **spatial slices**

Output: figures per metric - model leadtime comparison, as well as comparison model vs physical reference

---

## 🗂️ Project Structure

```
PACE-Toolkit/
├── docs/               
│   ├── metrics/         # Physical consistency metrics
│   ├── config.md        # Configuration documentation
├── pace/                
│   ├── configs/
|   |   ├── config.json  # Main configuration file
│   ├── evaluator.py     # Core evaluation code
│   ├── metrics/         # Physical consistency metrics
│   ├── utils/
│   |   ├── dataset.py   # Dataloader
│   |   ├── plot_utils/     
│   ├── postprocess.py   # Postprocessing & visualization    
│   ├── plots            # Some output figures     
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🧠 Maintainers

Built as part of the **Code for Earth 2025** Challenge.
Maintained by: **PACE Team**
Contact: [marek.rodny@iblsoft.com](mailto:marek.rodny@iblsoft.com)
