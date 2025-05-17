# 🌍 Physics-Aware Consistency Evaluator (PACE)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg)](https://pytorch.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Docs](https://img.shields.io/badge/docs-in%20progress-lightgrey)](https://pace4earth.github.io/toolkit/)

**PACE toolkit** provides a set of diagnostics to evaluate the **physical consistency** of machine learning-based Earth system predictions. It helps verify whether models like **GraphCast** (global) and **CorrDiff** (regional downscaling) respect fundamental physical laws across space, time, and variables.

---

## 💡 Why PACE?

Machine learning forecasts can appear statistically accurate while violating basic physical relationships (e.g., pressure-wind imbalance, unphysical temperature fields). **PACE** provides a framework to:

- Assess **multivariate physical consistency**
- Evaluate **spatial/temporal coherence**
- Compare ML outputs against physically grounded baselines

---

## 🔍 What It Does

- ✅ Spectral and power-law analysis  
- ✅ Spatial correlation length metrics  
- ✅ Cross-variable consistency checks  
- ✅ CRPS, Energy Score, and probabilistic diagnostics  
- ✅ Scattering coefficient analysis (Brochet et al. 2023)  
- ✅ Case-study support (e.g., typhoons, fronts)

---

## 📦 Installation (coming soon)

```bash
git clone https://github.com/PACE4Earth/PACE-Toolkit.git
cd PACE-Toolkit
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
PACE-Toolkit/
├── pace/                # Core diagnostics and evaluation code
│   ├── evaluator.py
│   ├── metrics/
│   └── utils/
├── notebooks/           # Jupyter demos and test cases
├── tests/               # Unit tests
├── data/                # Reference data / links
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🧠 Maintainers

Built as part of the **Code for Earth 2025** Challenge.  
Maintained by: **PACE Team**  
Contact: marek.rodny@iblsot.com
