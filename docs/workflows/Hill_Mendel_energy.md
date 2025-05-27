# Hill-Mandel Energy Consistency Evaluation

## 1. Objective

This workflow aims to evaluate the **energetic consistency** of machine learning model outputs (e.g., GraphCast) using the **Hill-Mandel principle**. This principle ensures that the energy calculated at the macro-scale matches the energy dissipated or stored at the micro-scale under homogenization assumptions.

It is commonly applied in continuum mechanics and is here adapted to atmospheric energy diagnostics to verify how consistent the energetics are across scales, particularly in coarse-grained (ML) atmospheric fields.

## 2. Theoretical Background

The Hill-Mandel condition in mechanics asserts that the macroscopic (averaged) stress-power equals the volume average of the microscopic stress-power:

$$
\langle oldsymbol{\sigma} : \dot{oldsymbol{arepsilon}} 
angle = oldsymbol{\Sigma} : \dot{oldsymbol{E}}
$$

**Adapted for atmospheric energy analysis**, this translates to checking if the total energy budget derived from ML-predicted macro-scale variables (e.g., wind, temperature, pressure) matches the expected aggregate energy changes (kinetic + internal + latent + potential).

We compute both sides of the equation:

- **Macro energy evolution** (from ML-predicted mean fields)
- **Micro energy terms** (from resolved gradients and fields within a grid box)

Inconsistencies may signal energetic leakage due to poor scale separation or ML model imbalances.

## 3. Input Data

| **Variable Type**      | **Variable Name** | **Pressure Levels (hPa)** |
|------------------------|-------------------|----------------------------|
| Temperature            | t                 | 100–1000                   |
| Specific humidity      | q                 | 100–1000                   |
| Geopotential           | z                 | 100–1000                   |
| U wind component       | u                 | 100–1000                   |
| V wind component       | v                 | 100–1000                   |
| Pressure               | p                 | 100–1000                   |
| Surface pressure       | ps                | —                          |
| Latitude, Longitude    | lat, lon          | —                          |

## 4. Workflow Steps

### Step 1: Calculate Energy Components

- **Kinetic Energy (KE)**:  
  $$ KE = \frac{1}{2} (u^2 + v^2) $$

- **Internal Energy (IE)**:  
  $$ IE = c_v T $$

- **Latent Heat Energy (LE)**:  
  $$ LE = L_v q $$

- **Potential Energy (PE)**:  
  $$ PE = g z $$

Total macro-scale energy:
$$
E_{macro} = KE + IE + LE + PE
$$

### Step 2: Micro-scale Consistency Check

Compute divergence of energy fluxes within control volumes or grid boxes and compare with expected energy tendency (using finite differences or advection tendencies).

Check:
$$
\langle \nabla \cdot (\text{Energy flux}) \rangle \overset{?}{=} \frac{\partial E_{macro}}{\partial t}
$$

### Step 3: Visualization

- Plot time evolution of domain-averaged energy terms.
- Visualize spatial distribution of imbalance (residuals between LHS and RHS).
- Histogram of residuals.

## 5. Output and Diagnostics

- Time series of total energy and its components.
- Residual map: |macro energy rate – micro flux divergence|
- Percentage of grid points exceeding acceptable imbalance thresholds.
- Optional: Compare with ERA5 or physically-based reanalysis.

## 6. Future Extensions

- Apply to 3D grid-resolving ML outputs (e.g., CorrDiff).
- Extend to diabatic processes and include radiative and surface fluxes.
- Use as loss function component during ML model training.

## 7. Tools and Libraries

- `xarray`, `numpy`, `dask` – data handling and computation
- `matplotlib`, `cartopy` – visualization and plotting
- `pytorch` – GPU acceleration and ML integration
