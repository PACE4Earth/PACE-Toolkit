# Mass Conservation Metric

### 1. Objective

This module evaluates **mass consistency** in ML weather model outputs on a pressure-based vertical grid.  
It reconstructs approximate surface pressure from mean sea-level pressure (MSLP) using temperature and geopotential at the lowest model level, then computes the **total atmospheric mass**.  

The metric provides a single scalar per sample that can be used to monitor whether a model conserves mass at the global or regional scale.

---

### 2. Theoretical Background

Atmospheric mass is related to **surface pressure** through the hydrostatic relation:

$$
p_s = p_{msl} \exp\left(\frac{\Phi_s}{R T_s}\right)
$$

where:  
- $p_s$ = reconstructed surface pressure [Pa]  
- $p_{msl}$ = mean sea-level pressure [Pa]  
- $\Phi_s$ = surface geopotential [m²/s²]  
- $T_s$ = surface temperature [K]  
- $R = 287.05$ J/(kg·K), gas constant for dry air  

The **total atmospheric mass** is then obtained as an area-weighted sum over the grid:

$$
M = \frac{1}{g} \sum_{i,j} p_s(i,j) \, \Delta A(i,j)
$$

where:  
- $g = 9.80665$ m/s², gravitational acceleration  
- $\Delta A(i,j)$ = grid cell area [m²]  

This approach assumes the hydrostatic approximation and uniform composition (dry air) to reconstruct the mass from MSLP.

---

### 3. Input Data

The metric requires model output fields on a latitude–longitude pressure grid:

| Variable                     | Name in sample dict           | Dimensions   |
|-------------------------------|-------------------------------|--------------|
| Mean sea-level pressure       | `"mean_sea_level_pressure"`  | [B, H, W]   |
| Temperature                   | `"temperature"`              | [B, L, H, W]|
| Geopotential                  | `"geopotential"`             | [B, L, H, W]|

Additional grid metadata is supplied separately:  
- `dx, dy` → horizontal grid spacing in meters [H, W]  
- `pressure_levels` → array of model pressure levels [Pa]  

The **lowest model level** (highest pressure) is automatically identified for reconstruction.

---

### 4. Workflow in Code

1. **Identify lowest model level**  
   - Index of highest pressure in `pressure_levels` is used to select surface temperature and geopotential.

2. **Reconstruct surface pressure**  
   - Surface geopotential $\Phi_s$ and temperature $T_s$ at the lowest level are combined with MSLP using:  
     ```python
     ps = mslp * torch.exp(phi_s / (R * T_s))
     ```

3. **Compute total mass**  
   - Area-weighted sum over all grid points:  
     ```python
     total_mass = (ps * area).sum(dim=(-2,-1)) / g
     ```
   - `area` is precomputed from `dx` and `dy`.

4. **Batch support**  
   - Computation is vectorized over batch dimension [B], returning one scalar mass per sample.

---

### 5. Outputs

- **Primary output:**  
  `"total_mass"` → Tensor [B]  
  Total atmospheric mass in kilograms per batch sample.

---

### 6. Interpretation

- **Consistent mass:** Metric should remain nearly constant for mass-conserving models.  
- **Deviations:** Large deviations indicate possible mass imbalance or numerical errors.  
- **Utility:** Can be used for model validation, physics-based loss functions, or evaluation of ML-generated atmospheric fields.
