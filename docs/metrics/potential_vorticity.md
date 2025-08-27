# Potential Vorticity Metric

### 1. Objective

This module evaluates **Ertel's potential vorticity (PV)** in atmospheric fields, assessing the physical consistency of ML model outputs (GraphCast) with theoretical expectations and ERA5 reanalysis.  
PV consistency is a strong indicator of **dynamic and thermodynamic coherence**, highlighting imbalances in jet streams, cyclones, tropopause folds, and frontal zones.

---

### 2. Theoretical Background

Ertel's PV in pressure coordinates is:

$$
PV = -g \, (\zeta + f) \, \frac{\partial \theta}{\partial p}
$$

where:  

- $g$ — gravitational acceleration [m/s²]  
- $\zeta$ — relative vorticity [1/s] computed from horizontal winds  
- $f$ — Coriolis parameter [1/s]  
- $\theta$ — potential temperature [K]  

Potential temperature is computed as:

$$
\theta = T \left(\frac{p_0}{p}\right)^{R_d/c_p}, \quad R_d/c_p \approx 0.286
$$

with $p_0 = 1000$ hPa.  

Relative vorticity is calculated from wind components:

$$
\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
$$

PV is **conserved in adiabatic, frictionless flow**, so deviations indicate inconsistencies between wind, temperature, and pressure fields.

---

### 3. Input Data

| Variable           | Name in sample dict           | Dimensions    |
|--------------------|------------------------------|---------------|
| Temperature        | `"temperature"`              | [B, L, H, W] |
| U wind component   | `"u_component_of_wind"`      | [B, L, H, W] |
| V wind component   | `"v_component_of_wind"`      | [B, L, H, W] |
| Geopotential       | `"geopotential"`             | [B, L, H, W] |

**Grid metadata (separately supplied):**

- `dx, dy` — grid spacing [m]  
- `f` — Coriolis parameter [1/s]  
- `pressure_levels` — [Pa]  
- Latitude / longitude arrays  

**Domain and resolution requirements:**

- Horizontal: ≤50 km (~GraphCast ~25 km)  
- Vertical: ≤50 hPa spacing (especially 300–100 hPa)  
- Lat/Lon: global recommended  

---

### 4. Workflow in Code

1. **Potential Temperature**  
   - Compute $\theta = T (p_0/p)^{R_d/c_p}$  
   - Top-to-bottom ordering enforced for pressure levels.

2. **Vertical Gradient**  
   - Compute $\partial \theta / \partial p$ using:  
     - Forward difference at top level  
     - Backward difference at bottom level  
     - Central difference in interior levels

3. **Relative Vorticity**  
   - $\zeta = dv/dx - du/dy$  
   - Handles global longitude wraparound if domain is periodic  
   - Forward/backward differences at edges, central inside

4. **Total Vorticity**  
   - $\eta = f + \zeta$  

5. **Compute PV**  
   - $PV = -g \, \eta \, \partial \theta / \partial p$  
   - Converted to PVU: $1 \, \text{PVU} = 10^{-6} \, \text{K m²/(kg s)}$  
   - Top and bottom levels masked as NaN due to unreliable boundaries

---

### 5. Outputs

- **Primary output:**  
  `"potential_vorticity"` → Tensor [B, L, H, W]  
  - PV in PVU  
  - Top/bottom levels set to NaN  

---

### 6. Interpretation

- **PV structure consistent with dynamics:** fields show smooth jet streams, tropopause folds, frontal zones  
- **Deviations from reference dataset or extreme values:** may indicate numerical instability or physical imbalance 

---
