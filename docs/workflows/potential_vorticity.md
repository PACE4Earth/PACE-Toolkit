### **1\. Objective**

This workflow aims to evaluate the physical consistency of GraphCast ML model outputs by calculating Ertel’s potential vorticity (PV) and comparing it against theoretical expectations and reanalysis data (e.g., ERA5). PV consistency is a strong indicator of dynamic and thermodynamic coherence in model fields and helps diagnose imbalances, especially in dynamically active regions such as jet streams, cyclones, and fronts.

The method will later be extended to other ML models (e.g., CorrDiff), but GraphCast serves as the initial implementation baseline.

###

### **2\. Theoretical Background**

Ertel's Potential Vorticity (PV) in pressure coordinates is given by:

$$
P V=-g\left(\zeta_\theta+f\right) \cdot \nabla_p \theta
$$

Where:
- $g$ : gravitational acceleration
- $\boldsymbol{\zeta}_{\boldsymbol{\theta}}$ : relative vorticity on isentropic surfaces
- $f$ : Coriolis parameter
- $\theta$ : potential temperature
- $\nabla_p \theta$ : gradient of potential temperature in pressure coordinates


###

Under adiabatic and frictionless conditions, PV is conserved. Therefore, significant deviations in PV fields may indicate inconsistencies between wind, temperature, and pressure fields.

### **3\. Input data**

**GraphCast Outputs Required (as Input):**

| **Variable Type** | **Variable Name** | **Pressure Levels (hPa)** |
| --- | --- | --- |
| Temperature | t   | 100 to 1000 (preferably ≤50 hPa spacing) |
| U wind component | u   | 100 to 1000 |
| V wind component | v   | 100 to 1000 |
| Geopotential | z   | 100 to 1000 |
| Latitude, Longitude | lat, lon | —   |

Lat, Lon range: Global (0° to 360°)

**Spatial and Vertical Resolution Requirements:**

- Horizontal resolution: ≤50 km recommended (GraphCast provides ~25 km) to resolve PV anomalies like tropopause folds and streamers.
- Vertical resolution: Fine spacing (≤50 hPa) is required, especially near upper troposphere/lower stratosphere (~300–100 hPa) where sharp gradients occur.

### **4\. Workflow Steps**

#### **Step 1: Compute Potential Temperature**

$\theta = T\left(\dfrac{1000}{p}\right)^\kappa,\qquad \kappa=\dfrac{R_d}{c_p} \approx 0.286$

#### **Step 2: Compute PV**

Calculate:

- Relative vorticity from wind fields

    $\zeta=\dfrac{\partial v}{\partial x} - \dfrac{\partial u}{\partial y}$

- Gradient of θ in pressure coordinates

Use the Ertel PV equation (in pressure coordinates):

$PV=-g(\zeta + f) \cdot \nabla_p \theta$

Tools: metpy.calc.relative_vorticity, xarray.gradient

#### **Step 3: Compare to ERA5 PV Baseline**

Compute PV using ERA5 reanalysis fields (same method)

Calculate:

- RMSE between GraphCast and ERA5 PV
- Skill Score 

    $\mathrm{Skill}_{\mathrm{bounded}}=\frac{R M S E_{\mathrm{ERA} 5}-R M S E_{\mathrm{GraphCast}}}{R M S E_{\mathrm{ERA} 5}+R M S E_{\mathrm{GraphCast}}}$

    +1 = perfect prediction (zero error)   
    0 = same as ERA5  
    –1 = very poor (RMSE much higher than baseline)

#### **Step 4: Flag Physically Unrealistic Values and Outliers**

- Identify and flag PV values outside expected ranges using

- Empirical distributions from PV climatology (eg. from ERA)

- Level-dependent bounds (e.g., PV > 5 PVU at 900 hPa is unlikely)

- Latitude-based expectations (PV generally increases toward the poles)

#### **Step 5: Visualization and Diagnostics**

Global Scale

- Zonal mean cross-sections (latitude vs. pressure) of PV
- Maps of PV at specific levels (e.g., 300, 500, 850 hPa)
- Difference maps: GraphCast PV – ERA5 PV
- PV anomaly plots

Local Scale

- Identify sharp PV gradients (e.g., along tropopause)
- Check for realistic structure of jet streams (PV streaks, frontal zones)
- Use scatter plots of GraphCast vs. ERA5 PV (expect 1:1 alignment)

### **5\. Results (Planned Output)**

- Skill score and RMSE maps for PV at different pressure levels
- Detection of PV anomalies (in space and time) relative to ERA5 and climatology
- Visual comparison of PV structures (maps, sections, scatter plots)

### **6\. Future Extensions and Ideas**

- Apply to other ML models (e.g., CorrDiff)
- Evaluate across all pressure levels (100–1000 hPa)
- Track PV conservation over time in forecasts

### **Tools and Libraries**

- **xarray, numpy, dask, pytorch** – data handling and computing
- **metpy** – PV calculations
- **matplotlib** – plotting and mapping
- **cdsapi** – data access (ERA5)
