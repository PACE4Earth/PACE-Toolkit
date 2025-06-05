# Definition of Physical Consistency Evaluation Metrics for ML Model Outputs

This document outlines the final set of diagnostics and methods used to evaluate the physical consistency of weather forecasts produced by machine learning (ML) models. These metrics aim to assess whether the outputs from ML models, such as GraphCast or CorrDiff, adhere to fundamental physical relationships governing the atmosphere. The framework is intended to be applicable across spatial scales, with some metrics computed at the gridpoint level and others involving broader global evaluations. The overall goal is to provide a coherent framework for evaluating multi-variable consistency, conservation principles, and physical realism.

This document provides a consolidated reference for the milestone. Individual metric workflows (detailed with objective, theory, inputs, computation, and output) are maintained under *docs/workflows/*.

## 1. Overview of Metrics

| Metric                     | Description                                               | Main Variables                      | Scale               |
| -------------------------- | --------------------------------------------------------- | ----------------------------------- | ---------------     |
| Geostrophic Balance        | Balance between Coriolis and pressure-gradient forces     | wind, geopotential                  | Midlatitudes/Global |
| Hydrostatic Balance        | Vertical pressure-gradient force vs. gravity              | pressure, geopotential, temperature | Vertical Column     |
| Hill–Mandel Energy Balance | Energy consistency between resolved and subgrid scales      | wind, temperature, pressure         | Gridpoint           |
| Energy Conservation        | Conservation of total energy  (internal, kinetic, and latent)               | temperature, geopotential, wind     | Global              |
| Humidity–Temperature Check | Physical bounds between temperature and specific humidity | temperature, humidity               | Gridpoint           |
| Mass Conservation          | Change in air mass vs. mass flux divergence (Continuity equation)          | wind, pressure                      | Global/Regional     |
| Potential Vorticity (PV)   | Dynamic-thermodynamic consistency using Ertel PV          | wind, temperature, pressure         | Synoptic/Global     |
| Spectral Power Analysis    | Distribution of energy across spatial scales              | wind, geopotential                  | Synoptic/Global              |

---
  
## 2. Metric Definitions

### Geostrophic Balance

**Objective:**
Evaluate the physical consistency of ML outputs by comparing actual wind fields to the geostrophic wind derived from pressure gradients. This assesses the balance between the Coriolis and pressure-gradient forces.

**Theoretical Background:**
Valid in large-scale, slowly varying flows. Geostrophic wind in pressure coordinates:

$$
u_g = -\frac{1}{f} \frac{\partial \Phi}{\partial y}, \quad v_g = \frac{1}{f} \frac{\partial \Phi}{\partial x}
$$

where $\Phi = gz$ is the geopotential and $f$ is the Coriolis parameter.

**Input Data (GraphCast):**

* Geopotential (z), U and V wind components at 850 hPa (later for other pressure levels)
* Latitude and longitude (global, but evaluated mainly 30°–80° latitudes)

**Computation Method:**

1. Compute horizontal gradients of geopotential (with `metpy.calc.geospatial_gradient`)
2. Calculate geostrophic wind
3. Compare to actual wind using RMSE, relative error, and skill scores
4. Optional: mask dynamically active regions (e.g., jet streams)

**Evaluation Output:**

* RMSE and skill score relative to ERA5
* Spatial maps of imbalance
* Histograms and scatter plots (e.g., $| \vec{v} - \vec{v}_g |$ )

---
### Hydrostatic Balance

**Objective:**
Assess the consistency of vertical pressure–geopotential gradients in ML outputs by testing hydrostatic balance. Deviations indicate physically unrealistic vertical structure.

**Theoretical Background:**
The hydrostatic relation in pressure coordinates:

$$
\Phi(p_2) - \Phi(p_1) \approx -R_d \cdot \bar{T} \cdot \ln\left(\frac{p_2}{p_1}\right)
$$

where $\Phi = gz$, $R_d \approx 287 \, \mathrm{J/(kg \cdot K)}$, and $\bar{T}$ is the mean temperature between levels.

**Input Data (GraphCast):**

* Geopotential (z) and temperature (t) at standard pressure levels (e.g., 850, 700 hPa)
* Latitude and longitude (global)

**Computation Method:**

1. Compute theoretical hydrostatic geopotential difference using mean temperature
2. Subtract actual geopotential difference to get error
3. Evaluate RMSE, relative error, and skill score
4. Compare to ERA5 baseline
5. Flag physically implausible or statistically extreme values

**Evaluation Output:**

* RMSE, relative error, skill score vs. ERA5
* Spatial error maps, histograms, scatter plots
* Percent of grid points violating hydrostatic expectations
* Latitudinal/seasonal diagnostics and outlier detection

---

### Energy Conservation

**Objective:**
Assess whether GraphCast conserves total atmospheric energy (internal, kinetic, latent) over time. This checks the model’s physical consistency in the absence of external sources or sinks.

**Theoretical Background:**
Total energy per unit mass:

$$
E_{\text{total}} = c_{v,\text{moist}} T + \frac{1}{2}(u^2 + v^2) + L_v q_v
$$

where:

* $c_{v,\text{moist}}$: specific heat at constant volume
* $T$: temperature
* $u, v$: horizontal wind components
* $q_v$: specific humidity
* $L_v$: latent heat of vaporization (\~2.5×10⁶ J/kg)

**Input Data (GraphCast):**

* Temperature, u, v, specific humidity at 100–1000 hPa
* Geopotential, surface pressure, lat/lon (global)

**Computation Method:**

1. Compute energy components per gridpoint
2. Vertically integrate total energy using pressure-layer thickness
3. Calculate temporal change $\partial E / \partial t$
4. Analyze energy tendency and compare to ERA5

**Evaluation Output:**

* RMSE and max error of energy tendency
* Time series and zonal means of total energy
* Energy tendency maps highlighting imbalanced regions
* Skill score vs. ERA5

---

### Hill–Mandel Energy Consistency

**Objective:**
Evaluate whether GraphCast respects physical energy exchange between resolved and subgrid scales, using the Hill–Mandel condition.

**Theoretical Background:**
The Hill–Mandel condition tests if subgrid-scale energy transfer is consistent:

$$
\langle {\tau} : \nabla \mathbf{u} \rangle \overset{?}{=} \langle {\tau} \rangle : \langle \nabla \mathbf{u} \rangle
$$

* ${\tau}$: subgrid stress tensor
* $\nabla \mathbf{u}$: velocity gradient tensor
* $\langle \cdot \rangle$: spatial average
* $:$: tensor contraction (double dot product)

**Input Data (GraphCast):**

* Wind components: $u, v, w$ (or compute $w$)
* Subgrid stress tensor ${\tau}$
* Grid spacing $\Delta x, \Delta y, \Delta p$

**Computation Method:**

1. Compute 3D velocity gradient tensor
2. Evaluate pointwise stress power: $P(\mathbf{x}) = {\tau} : \nabla \mathbf{u}$
3. Compute volume average of $P$ and compare to $\langle {\tau} \rangle : \langle \nabla \mathbf{u} \rangle$
4. Quantify discrepancy between both sides of the Hill–Mandel equation

**Evaluation Output:**

* Relative error of Hill–Mandel balance
* Maps of stress power $P(\mathbf{x})$
* Time series of energy consistency metric

---

### Mass Conservation

**Objective:**
Evaluate whether GraphCast preserves mass continuity by comparing local mass tendency with the divergence of horizontal mass flux.

**Theoretical Background:**
Mass continuity (moist air):

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v}) = 0
$$

where
$$\rho = \dfrac{p}{R_d T (1 + 0.61 q)}, \quad \vec{v} = (u, v)$$

**Input Data (GraphCast):**

* Temperature, u, v, specific humidity, pressure at 100–1000 hPa
* Latitude, longitude
* Temporal resolution: ≥3-hourly sufficient

**Computation Method:**

1. Compute moist air density $\rho$
2. Compute mass flux $\rho u, \rho v$
3. Compute divergence $\nabla \cdot (\rho \vec{v})$
4. Calculate local time derivative $\partial \rho / \partial t$
5. Evaluate residual:

$$
\text{Residual} = \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v})
$$

**Evaluation Output:**

* Residual RMSE and max error
* Relative error: residual normalized by $\rho$
* Maps and cross-sections of residuals
* Skill score vs. ERA5

---

### Humidity–Temperature Consistency

**Objective:**
Assess whether GraphCast’s specific humidity output is physically consistent with the Clausius–Clapeyron relation by comparing modeled moisture with saturation expectations.

**Theoretical Background:**
Clausius–Clapeyron scaling sets the temperature dependence of saturation vapor pressure:

$$
\frac{d e_s}{d T} = \frac{L_v e_s}{R_v T^2}, \quad
e_s(T) = e_0 \exp\left(\frac{L_v}{R_v} \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
$$

Expected specific humidity at saturation:

$$
q_s(T, p) = 0.622 \cdot \frac{e_s}{p - e_s}
$$

Check if ML humidity $q_{ML}$ stays within physically reasonable bounds.

**Input Data (GraphCast):**

* Temperature $T$, Specific Humidity $q$
* Pressure levels: 1000 to 500 hPa
* Grid: Lat–Lon domain (global)

**Computation Method:**

1. Compute theoretical $e_s(T)$ and $q_s(T, p)$
2. Calculate model relative humidity: $RH_{ML} = q_{ML} / q_s$
3. Flag nonphysical values (e.g. $RH > 1.2$ or $RH < 0$)
4. Compute error metrics: RMSE, MAE, bias between $q_{ML}$ and $q_s$

**Evaluation Output:**

* Skill scores relative to Clausius–Clapeyron constraint and ERA5 distributions
* Maps of RH anomalies or violations
* Histograms and scatter plots of $q_{ML}$ vs. $q_s$

---

### Potential Vorticity Consistency

**Objective:**
Assess physical consistency of GraphCast wind–temperature–pressure fields by computing Ertel’s potential vorticity (PV) and comparing with ERA5. PV is a strong diagnostic of dynamical balance in the upper troposphere and stratosphere.

**Theoretical Background:**
Ertel’s PV (in pressure coordinates):

$$
PV = -g(\zeta + f) \cdot \nabla_p \theta
$$

Where:

* $\theta = T \left( \frac{1000}{p} \right)^\kappa$ (potential temperature)
* $\zeta$: relative vorticity from wind
* $f$: Coriolis parameter
* $\nabla_p \theta$: θ-gradient in pressure coordinates

**Input Data (GraphCast):**

* Temperature $T$, Wind $u, v$, Geopotential $z$
* Pressure: 100–1000 hPa (≤50 hPa spacing)
* Resolution: ≤50 km
* Domain: Global

**Computation Method:**

1. Compute $\theta$ from $T, p$
2. Calculate relative vorticity $\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$
3. Compute $\nabla_p \theta$, then PV
4. Compare GraphCast PV to ERA5:

   * RMSE at each level
   * Skill score:

$\text{Skill}=\frac{RMSE_{\text{ERA5}}-RMSE_{\text{GraphCast}}}{RMSE_{\text{ERA5}}+RMSE_{\text{GraphCast}}}$

**Evaluation Output:**

* Zonal-mean PV sections (lat–pressure)
* Maps of PV at key levels (e.g., 300, 500 hPa)
* Difference plots (GraphCast − ERA5)
* Outlier flags: PV > 5 PVU at low levels, negative PV in mid/upper troposphere
* Scatter plots (GraphCast vs ERA5 PV)

---

### Spectral Analysis

**Objective:**
Assess the spatial scale representation of ML model outputs (e.g., GraphCast) by evaluating their **power spectral density (PSD)**. Compare with reference datasets (e.g., ERA5) using **Wasserstein distance** to quantify spectral similarity.

**Theoretical Background:**

**Power Spectrum Density (PSD):**
Quantifies energy distribution across spatial scales.
For regular grids:

$$
\text{PSD} = \frac{|\text{FFT2}(x)|^2}{L_x L_y}
$$

For irregular/global grids:
  * Interpolate to uniform grid
  * Use Lomb-Scargle periodogram (LSSA)
  * Use spherical harmonics (e.g., `torch-harmonics`)
  * Tile and assume locally uniform grid

 **Wasserstein Distance (WD):**
Measures difference between distributions (e.g., PSDs):

$$
W_1(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|] 
$$

$$
= \int_{-\infty}^{\infty}|P_r - P_g|
$$

* $p_r, p_g$ - PSDs
* $P_r, P_g$ - respective CDFs

**Input Data:**

* 2D spatial fields (e.g., temperature, wind)
* Grid info (e.g., lat/lon, Lambert Conformal)

**Workflow Steps:**

1. Preprocess (optional: interpolate to regular grid)
2. Compute 2D FFT
3. Calculate radially averaged PSD
4. Plot PSD in log–log space
5. (Optional) Compute WD between ML and reference PSDs

**Diagnostics and Output:**

* PSD plots (log–log)
* Wasserstein distance score
* Spectral slope (optional)
* Identify missing/high-bias scales in ML model, effective resolution

---
### Case studies

- Standard metrics computed on regions with a signifficant event.
- Gradient matching (confluence on fronts).

---
### Climatology

- Extrema localization.
- Diurnal and annual cycle (wind direction).


---
## 3. References and Further Notes

* MetPy, xarray, dask, Pytorch for implementation
* ERA5, Graphcast, Cordiff documentation
