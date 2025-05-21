### **1\. Objective**

This workflow evaluates the physical consistency of ML model outputs (starting with GraphCast) by testing for hydrostatic balance — the equilibrium between the vertical pressure gradient and gravitational force. The goal is to quantify how much and where the model deviates from this fundamental assumption of large-scale atmospheric dynamics.

###

### **2\. Theoretical Background**

Hydrostatic balance is defined by:

$$
\frac{\partial \Phi}{\partial p}=-\frac{R_d T}{p}
$$

Where:
- $\Phi=g z=$ geopotential
- $\boldsymbol{p}=$ pressure level (Pa)
- $R_d=$ dry air gas constant $(\sim 287 \mathrm{~J} / \mathrm{kg} \cdot \mathrm{K})$
- $T=$ temperature (K)

Rewriting for vertical derivative of geopotential between pressure levels:

$\Delta \Phi = -\displaystyle \int_{p_1}^{p_2} \dfrac{R_d T}{p} \, dp$

Assuming discrete levels, this can be approximated as:

$\Phi_{p_2}-\Phi_{p_1} \approx-R_d \cdot \bar{T} \cdot \ln \left(\frac{p_2}{p_1}\right)$

Where:
- $\bar{T}=$ average temperature between $p_1$ and $p_2$

This gives the theoretical hydrostatic geopotential difference.

We will compare this to the actual geopotential difference:

$\Delta \Phi_{\text {actual }}=\Phi\left(p_2\right)-\Phi\left(p_1\right)$

### **3\. Input data**

**GraphCast Outputs Required (as Input):**

| **Variable Type** | **Variable Name** | **Pressure Levels (hPa)** |
| --- | --- | --- |
| Geopotential | z   | 850, 700 |
| Temperature | t   | 850, 700 |
| Latitude, Longitude | lat, lon | —   |

**Domain**

Pressure levels: Apply from ~1000 (900) to 100 hPa (excluding near-surface and stratospheric extremes)

Vertical resolution: At least ~25–50 hPa spacing for meaningful evaluation

Lat, Lon range: Global (0° to 360°)

**Spatial Resolution Considerations:**

- Hydrostatic balance holds best on synoptic scales; model resolution of ~25 km is generally suitable.
- Gradients may be noisy at high resolution; smoothing or averaging over gridboxes may be tested.

### **4\. Workflow Steps**

#### **Step 1: Compute Theoretical Geopotential Difference**

Use hydrostatic equation to compute:

$\Delta \Phi_{\mathrm{hydro}}=-R_d \cdot \bar{T} \cdot \ln \left(\frac{p_2}{p_1}\right)$

#### **Step 2: Compute Actual Geopotential Difference**

$\Delta \Phi_{\text {actual }}=\Phi\left(p_2\right)-\Phi\left(p_1\right)$

#### **Step 3: Calculate errors**

Absolute Error:  
$\Delta \Phi_{\text {error }}=\Delta \Phi_{\text {actual }}-\Delta \Phi_{\text {hydro }}$

RMSE across all gridpoints:  
$RMSE=\sqrt{\dfrac{1}{N} \sum\left(\Delta \Phi_{\mathrm{error}}\right)^2}$

Relative Error:  

$\text{RelError} =\dfrac{\Delta \Phi_{\text {error }}}{\Delta \Phi_{\text {hydro }}}$

#### **Step 4: Compare to ERA5 Baseline**

- Apply the same computation to ERA5 geopotential and temperature
- Compute RMSE for ERA5
- Define skill score:

$\text{Skill}=\frac{RMSE_{\text{ERA5}}-RMSE_{\text{GraphCast}}}{RMSE_{\text{ERA5}}+RMSE_{\text{GraphCast}}}$

+1 = perfect prediction (zero error)   
0 = same as ERA5  
–1 = very poor (RMSE much higher than baseline)

####

####

#### **Step 5: Detect Physically Unrealistic Values and Outliers**

To enhance diagnostic accuracy, flag data points that violate known physical relationships or exhibit statistically extreme behavior:

- Physically implausible values (e.g., unrealistic temperature, pressure gradients, or geopotential changes) will be detected using empirically derived or literature-based thresholds.
- Outliers will be identified using statistical methods such as z-scores or interquartile range (IQR), based on the distributions of errors.

#### **Step 6: Visualization and Diagnostics**

- Skill scores
- Maps of
$\Delta \Phi_{\text {error }}$
- Histogram of errors and relative error
- Scatter plots:  
$\Delta \Phi_{\text {hydro }} \text {vs } \Delta \Phi_{\text {actual }}$
- Spatial frequency map of flagged imbalanced points

### **5\. Results (Planned Output)**

- RMSE and Skill Score between actual and hydrostatic geopotential differences
- % of points exceeding hydrostatic imbalance thresholds
- Error patterns across latitude, seasons, and levels
- Visuals: spatial error maps, scatter plots, histograms

### **6\. Future Extensions and Ideas**

- Extend to multiple / all level pairs (e.g., 925–850, 700–500, 300–200)
- Apply to other ML models, if vertical resolution allows
- Define typical RMSE ranges across pressure levels
- Explore additional metrics: MAE, bias, relative error

### **Tools and Libraries**

- **xarray, numpy, dask, pytorch** – data handling and computing
- **metpy** – gradient, Coriolis parameter, projections
- **matplotlib** – plotting and mapping
- **cdsapi** – data access (ERA5)
