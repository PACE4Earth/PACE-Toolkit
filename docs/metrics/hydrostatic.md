### **Hydrostatic Balance Diagnostics**

---

### **1. Objective**

This workflow evaluates the physical consistency of ML model outputs by testing for **hydrostatic balance** — the equilibrium between the vertical pressure gradient and gravitational force.  

The goal is to quantify **how much the model deviates** from this fundamental assumption of large-scale atmospheric dynamics, including the effects of moisture through virtual temperature.

---

### **2. Theoretical Background**

Hydrostatic balance is defined by:

$$
\frac{\partial \Phi}{\partial p}=-\frac{R_d T_v}{p}
$$

Where:

- $\Phi = g z$ — geopotential \($[m^2/s^2]$\)  
- $p$ — pressure (Pa)  
- $R_d$ — gas constant for dry air ($\sim 287$ J/(kg·K))  
- $T_v$ — **virtual temperature**, accounting for moisture:  

$$
T_v = T \cdot (1 + 0.61 \, q)
$$

with $q$ being the specific humidity (kg/kg).

For discrete levels, the geopotential difference is approximated as:

$$
\Delta \Phi_{\text{hydro}} = -R_d \cdot \bar{T_v} \cdot \ln\frac{p_2}{p_1}
$$

where $\bar{T_v}$ is the mean virtual temperature between pressure levels $p_1$ and $p_2$.  

The actual geopotential difference is:

$$
\Delta \Phi_{\text{actual}} = \Phi(p_2) - \Phi(p_1)
$$

---

### **3. Input Data**

The metric requires model output fields on a latitude–longitude pressure grid:

| Variable       | Name in sample dict          | Dimensions |
|-------------------|---------------------|----------------------|
| Geopotential       | `geopotential`       | [B, L, H, W]   |
| Temperature        | `temperature`        | [B, L, H, W]         |
| Specific Humidity  | `specific_humidity`  | [B, L, H, W]         |

**Domain Considerations:**

- Vertical resolution: ~25–50 hPa spacing  
- Spatial resolution: hydrostatic balance is meaningful at synoptic scales (~25 km or coarser)  

---

### **4. Workflow Steps**

#### **Step 1: Compute Virtual Temperature**

From temperature and specific humidity:

$$
T_v = T \cdot (1 + 0.61 \, q)
$$

#### **Step 2: Compute Hydrostatic Geopotential Differences**

For each adjacent level pair:

$$
\Delta \Phi_{\text{hydro}} = -R_d \cdot \bar{T_v} \cdot \ln\frac{p_{i+1}}{p_i}
$$

- $\bar{T_v} = 0.5 \, (T_{v,i} + T_{v,i+1})$  
- Levels are assumed **top-to-bottom**; if input is bottom-to-top, the code flips arrays.

#### **Step 3: Compute Actual Geopotential Differences**

$$
\Delta \Phi_{\text{actual}} = \Phi_{i+1} - \Phi_i
$$

#### **Step 4: Calculate Errors**

- **Absolute error per level:**

$$
\Delta \Phi_{\text{error}} = \Delta \Phi_{\text{actual}} - \Delta \Phi_{\text{hydro}}
$$

- **Relative error per level:**

$$
\text{RelError} = \frac{|\Delta \Phi_{\text{error}}|}{|\Delta \Phi_{\text{hydro}}| + \epsilon}
$$

with $\epsilon = 10^{-5}$ to avoid division by zero.

- **RMSE aggregated over levels (keep H, W):**

$$
\text{RMSE}_{H,W} = \sqrt{\frac{1}{L-1} \sum_{i=1}^{L-1} (\Delta \Phi_{\text{error},i})^2 }
$$

- Absolute and relative errors are **padded with NaN** to match the original level dimension.


---

### **5. Outputs**

- `hydrostatic_abs_error` — absolute deviation per level [B, L, H, W]  
- `hydrostatic_rel_error` — relative deviation per level [B, L, H, W]  
- `hydrostatic_rmse` — RMSE aggregated over levels [B, 1, H, W]  
- Maps, plots, and summary statistics  

---
