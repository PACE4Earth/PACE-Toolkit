### **1. Objective**

This workflow aims to evaluate the physical consistency of GraphCast ML model outputs by assessing **energy conservation**. We focus on the **total energy** of the atmosphere, including internal, kinetic, and latent energy, and verify that changes in energy across time are physically consistent.

This diagnostic is critical for determining whether machine learning models respect fundamental atmospheric energy constraints and can be extended to other models such as CorrDiff.

---

### **2. Theoretical Background**

The total energy per unit mass of moist air includes:

- **Internal energy**:  
  \( E_{\text{int}} = c_{v,\text{moist}} \cdot T \)
- **Kinetic energy**:  
  \( E_{\text{kin}} = \frac{1}{2}(u^2 + v^2) \)
- **Latent energy** (from water vapor):  
  \( E_{\text{lat}} = L_v \cdot q_v \)

Thus, the **total energy** per unit mass is:  
\[
E_{\text{total}} = c_{v,\text{moist}} \cdot T + \frac{1}{2}(u^2 + v^2) + L_v \cdot q_v
\]

Assumptions:
- No vertical velocity or geopotential terms are considered at this stage.
- Mass-weighted grid averages are used for global diagnostics.

---

### **3. Input data**

**GraphCast Outputs Required (as Input):**

| **Variable Type** | **Variable Name** | **Pressure Levels (hPa)** |
|-------------------|-------------------|----------------------------|
| Temperature       | t                 | 1000 to 100 (≤50 hPa spacing) |
| U wind component  | u                 | 1000 to 100 |
| V wind component  | v                 | 1000 to 100 |
| Specific humidity | q                 | 1000 to 100 |
| Latitude, Longitude | lat, lon       | — |

Spatial and vertical resolution:

- Horizontal resolution ≤25 km (GraphCast default)
- Vertical resolution: ≤50 hPa spacing, especially in the upper troposphere

---

### **4. Workflow Steps**

#### **Step 1: Compute Total Energy**

At each grid point and pressure level:

- Use constants:
  - \( c_{v,\text{moist}} \approx 718 \, \mathrm{J/(kg \cdot K)} \)
  - \( L_v \approx 2.5 \times 10^6 \, \mathrm{J/kg} \)

- Compute:
  \[
  E_{\text{total}} = 718 \cdot T + \frac{1}{2}(u^2 + v^2) + 2.5 \times 10^6 \cdot q
  \]

#### **Step 2: Aggregate Over Grid**

- Integrate vertically using pressure thickness (e.g. trapezoidal rule)
- Average horizontally over all lat-lon grid points

#### **Step 3: Time Evolution Analysis**

- Compute total atmospheric energy at each time step
- Calculate time derivative (energy tendency)
- Compare to expected net radiative flux divergence (if available)

#### **Step 4: Comparison and Diagnostics**

- Visualize energy tendencies
- Detect unphysical jumps or inconsistencies
- Compare against ERA5 energy time series if available

---

### **5. Planned Outputs**

- Global time series of total energy
- Time derivative of total energy (tendency)
- Plots of each energy component (internal, kinetic, latent)
- Skill score or RMSE if compared to reanalysis

---

### **6. Tools and Libraries**

- `xarray`, `numpy`, `torch` – data handling
- `matplotlib`, `seaborn` – visualization
- `dask` – parallel processing for large datasets
