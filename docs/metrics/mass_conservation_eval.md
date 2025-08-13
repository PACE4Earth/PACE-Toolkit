### **1. Objective**

This workflow aims to evaluate the mass conservation consistency of GraphCast ML model outputs by calculating the mass tendency and comparing it with the divergence of the mass flux. The evaluation will help identify imbalances in mass continuity, particularly in dynamically active regions or during rapid developments (e.g., frontal zones, cyclogenesis).

The method can later be extended to other ML models (e.g., CorrDiff) and compared with reanalysis datasets (e.g., ERA5).

---

### **2. Theoretical Background**

Under the assumption of hydrostatic balance and using pressure coordinates, the continuity equation for dry or moist air can be written as:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v}) = 0
$$

For dry air:

$$
\rho = \frac{p}{R_d T}
$$

For moist air:

$$
\rho = \frac{p}{R_d T} \cdot \frac{1}{1 + 0.61q_v}
$$

Where:

* \$\rho\$ is air density
* \$\vec{v} = (u, v)\$ is the horizontal wind vector
* \$p\$ is pressure
* \$T\$ is temperature
* \$q\_v\$ is specific humidity
* \$R\_d\$ is the gas constant for dry air

Mass consistency implies that the local rate of change of mass matches the net inflow/outflow of mass per unit area.

---

### **3. Input Data**

**GraphCast Outputs Required (as Input):**

| **Variable Type**   | **Variable Name** | **Pressure Levels (hPa)**            |
| ------------------- | ----------------- | ------------------------------------ |
| Temperature         | `t`               | 100 to 1000                          |
| U wind component    | `u`               | 100 to 1000                          |
| V wind component    | `v`               | 100 to 1000                          |
| Specific Humidity   | `q`               | 100 to 1000                          |
| Pressure            | `p`               | 100 to 1000 (can be fixed per level) |
| Latitude, Longitude | `lat`, `lon`      | —                                    |

Resolution:

* Horizontal: ≤25 km (GraphCast default)
* Temporal: e.g., 3-hourly forecasts to compute tendencies

---

### **4. Workflow Steps**

#### Step 1: Compute Density ($\rho$) at each level

$$
\rho = \frac{p}{R_d T (1 + 0.61 q)}
$$

#### Step 2: Compute Mass Flux

$$
\vec{F}_m = \rho \vec{v} = \rho u, \rho v
$$

#### Step 3: Compute Divergence of Mass Flux

Using centered finite differences (or `xarray` gradient):

$$
\nabla \cdot (\rho \vec{v}) = \frac{\partial (\rho u)}{\partial x} + \frac{\partial (\rho v)}{\partial y}
$$

#### Step 4: Compute Time Derivative of Mass (Local Tendency)

From forecast snapshots:

$$
\frac{\partial \rho}{\partial t} \approx \frac{\rho(t+\Delta t) - \rho(t)}{\Delta t}
$$

#### Step 5: Compute Residual

$$
\text{Residual} = \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v})
$$

---

### **5. Comparison and Diagnostics**

* Compute residual RMS over space/time
* Visualize 2D maps of divergence, tendencies, and residuals
* Identify regions of significant inconsistency (e.g., fronts, jets)
* Normalize residual by typical mass magnitude for relative error

---

### **6. Tools and Libraries**

* `xarray`, `numpy`, `torch` – for data operations and model interfacing
* `matplotlib` or `cartopy` – for plotting and diagnostics
* `scipy.ndimage` – for finite-difference operations if needed

---

### **7. Extensions**

* Apply to other ML models
* Compare with ERA5 divergence diagnostics
* Vertical integration for column mass budget
* Introduce precipitation/evaporation terms for full water mass budget

---

### **8. Output**

* Residual maps and zonal mean sections
* Skill scores and RMSE for conservation consistency
* Optional: animation of time evolution
