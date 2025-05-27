### **1. Objective**

This workflow aims to evaluate the physical consistency of GraphCast ML model outputs by analyzing **energy conservation** in the atmosphere. Specifically, we assess whether the model conserves total energy (internal, kinetic, and latent) in the absence of external forcing, sources, or sinks.

This check is foundational for assessing physical realism in any ML-based weather forecasting model.

---

### **2. Theoretical Background**

The total energy per unit mass of moist air includes:

* **Internal energy**:
  $E_{\text{int}} = c_{v,\text{moist}} \cdot T$

* **Kinetic energy**:
  $E_{\text{kin}} = \frac{1}{2}(u^2 + v^2)$

* **Latent energy** (from water vapor):
  $E_{\text{lat}} = L_v \cdot q_v$

Thus, the **total energy** per unit mass is:
$E_{\text{total}} = c_{v,\text{moist}} \cdot T + \frac{1}{2}(u^2 + v^2) + L_v \cdot q_v$

Where:

* $c_{v,\text{moist}}$: specific heat at constant volume for moist air
* $T$: temperature (K)
* $u, v$: horizontal wind components (m/s)
* $L_v$: latent heat of vaporization (\~2.5 × 10^6 J/kg)
* $q_v$: specific humidity (kg/kg)

---

### **3. Input Data**

| **Variable Type**   | **Variable Name** | **Levels**   |
| ------------------- | ----------------- | ------------ |
| Temperature         | `t`               | 100–1000 hPa |
| U wind component    | `u`               | 100–1000 hPa |
| V wind component    | `v`               | 100–1000 hPa |
| Specific humidity   | `q`               | 100–1000 hPa |
| Surface pressure    | `sp`              | surface      |
| Geopotential        | `z`               | 100–1000 hPa |
| Latitude, Longitude | `lat`, `lon`      | —            |

---

### **4. Workflow Steps**

**Step 1: Compute energy components**
Using xarray and PyTorch:

```python
E_int = c_v_moist * T
dE_kin = 0.5 * (u**2 + v**2)
E_lat = L_v * q
E_total = E_int + E_kin + E_lat
```

**Step 2: Integrate vertically**
Approximate vertical integration via pressure thickness:

```python
dp = pressure.diff('level')  # Or predefined layer thickness
E_column = (E_total * dp / g).sum(dim='level')
```

**Step 3: Check temporal change**
Evaluate conservation by computing the total column energy tendency:

```python
dE_dt = E_column.diff('time') / dt
```

Expected: $\frac{\partial E}{\partial t} \approx 0$ in the absence of sources/sinks.

**Step 4: Visualize and Compare**

* Global maps of total column energy
* Time series of domain-mean $E_{\text{total}}$
* Zonal means

---

### **5. Diagnostics and Metrics**

* **Energy budget closure error**: max, mean, RMSE of $\frac{\partial E}{\partial t}$
* **Energy tendency maps** (regions of large imbalance)
* **Zonal and vertical energy cross-sections**
* **Skill Score** compared to ERA5 total energy budget

---

### **6. Tools and Libraries**

* `xarray`, `numpy`, `torch` – array manipulation
* `matplotlib`, `cartopy` – plotting
* `dask` – lazy parallel computation (for large data)
* `metpy.constants` – for constants like $R_d, L_v$

---

### **7. Future Extensions**

* Incorporate vertical motion $\omega$ and 3D flux divergence
* Compare total energy budget against reanalysis (ERA5)
* Apply to CorrDiff and other ML models
* Assess conservation in perturbed (stochastic) forecasts

---
