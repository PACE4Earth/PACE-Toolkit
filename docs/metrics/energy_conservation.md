# Energy Conservation Metric

### 1. Objective

This module evaluates **energy consistency** in ML weather model outputs by computing the **total atmospheric energy** in each column.  
It combines **internal, kinetic, latent, and potential energy**, optionally performing a **vertical integration** to yield column-integrated and domain-total energy.  

The metric is useful for monitoring whether a model conserves total energy over time and space.

---

### 2. Theoretical Background

The **total energy per unit mass** in an atmospheric column consists of four components:

1. **Internal (enthalpy) energy:**
$$
E_\mathrm{int} = c_\mathrm{moist} \, T
$$
with  
$$
c_\mathrm{moist} = (1-q) \, c_{pd} + q \, c_{pv}
$$  
where:  
- $T$ = temperature [K]  
- $q$ = specific humidity [kg/kg]  
- $c_{pd} = 1005$ J/(kg·K), dry air heat capacity  
- $c_{pv} = 1855$ J/(kg·K), water vapor heat capacity  

2. **Kinetic energy:**
$$
E_\mathrm{kin} = \frac{1}{2} (u^2 + v^2)
$$
with horizontal wind components $u, v$ [m/s].

3. **Latent energy:**
$$
E_\mathrm{lat} = L_v \, q
$$
with latent heat of vaporization $L_v = 2.5\times 10^6$ J/kg.

4. **Potential energy:**
$$
E_\mathrm{pot} = \Phi
$$
where $\Phi$ is geopotential [m²/s²].

The **column-integrated energy** is computed using hydrostatic approximation:

$$
E_\mathrm{col} = \frac{1}{g} \sum_{k} \bar{E}_k \, \Delta p_k
$$

where $\bar{E}_k$ is the mean energy of layer $k$ and $\Delta p_k$ its pressure thickness, and $g = 9.81$ m/s².

Finally, the **domain-total energy** is obtained by summing over all horizontal grid points.

---

### 3. Input Data

The metric requires model output fields on a latitude–longitude pressure grid:

| Variable                  | Name in sample dict              | Dimensions   |
|----------------------------|---------------------------------|--------------|
| Temperature               | `"temperature"`                 | [B, L, H, W]|
| U wind component          | `"u_component_of_wind"`         | [B, L, H, W]|
| V wind component          | `"v_component_of_wind"`         | [B, L, H, W]|
| Specific humidity         | `"specific_humidity"`           | [B, L, H, W]|
| Geopotential              | `"geopotential"`                | [B, L, H, W]|

Additional grid metadata (optional for column integration):  
- `pressure_levels` → array of model pressure levels [Pa]  

---

### 4. Workflow in Code

1. **Compute level energy**  
   - Compute moist heat capacity $c_\mathrm{moist}$ from temperature and specific humidity.  
   - Calculate each energy component: internal, kinetic, latent, potential.  
   - Sum components to get total energy per level: `E_total = E_int + E_kin + E_lat + E_pot`.

2. **Vertical integration (optional)**  
   - Compute layer thickness from pressure differences: $\Delta p_k = p_{k+1} - p_k$.  
   - Average adjacent levels for each layer and multiply by $\Delta p_k / g$.  
   - Sum over layers to obtain **column-integrated energy** per grid point.

3. **Domain integration (optional)**  
   - Sum column-integrated energy over all horizontal grid points to get **domain-total energy** per batch sample.

4. **Batch support**  
   - Vectorized over batch [B], returning level, column, and domain energies.

---

### 5. Outputs

- **Per-level energy:**  
  `"energy"` → Tensor [B, L, H, W], total energy per grid point and level.  

- **Column-integrated energy:**  
  `"total_energy_column"` → Tensor [B, H, W], energy integrated over vertical layers.  

- **Domain-total energy:**  
  `"total_energy"` → Tensor [B, 1], scalar total energy per batch sample.

---

### 6. Interpretation

- **Consistency:** Metric should remain approximately constant for energy-conserving models.  
- **Deviations:** Large changes indicate potential numerical dissipation, unphysical energy sources, or model errors.  
- **Applications:** Can be used for evaluation of ML-generated fields, physics-informed losses, or climate diagnostics.
