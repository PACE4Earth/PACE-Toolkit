# Geostrophic Wind Metric

### 1. Objective

This module evaluates geostrophic balance in atmospheric fields by comparing model winds with geostrophic winds derived from geopotential gradients.  
The metric is expressed as the **ratio of ageostrophic to geostrophic wind magnitude**, smoothed to suppress grid-scale noise.  
A ratio close to zero indicates near-geostrophic flow, while higher ratios highlight stronger ageostrophic contributions.

---

### 2. Theoretical Background

Geostrophic balance describes large-scale midlatitude flow where the Coriolis force balances the horizontal pressure gradient force.  

In pressure coordinates, the geostrophic wind components are:

$$
u_g = -\frac{1}{f}\frac{\partial \Phi}{\partial y}, \quad v_g = \frac{1}{f}\frac{\partial \Phi}{\partial x}
$$

where:  
- $\Phi$ = geopotential $\,[m^2 s^{-2}]$  
- $f = 2 \Omega \sin(\phi)$ = Coriolis parameter at latitude $\phi$  
- $x, y$ = horizontal Cartesian coordinates derived from grid spacing  

The ageostrophic wind is defined as the residual:

$$
u_{ag} = u - u_g, \quad v_{ag} = v - v_g
$$

and the evaluated metric is the ratio:

$$
r = \frac{|\vec{v}_{ag}|}{|\vec{v}_g| + \epsilon}
$$

where $\epsilon$ is a small constant to avoid division by zero.

---

### 3. Input Data

The metric requires model output fields on a latitude–longitude pressure grid:

| Variable           | Name in sample dict        | Dimensions   |
|--------------------|----------------------------|--------------|
| Geopotential       | `"geopotential"`           | [B, L, H, W] |
| U wind component   | `"u_component_of_wind"`    | [B, L, H, W] |
| V wind component   | `"v_component_of_wind"`    | [B, L, H, W] |

Additional grid metadata is supplied separately:
- $f$: Coriolis parameter $\,[1/s]$  
- $dx, dy$: grid spacing in $x/y \,[m]$  
- $lat$: latitude array [degrees]  

**Latitude mask:** The ratio is only evaluated for $30^\circ–80^\circ$ N/S, where geostrophic balance is valid. Outside this range, results are masked with NaN.

---

### 4. Workflow in Code

1. **Geopotential gradients**  
   - Gradients $\partial \Phi/\partial x, \partial \Phi/\partial y$ are computed using Sobel filters with finite-difference padding.  
   - Grid spacing ($dx, dy$) scales the derivatives.

2. **Geostrophic wind**  
   - Computed directly from gradients using the balance equations.  
   - Latitude mask applied ($30^\circ–80^\circ$ N/S).

3. **Ageostrophic wind**  
   - Defined as residual of actual wind minus geostrophic wind.  

4. **Ratio calculation**  
   - Ratio of magnitudes computed elementwise.  
   - $\epsilon$ prevents division by zero.

5. **Smoothing**  
   - Final ratio field smoothed with a selectable kernel:  
     - `"uniform"` (default): $4\times4$ averaging  
     - `"gaussian"`: $9\times9$ with $\sigma=1.25$  
   - Hard-coded kernel sizes correspond to $\sim 0.25^\circ \to 2^\circ$ effective resolution.

---

### 5. Outputs

- **Primary output:**  
  `"geostrophic_wind_ratio"` → Tensor [B, L, H, W]  
  Smoothed ratio field of ageostrophic to geostrophic wind magnitude.  

---

### 6. Interpretation

- **$r \approx 0$:** Flow close to geostrophic balance.  
- **Large $r$:** Strong ageostrophic contribution (e.g., near fronts, cyclones, or jet streams).  
- **Masked (NaN):** Outside valid latitude range.

---
