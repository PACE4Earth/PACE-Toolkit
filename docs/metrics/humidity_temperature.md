# Relative Humidity Consistency Metric

### 1. Objective

This module evaluates the physical consistency of ML model outputs by comparing **specific humidity** with **temperature-dependent saturation values** derived from the Clausius–Clapeyron relation.  
The metric outputs **relative humidity (RH, %)** and highlights unrealistic moisture–temperature relationships such as supersaturation or negative humidity.

---

### 2. Theoretical Background

The Clausius–Clapeyron equation describes the exponential dependence of saturation vapor pressure $e_s$ on temperature $T$:

$$
\frac{d e_s}{d T} = \frac{L_v e_s}{R_v T^2}
$$

where:
- $L_v \approx 2.5 \times 10^6 \,\text{J kg}^{-1}$ = latent heat of vaporization  
- $R_v \approx 461 \,\text{J kg}^{-1}\,\text{K}^{-1}$ = gas constant for water vapor  
- $T$ = absolute temperature [K]  
- $e_s$ = saturation vapor pressure [Pa]  

Using an integrated form, the saturation vapor pressure is:

$$
e_s(T) = e_0 \, \exp \!\left( \frac{L_v}{R_v}\left(\frac{1}{T_0} - \frac{1}{T}\right) \right)
$$

with $e_0 = 611.2 \,\text{Pa}$ at $T_0 = 273.15 \,\text{K}$.

From $e_s$, the saturation specific humidity $q_s$ is computed as:

$$
q_s(T,p) = \frac{\epsilon \, e_s}{p - (1 - \epsilon) e_s}
$$

where $\epsilon = 0.622$ and $p$ is pressure [Pa].

Relative humidity is then defined as:

$$
RH = \frac{q}{q_s} \times 100
$$

---

### 3. Input Data

The metric requires model outputs on a latitude–longitude pressure grid:

| Variable           | Name in sample dict     | Dimensions   |
|--------------------|-------------------------|--------------|
| Temperature        | `"temperature"`         | [B, L, H, W] |
| Specific Humidity  | `"specific_humidity"`   | [B, L, H, W] |

Additional grid metadata:
- `pressure_levels`: pressure levels in hPa (converted to Pa internally)  

**Domain:**  
- Pressure levels: 1000–500 hPa (low–mid troposphere)  
- Global coverage at model resolution (~25 km for GraphCast)  

---

### 4. Workflow in Code

1. **Compute $e_s(T)$**  
   - From Clausius–Clapeyron exponential formulation.  
   - Constants: $L_v, R_v, e_0, T_0$.  

2. **Compute $q_s(T,p)$**  
   - Using definition above with pressure at each level.  

3. **Compute relative humidity**  
   - $RH = (q / q_s) \times 100$  
   - Small $\epsilon$ constant avoids division by zero.  

4. **Output**  
   - Field of relative humidity [%], dimension [B, L, H, W].  

---

### 5. Outputs

- **Primary output:**  
  `"relative_humidity"` → Tensor [B, L, H, W]  
  Relative humidity in percent (%).  

---

### 6. Interpretation

- **$RH \approx 40–80\%$:** Typical free-tropospheric range.  
- **$RH > 100\%$:** Supersaturation → physically unrealistic, should be flagged.  
- **$RH < 0$:** Negative humidity → invalid, indicates model inconsistency.  

---
