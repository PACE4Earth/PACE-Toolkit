### **1\. Objective**

This workflow aims to assess the physical consistency of ML model outputs by evaluating the relationship between specific humidity and temperature. We compare the ML-generated moisture variations with theoretical expectations given by the Clausius-Clapeyron equation, to detect unrealistic moisture–temperature relationships.

###

### **2\. Theoretical Background**

The Clausius-Clapeyron equation describes the equilibrium vapor pressure ($e_s$​) dependence on temperature (T):

$$
\frac{d e_s}{d T}=\frac{L_v e_s}{R_v T^2}
$$

Where:
- $L_v=$ latent heat of vaporization $\left(\approx 2.5 \times 10^6 \mathrm{~J} / \mathrm{kg}\right)$
- $R_v=$ gas constant for water vapor ( $\approx 461 \mathrm{~J} / \mathrm{kg} \cdot \mathrm{K}$ )
- $T=$ absolute temperature (K)
- $e_s=$ saturation vapor pressure (Pa)

###

Specific humidity (q) should vary in a physically consistent manner with temperature, roughly following Clausius-Clapeyron scaling for saturation vapor pressure and relative humidity.

### **3\. Input data**

**GraphCast Outputs Required (as Input):**

| **Variable Type** | **Variable Name** | **Pressure Levels (hPa)** |
| --- | --- | --- |
| Temperature | T   | 500, 600, 700, 850, 925, etc. |
| Specific Humidity | q   | 500, 600, 700, 850, 925, etc. |
| Latitude, Longitude | lat, lon | —   |

**Domain**

Pressure levels: 1000 to 500 hPa (excluding upper-level low humidity levels)

Lat, Lon range: Global (0° to 360°)

Use same spatial resolution as GraphCast (~25 km)

### **4\. Workflow Steps**

#### **Step 1: Compute Saturation Vapor Pressure $e_s$ from Temperature**

$e_s(T)=e_0 \cdot \exp \left(\frac{L_v}{R_v}\left(\frac{1}{T_0}-\frac{1}{T}\right)\right)$

Where:
- $e_0=$ reference vapor pressure (e.g., 611 Pa at 273.15 K)
- $T_0=$ reference temperature (e.g., 273.15 K)

#### **Step 2: Estimate Expected Specific Humidity $q_s$**

Assuming relative humidity RH≈1 for upper bound consistency:

$q_s(T)=0.622 \cdot \dfrac{e_s}{p-e_s}$

Where $p$ is atmospheric pressure at the level.

#### **Step 3: Compare ML Output Specific Humidity $q_{ML}$​ to Theoretical Expectation $q_s$**

- Calculate the relative humidity estimate from model outputs:

$RH_{ML}=\dfrac{q_{ML}}{q_s}$

- Flag unrealistic values where:

$RH_{ML} > 1.2$ (supersaturation beyond physical plausibility)  
$RH_{ML} < 0$ (negative humidity)

#### **Step 4: Calculate Consistency Metrics**
- RMSE, MAE, bias between $q_{ML} \text{ and } {q_s}$
- Compare with baseline, as in previous workflows

#### **Step 5: Visualization and Diagnostics**

- Scatter plots of $q_{ML}$​ vs. $q_s$​, ideally clustered near 1:1 line.
- Histograms of relative humidity and error distributions.
- Spatial maps highlighting flagged regions of inconsistent moisture–temperature behavior.

### **5\. Results (Planned Output)**

- Quantitative skill scores (MAE, RMSE, bias) for humidity consistency relative to Clausius-Clapeyron expectation and to baseline nwp model
- Visualizations as described above

### **6\. Future Extensions and Ideas**

- Apply across multiple pressure levels (e.g., 500–925 hPa)
- Investigate temporal variability and extreme event cases

### **Tools and Libraries**

- **xarray, numpy, dask, pytorch** – data handling and computing
- **metpy** – Thermodynamic calculations (e.g., saturation vapor pressure)
- **matplotlib** – plotting and mapping
- **cdsapi** – data access (ERA5)
