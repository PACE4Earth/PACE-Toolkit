### **1. Objective**

This workflow aims to assess the **Hill-Mandel energy consistency** between resolved and subgrid scales in ML-based atmospheric models like GraphCast. This consistency condition ensures that energy exchange between scales respects physical principles, thereby improving the reliability of model outputs.

---

### **2. Theoretical Background**

The Hill-Mandel condition ensures consistency between the macroscopic (resolved) and microscopic (subgrid) scale work contributions. It is expressed as:

$$
\langle \boldsymbol{\tau} : \nabla \mathbf{u} \rangle = \langle \boldsymbol{\tau} \rangle : \langle \nabla \mathbf{u} \rangle
$$

Where:

* \$\boldsymbol{\tau}\$ is the subgrid-scale stress tensor
* \$\mathbf{u}\$ is the velocity vector
* \$\nabla \mathbf{u}\$ is the velocity gradient tensor
* \$\langle \cdot \rangle\$ denotes spatial (volume) averaging
* \$:\$ represents the double inner product (tensor contraction)

This condition implies that the average energy exchange due to subgrid stresses must match the energy exchange computed from averaged fields.

---

### **3. Input Data**

**GraphCast Outputs Required (as Input):**

| **Variable Type**       | **Variable Name**                | **Levels / Grid**                    |
| ----------------------- | -------------------------------- | ------------------------------------ |
| U wind component        | \$u\$                            | All relevant pressure levels         |
| V wind component        | \$v\$                            | All relevant pressure levels         |
| W wind component        | \$w\$                            | Derived or computed if needed        |
| Subgrid stress tensor   | \$\boldsymbol{\tau}\$            | Same spatial & vertical grid         |
| Grid geometry / spacing | \$\Delta x, \Delta y, \Delta p\$ | Required for gradients and averaging |

---

### **4. Workflow Steps**

**Step 1: Compute the velocity gradient tensor**

In 3D Cartesian coordinates:

$$
\nabla \mathbf{u} = \begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial z} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial z} \\
\frac{\partial w}{\partial x} & \frac{\partial w}{\partial y} & \frac{\partial w}{\partial z} \\
\end{bmatrix}
$$

**Step 2: Compute pointwise energy transfer**

Stress power at each grid point:

$$
P(\mathbf{x}) = \boldsymbol{\tau}(\mathbf{x}) : \nabla \mathbf{u}(\mathbf{x})
$$

**Step 3: Compute averages**

Volume average:

$$
\langle P \rangle = \frac{1}{V} \int_V \boldsymbol{\tau} : \nabla \mathbf{u} \, dV
$$

Compare with:

$$
\langle \boldsymbol{\tau} \rangle : \langle \nabla \mathbf{u} \rangle
$$

**Step 4: Verify Hill-Mandel condition**

Check:

$$
\langle \boldsymbol{\tau} : \nabla \mathbf{u} \rangle \approx \langle \boldsymbol{\tau} \rangle : \langle \nabla \mathbf{u} \rangle
$$

---

### **5. Output and Diagnostics**

* Maps of local stress power \$P(\mathbf{x})\$
* Time evolution of Hill-Mandel discrepancy
* RMSE or relative difference:

$$
\text{Error} = \frac{|\langle \boldsymbol{\tau} : \nabla \mathbf{u} \rangle - \langle \boldsymbol{\tau} \rangle : \langle \nabla \mathbf{u} \rangle|}{|\langle \boldsymbol{\tau} : \nabla \mathbf{u} \rangle|}
$$

---

### **6. Notes and Extensions**

* Include thermal and moisture fluxes in extended definitions.
* Apply the method to different vertical levels.
* Compare across models (GraphCast vs CorrDiff).
