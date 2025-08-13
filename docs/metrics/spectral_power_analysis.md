### **1\. Objective**

Evaluation of power spectrum provides many insights about atmospheric fields. Issues related to reproduction of high-resolution features are a known limitations of many current MLWP approaches.

Objective is to compute and plot PSD (power spectrum density) for individual variable fields. If reference dataset is provided, first order Wasserstein distance is computed.

###

### **2\. Theoretical background**

**Power Spectrum Density**

Basic implementations (numpy, torch) assume uniform grid. This is appropriate for LAM grids (e.g. LCC).

Computations for other common grids (e.g. regular latlon,gaussian grid) can be addressed mutliple ways:
* interpolation to uniform grid
* LSSA (Lomb-Scargle periodogram)
* tiling, assumed locally uniform grid $\overline{\delta x}, \overline{\delta y}$

For global grids, direct use of spherical harmonics (utilizing [torch-harmonics](https://github.com/NVIDIAtorch-harmonics/tree/main)) might be preferred.

Restriction to computation on midlatitudes might be considered.


**Wasserstein distance**
On surface level, the metric computes the difference of two distributions. 

$$
    W_d(\text{P}_r, \text{P}_g) = \inf_{\gamma \in \Pi
    (\text{P}_r, \text{P}_g)} \text{E}_{(x,y) \sim \Pi}
    [||x-y||_d]
$$

where $\text{P}_g, \text{P}_d$ are marginal distributions,$\Pi$ is a set of all joint distributions. 

For the most  common case $d=1$ the metric is often called Earth Mover's Distance, from the intuition of "moving histogram height".

In many modern GAN architectures, WD has been prefered over KL or JS divergence due to stabilty of the metric.
[Arjovsky et al, 2017](https://arxiv.org/pdf/1701.07875v3)


###

### **3\. Input data**

Any 2D field, grid description.

### **4\. Workflow**

**Basic version:**
Discrete walues of wavenumbers $r$ are computed based on the input reoslution and grid size.

FFT is performed, magnitude and normalization is computed. 

$$
\begin{aligned}
    \xi &\leftarrow \text{torch.fft.fft2}(x) \\
    \text{PSD} &\leftarrow \frac{|\xi|^2}{L_xL_y}
\end{aligned}
$$

**Visualization**
Most commonly on log log scale.

**Wasserstein Distance Computation**
