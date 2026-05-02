# Kernel Density Estimation, Autoencoders, and Anomaly Detection

## 1. Introduction to Unsupervised Learning

In unsupervised learning, the dataset consists of unlabeled examples:

$$\mathcal{D} = \{x_i\}_{i=1}^N$$

The goal is to discover hidden structure or patterns in the data, such as clusters, low-dimensional representations, or density estimates.

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|---------------------|----------------------|
| Data | $\{(x_i, y_i)\}$ | $\{x_i\}$ |
| Goal | Learn mapping $x \to y$ | Discover hidden structure |
| Evaluation | Label-based metrics (accuracy, etc.) | Task-dependent metrics |

### Main Approaches

| Approach | Goal |
|----------|------|
| **Clustering** | Partition data into groups where samples within a cluster are more similar to each other |
| **Dimensionality Reduction** | Find a low-dimensional representation preserving important structure |
| **Density Estimation** | Estimate the probability density function of the unknown data distribution |
| **Autoencoders** | Learn compact representations by reconstructing input from a lower-dimensional code |
| **Anomaly Detection** | Identify rare or unusual samples that differ from the majority |

---

## 2. Kernel Density Estimation

### Model

Given samples $\{x_i\}_{i=1}^N$ drawn from an unknown pdf $f(x)$, the kernel density estimator is:

$$\hat{f}_b(x) = \frac{1}{Nb} \sum_{i=1}^N k\left(\frac{x - x_i}{b}\right)$$

where $b > 0$ is the **bandwidth** and $k(\cdot)$ is a kernel function.

A common choice is the **Gaussian kernel**:

$$k(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

### Role of the Bandwidth

The bandwidth is the most important hyperparameter:

* **Too small ($b$ small):** estimate becomes spiky, fits noise $\rightarrow$ overfitting
* **Too large ($b$ large):** estimate becomes overly smooth, loses structure $\rightarrow$ underfitting

KDE is a **non-parametric** method — it does not assume a specific parametric form for the density.

---

## 3. Autoencoders

### Formulation

An autoencoder is a neural network that learns to reconstruct its input. It consists of:

* **Encoder:** maps input to a latent representation $z = f_\theta(x)$
* **Decoder:** reconstructs input from the latent code $\hat{x} = g_\phi(z)$

The model is trained to minimize the reconstruction error:

$$\min_{\theta, \phi} \sum_{i=1}^N \|x_i - \hat{x}_i\|^2$$

If the latent dimension $m < d$ (input dimension), the bottleneck forces the network to learn a compact representation. Autoencoders can be viewed as a **nonlinear** approach to dimensionality reduction.

---

## 4. Anomaly Detection

Anomaly detection identifies samples that differ significantly from the majority. It is often treated as unsupervised because anomaly labels are scarce.

**Common assumptions:**
* Normal samples are much more frequent than anomalies
* Normal samples form meaningful patterns
* Anomalies deviate from these patterns

### Main Approaches

| Approach | Idea |
|----------|------|
| **Distance-based** | A sample is anomalous if far from most other samples |
| **Density-based** | A sample is anomalous if it lies in a low-density region ($p(x)$ small) |
| **Clustering-based** | Samples not belonging to any major cluster are anomalies |
| **Reconstruction-based** | If an autoencoder reconstructs a sample poorly, it may be anomalous |

For reconstruction-based methods, the anomaly score is:

$$\|x - \hat{x}\|^2$$

A large reconstruction error suggests the sample may be abnormal.
