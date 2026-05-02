# Gaussian Mixture Models and the EM Algorithm

## Gaussian Mixture Model

### From K-Means to GMM

K-means limitations:
* Hard assignments (each point to exactly one cluster)
* Represents each cluster only by its centroid
* No probabilistic model for the data distribution

GMM provides a **soft assignment** approach via probabilistic modeling.

### Mixture Model

Each sample $x$ is generated with a hidden variable $z \in \{1, \dots, K\}$ indicating which component generated it:

$$p(x, z) = p(x \mid z) \, p(z)$$

The marginal distribution (since $z$ is unobserved):

$$p(x) = \sum_{k=1}^K p(x \mid z = k) \, \pi_k$$

where $\pi_k = p(z = k)$ are mixing coefficients with $\pi_k \geq 0$ and $\sum_k \pi_k = 1$.

### Gaussian Mixture Model

In a GMM, each component is Gaussian:

$$p(x \mid z = k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Therefore:

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

With sufficiently many components, GMMs can approximate a wide class of densities.

### Maximum Likelihood Estimation

Given data $\mathcal{X} = \{x^{(1)}, \dots, x^{(N)}\}$, the log-likelihood is:

$$\ell(\theta) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(x^{(n)} \mid \mu_k, \Sigma_k) \right)$$

**Difficulty:** the logarithm is outside the summation, so no closed-form solution exists. This motivates the EM algorithm.

---

## Expectation-Maximization Algorithm

### Preliminaries

**Jensen's Inequality:** If $f$ is convex, then $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$. For concave $f$ (like $\log$), the inequality reverses.

**KL Divergence:** For distributions $p$ and $q$:

$$\text{KL}(q \parallel p) = \sum_z q(z) \log \frac{q(z)}{p(z)} \geq 0$$

with equality iff $q(z) = p(z)$. KL divergence is non-negative but not symmetric.

### Derivation via Lower Bound

For one sample $x$ and any distribution $q(z)$:

$$\log p(x \mid \theta) = \log \sum_z q(z) \frac{p(x, z \mid \theta)}{q(z)}$$

By Jensen's inequality (since $\log$ is concave):

$$\log p(x \mid \theta) \geq \sum_z q(z) \log \frac{p(x, z \mid \theta)}{q(z)} := \mathcal{F}(q, \theta)$$

This decomposes as:

$$\log p(x \mid \theta) = \mathcal{F}(q, \theta) + \text{KL}(q(z) \parallel p(z \mid x, \theta))$$

For the full dataset with auxiliary distributions $q_n(z^{(n)})$:

$$\log p(\mathcal{D} \mid \theta) = \sum_{n=1}^N \mathcal{F}_n(q_n, \theta) + \sum_{n=1}^N \text{KL}(q_n(z^{(n)}) \parallel p(z^{(n)} \mid x^{(n)}, \theta))$$

### E-Step

Fix $\theta^{\text{old}}$. To maximize the lower bound, minimize the sum of KL divergences. Since KL $\geq 0$, the minimum is 0, achieved when:

$$q_n^*(z^{(n)}) = p(z^{(n)} \mid x^{(n)}, \theta^{\text{old}})$$

With this choice, the lower bound becomes **tight** at $\theta^{\text{old}}$:

$$\log p(\mathcal{D} \mid \theta^{\text{old}}) = \mathcal{F}(\{q_n^*\}, \theta^{\text{old}})$$

### M-Step

Fix $q_n$. Update $\theta$ by maximizing the lower bound:

$$\theta^{\text{new}} = \arg\max_\theta \sum_{n=1}^N \sum_{z^{(n)}} q_n(z^{(n)}) \log p(x^{(n)}, z^{(n)} \mid \theta)$$

Define the **Q-function**:

$$Q(\theta, \theta^{\text{old}}) = \sum_{n=1}^N \mathbb{E}_{p(z^{(n)} \mid x^{(n)}, \theta^{\text{old}})} \left[ \log p(x^{(n)}, z^{(n)} \mid \theta) \right]$$

Then $\theta^{\text{new}} = \arg\max_\theta Q(\theta, \theta^{\text{old}})$.

### Why EM Increases the Log-Likelihood

Let $\{q_n\}$ be the distributions from the E-step.

1. After E-step: $\log p(\mathcal{D} \mid \theta^{\text{old}}) = \mathcal{F}(\{q_n\}, \theta^{\text{old}})$ (tight)
2. After M-step: $\mathcal{F}(\{q_n\}, \theta^{\text{new}}) \geq \mathcal{F}(\{q_n\}, \theta^{\text{old}})$ (by maximization)
3. Since log-likelihood $\geq$ lower bound: $\log p(\mathcal{D} \mid \theta^{\text{new}}) \geq \mathcal{F}(\{q_n\}, \theta^{\text{new}})$

Therefore:

$$\log p(\mathcal{D} \mid \theta^{\text{new}}) \geq \log p(\mathcal{D} \mid \theta^{\text{old}})$$

EM guarantees monotonic increase of the observed-data log-likelihood.

---

## EM for GMM

**E-step:** Compute responsibilities (posterior probabilities):

$$\gamma_k^{(n)} := q_n(z^{(n)} = k) = \frac{\pi_k \, \mathcal{N}(x^{(n)} \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x^{(n)} \mid \mu_j, \Sigma_j)}$$

**M-step:** Update parameters using responsibilities as weights. Define effective sample size:

$$N_k = \sum_{n=1}^N \gamma_k^{(n)}$$

Then:

$$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_k^{(n)} x^{(n)}$$

$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_k^{(n)} (x^{(n)} - \mu_k^{\text{new}})(x^{(n)} - \mu_k^{\text{new}})^\top$$

$$\pi_k^{\text{new}} = \frac{N_k}{N}$$

---

## Interpretation

| Aspect | K-Means | GMM with EM |
|--------|---------|-------------|
| Assignment | Hard (0 or 1) | Soft (probabilistic responsibilities) |
| Cluster shape | Spherical (implicitly) | Ellipsoidal (via covariance matrices) |
| Update | Mean of assigned points | Weighted mean of all points |
| Objective | Minimize within-cluster variance | Maximize log-likelihood |

If all covariances are fixed as $\Sigma_k = I$, EM for GMM reduces to a soft version of K-means.

## GMM Advantages and Disadvantages

**Advantages:**
* Soft clustering with posterior probabilities
* Flexible cluster shapes via covariance matrices
* Probabilistic density estimation
* Can handle overlapping clusters

**Disadvantages:**
* Need to choose $K$
* Sensitive to initialization (often initialize means with K-means)
* Gaussian assumption may not hold
* Risk of singular covariance matrices in high dimensions
