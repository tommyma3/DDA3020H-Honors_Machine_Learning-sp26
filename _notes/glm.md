# Generalized Linear Models (GLM)

### 1. The Exponential Family
A probability distribution belongs to the exponential family if its density can be expressed as:
$$p(y|\eta) = h(y) \exp\left(\eta^\top T(y) - A(\eta)\right)$$
* **$\eta$ (Natural Parameter):** The parameter that defines the distribution.
* **$T(y)$ (Sufficient Statistic):** Often $T(y) = y$.
* **$A(\eta)$ (Log-Partition Function):** Ensures the distribution integrates to 1.
* **$h(y)$ (Base Measure):** A scaling factor independent of $\eta$.

### 2. Mathematical Properties
* **Mean and Variance:** * $\mathbb{E}[T(y)] = \nabla_\eta A(\eta)$.
    * $\text{Var}[T(y)] = \nabla_\eta^2 A(\eta)$.
* **Convexity:** Since the covariance matrix $\nabla_\eta^2 A(\eta)$ is positive semidefinite, $A(\eta)$ is a convex function.
* **Log-Likelihood:** For a GLM where $\eta = w^\top x$, the log-likelihood $\ell(w) = \sum_{i=1}^m [y_i w^\top x_i - A(w^\top x_i)] + \text{const}$ is always concave in $w$, making it suitable for global optimization.

### 3. GLM Design Assumptions
To derive a GLM, three assumptions are made:
1.  **Response Distribution:** $y | x$ follows an exponential family distribution.
2.  **Linear Predictor:** The natural parameter $\eta$ is a linear combination of features: $\eta = w^\top x$.
3.  **Hypothesis:** The prediction $h(x)$ is the expected value of the sufficient statistic: $h(x) = \mathbb{E}[T(y)|x]$.

### 4. Canonical Link Functions
The canonical link function $g(\mu)$ maps the mean $\mu$ to the natural parameter $\eta$, such that $g(\mu) = \eta$.
* **Gaussian (Linear Regression):** $\eta = \mu$; identity link $g(\mu) = \mu$.
* **Bernoulli (Logistic Regression):** $\eta = \log(\frac{\mu}{1-\mu})$; logit link $g(\mu) = \ln(\frac{\mu}{1-\mu})$.

