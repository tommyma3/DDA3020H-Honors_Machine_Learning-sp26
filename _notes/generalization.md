# Generalization

## 1. Statistical Framework: MLE vs. MAP

The choice of learning algorithm can be viewed through a probabilistic lens based on the likelihood and prior assumptions.

| Likelihood $p(y\|x,w)$ | Prior $p(w)$ | Resulting Method |
| :--- | :--- | :--- |
| Gaussian | Uniform | **Ordinary Least Squares (MLE)**  |
| Gaussian | Gaussian | **Ridge Regression (MAP)**  |
| Gaussian | Laplace | **Lasso Regression (MAP)**  |
| Laplace | Uniform | **Robust Regression ($l_1$ loss)**  |

**Key Property:** Maximum A Posteriori (MAP) estimation reduces to Maximum Likelihood Estimation (MLE) when the prior $p(w)$ is uniform.



---

## 2. Generalization Terminologies

* **Generalization Error ($E_{gen}$):** The expected loss on a new data point drawn from the population distribution $P$. Also known as *population risk* or *out-of-sample error*.



$$E_{gen}(f) = \mathbb{E}_{(x,y) \sim P}[(f(x) - y)^2]$$


* 
**Training Error:** The average loss on the finite training set $D$.



$$\hat{E}_{train}(f) = \frac{1}{m} \sum_{i=1}^m (f(x_i) - y_i)^2$$


* 
**Generalization Gap:** The difference between generalization error and training error.


* 
**Excess Risk:** Measures how much worse a learned model $f$ performs compared to the Bayes optimal predictor $f^*$.



$$Excess\ Risk = E_{gen}(f) - E_{gen}(f^*)$$



---

## 3. Excess Risk Decomposition

Excess risk can be split into three fundamental components:

1. **Approximation Error:** The gap between the best-in-class predictor $\bar{f}$ and the Bayes optimal $f^*$. It reflects the limitation of the model class $\mathcal{F}$.


2. **Estimation Error:** The gap between the empirical risk minimizer $f^*_D$ and the best-in-class predictor $\bar{f}$. It results from learning from a finite dataset.


3. 
**Optimization Error:** The gap between the approximate solution $\hat{f}_D$ returned by an algorithm and the true empirical minimizer $f^*_D$.



---

## 4. Mathematical Proof: Bias-Variance Decomposition

Assume the true target is $y = t(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Let $h_D$ be the hypothesis learned from dataset $D$, and $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ be the expected hypothesis.

**Proof Step 1: Decomposition of Expected Error at $x$**


$$\mathbb{E}_{y,D}[(h_D(x) - y)^2] = \mathbb{E}_{y,D}[(h_D(x) - \bar{h}(x) + \bar{h}(x) - y)^2]$$

$$= \mathbb{E}_{y,D}[(h_D(x) - \bar{h}(x))^2] + \mathbb{E}_{y,D}[(\bar{h}(x) - y)^2] + 2\mathbb{E}_{y,D}[(h_D(x) - \bar{h}(x))(\bar{h}(x) - y)]$$

The cross term vanishes because $\mathbb{E}_D[h_D(x) - \bar{h}(x)] = 0$.

**Proof Step 2: Decomposing the second term**


$$\mathbb{E}_y[(\bar{h}(x) - y)^2] = \mathbb{E}_y[(\bar{h}(x) - t(x) + t(x) - y)^2]$$

$$= (\bar{h}(x) - t(x))^2 + \mathbb{E}_y[(t(x) - y)^2] + 2\mathbb{E}_y[(\bar{h}(x) - t(x))(t(x) - y)]$$

The cross term vanishes because $\mathbb{E}_y[t(x) - y] = 0$.

**Final Result:**


$$\mathbb{E}_{y,D}[(h_D(x) - y)^2] = \underbrace{\mathbb{E}_D[(h_D(x) - \bar{h}(x))^2]}_{\text{Variance}} + \underbrace{(\bar{h}(x) - t(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_y[(t(x) - y)^2]}_{\text{Irreducible Noise}}$$

.

---

## 5. Underfitting, Overfitting, and Regularization

* 
**Underfitting (High Bias):** Model is too simple to capture the underlying structure. Remedied by increasing model complexity or adding informative features.


* 
**Overfitting (High Variance):** Model fits training noise. Remedied by collecting more data or reducing complexity.


* 
**Regularization:** Explicitly penalizes complexity $\Omega(f)$.


* **Ridge ($L_2$):** $\lambda \|w\|_2^2$. Reduces variance and stabilizes solutions.


* **Lasso ($L_1$):** $\lambda \|w\|_1$. Encourages sparse solutions (feature selection).





---

## 6. The "Double Descent" Phenomenon

Modern machine learning (e.g., deep learning) often operates in the **overparameterized regime** where the number of parameters $p$ exceeds the number of samples $m$.

* **Interpolation Threshold ($p = m$):** The point where the model can exactly fit the training data; classically, this is where generalization is worst.
* 
**Double Descent:** Beyond the interpolation threshold, increasing model complexity further can actually *decrease* the test error again, contradicting the classical trade-off.