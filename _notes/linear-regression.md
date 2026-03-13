# Linear Regression

## 1. Mathematical Fundamentals & Modeling

Linear regression models the relationship between labeled examples $\{(x_i, y_i)\}_{i=1}^{m}$.

* **Feature Vectors**: $x_i = [1, x_{i1}, \dots, x_{id}]^\top \in \mathbb{R}^{d+1}$.


* **Hypothesis Function**: $f_w(x_i) = x_i^\top w$, where $w = [w_0, \dots, w_d]^\top$.


* **Design Matrix ($X$)**: Stacks inputs into $X \in \mathbb{R}^{m \times (d+1)}$.


* **Deterministic Perspective**: Minimizes the Residual Sum of Squares (RSS), $J(w) = \frac{1}{2} \|Xw - y\|^2$.


* **Probabilistic Perspective**: Assumes noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Maximum Likelihood Estimation (MLE) of $w$ with Gaussian noise is equivalent to minimizing RSS.



---

## 2. Optimization Algorithms

### Gradient Descent (GD)

* **Update Rule**: $w \leftarrow w - \alpha \nabla J(w)$, where $\nabla J(w) = X^\top (Xw - y)$.


* **Convergence**: For strongly convex functions with Lipschitz constant $L$ and parameter $\mu$, using step size $\bar{\alpha} = \frac{2}{L+\mu}$ yields an iteration complexity of $O(\log \frac{1}{\epsilon})$.



### Stochastic Gradient Descent (SGD)

* **Mechanism**: Updates parameters using a single randomly chosen sample $(x_i, y_i)$.


* **Gradient Estimate**: $\nabla J_i(w) = (x_i^\top w - y_i)x_i$.


* **Advantage**: Efficient for large datasets where computing the full gradient is expensive.



Comparison of Methods 

| Method | Solution Type | Complexity | Best Use Case |
| --- | --- | --- | --- |
| **Analytical** | $w^* = (X^\top X)^{-1} X^\top y$ | $O(md^2 + d^3)$ | Moderate $m$ and $d$; exact solution needed. |
| **GD** | Iterative Approx. | $O(md \log \frac{1}{\epsilon})$ | Large $d$ but moderate $m$. |
| **SGD** | Iterative Approx. | $O(d \cdot \frac{1}{\epsilon})$ | Both $m$ and $d$ are very large. |

---

## 3. Advanced Modeling Techniques

### Multiple Outputs & Classification

* **Multiple Outputs**: For vector-valued $y_i \in \mathbb{R}^h$, the objective is $\min J(W) = \text{trace}((XW - Y)^\top (XW - Y))$. This is equivalent to solving $h$ independent regression problems.


* **Binary Classification**: Uses $y_i \in \{-1, +1\}$ and a prediction rule $\hat{y} = \text{sgn}(x_{new}^\top \hat{w})$.


* **Multi-Category Classification**: Uses **one-hot encoding** for labels $y_i \in \mathbb{R}^C$. Prediction follows $\hat{y} = \arg \max_{c} x_{new}^\top W_c$.



### Polynomial Regression

* **Feature Mapping**: Maps data to a higher-dimensional space $\phi(x)$ to handle non-linearly separable data (e.g., the XOR problem).


* **Expanded Matrix**: Linear regression is applied to the expanded design matrix $P(X) = [\phi(x_1)^\top; \dots; [cite_start]\phi(x_m)^\top]$.



---

## 4. Regularization (MAP Estimation)

Regularization adds a penalty term to the loss to avoid overfitting and ensure $X^\top X$ is invertible.

* **Ridge Regression ($L_2$ Penalty)**:
    * **Objective**: $\min_w \|Xw - y\|^2 + \lambda \|w\|_2^2$.


    * **Prior**: Equivalent to MAP with a **Gaussian prior** $\mathcal{N}(0, \tau^2 I)$.


    * **Solution**: $\hat{w} = (X^\top X + \lambda I)^{-1} X^\top y$.




* **Lasso Regression ($L_1$ Penalty)**:
    * **Objective**: $\min_w \|Xw - y\|^2 + \alpha \|w\|_1$.


    * **Prior**: Equivalent to MAP with a **Laplacian prior**.


    * **Property**: Encourages **sparsity** in weights.





---

## 5. Robust Linear Regression

* **The Problem**: Least squares ($L_2$ loss) is highly sensitive to outliers because errors are squared.


* **The Solution**: Replace $L_2$ loss with **$L_1$ loss** ($J(w) = \sum |x_i^\top w - y_i|$).


* **Probabilistic View**: $L_1$ regression is the MLE solution when observation noise follows a **Laplacian distribution** rather than a Gaussian one.