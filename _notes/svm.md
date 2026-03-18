# Support Vector Machines (SVM)


## 1. Geometric Intuition and the Margin
The goal of a Support Vector Machine is to find a hyperplane $w^\top x + b = 0$ that separates two classes with the **maximum margin**.

### 1.1 Distance to the Hyperplane
For any point $x_i$, the signed distance to the hyperplane is given by:
$$\gamma_i = \frac{w^\top x_i + b}{\|w\|}$$
To ensure the distance is always positive for correctly classified points, we define the **functional margin** as $y_i(w^\top x_i + b)$. The **geometric margin** is then:
$$\text{Margin} = \frac{y_i(w^\top x_i + b)}{\|w\|}$$

### 1.2 The Canonical Hyperplane
Since scaling $w$ and $b$ by a constant does not change the hyperplane, we can fix the functional margin of the points closest to the hyperplane (the support vectors) to be exactly 1:
$$y_i(w^\top x_i + b) = 1$$
Under this convention, the total margin between the two classes is $\frac{2}{\|w\|}$.



---

## 2. Primal Optimization Problems

### 2.1 Hard-Margin SVM (Linearly Separable)
When the data is perfectly separable, we maximize the margin by minimizing the inverse:
$$\begin{aligned} \min_{w, b} & \quad \frac{1}{2}\|w\|^2 \\ \text{s.t.} & \quad y_i(w^\top x_i + b) \ge 1, \quad i=1, \dots, m \end{aligned}$$

### 2.2 Soft-Margin SVM (Non-Separable)
To handle noise or overlapping distributions, we introduce slack variables $\xi_i \ge 0$.
$$\begin{aligned} \min_{w, b, \xi} & \quad \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i \\ \text{s.t.} & \quad y_i(w^\top x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \end{aligned}$$
* **$C$ (Hyperparameter):** Controls the trade-off between maximizing the margin and minimizing classification error. 
* **Hinge Loss Equivalence:** For a fixed $(w, b)$, the optimal $\xi_i$ is $\max(0, 1 - y_i(w^\top x_i + b))$. Thus, the objective is equivalent to:
    $$\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \text{HingeLoss}(y_i, w^\top x_i + b)$$

---

## 3. Lagrange Duality and the Dual Problem

To solve the constrained optimization efficiently and enable the use of kernels, we derive the **Wolfe Dual**.

### 3.1 The Lagrangian
For the soft-margin case, we define the Lagrangian $L$ with multipliers $\alpha_i \ge 0$ and $\mu_i \ge 0$:
$$L(w, b, \xi, \alpha, \mu) = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i - \sum_{i=1}^m \alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i] - \sum_{i=1}^m \mu_i \xi_i$$

### 3.2 Stationarity Conditions
To find the dual, we take the partial derivatives of $L$ and set them to zero:
1.  $\frac{\partial L}{\partial w} = 0 \implies w = \sum_{i=1}^m \alpha_i y_i x_i$
2.  $\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^m \alpha_i y_i = 0$
3.  $\frac{\partial L}{\partial \xi_i} = 0 \implies C - \alpha_i - \mu_i = 0$

From (3), since $\mu_i \ge 0$, we conclude that $0 \le \alpha_i \le C$.

### 3.3 The Dual Objective
Substituting the stationarity results back into the Lagrangian yields the **Dual Problem**:
$$\begin{aligned} \max_{\alpha} & \quad \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j (x_i^\top x_j) \\ \text{s.t.} & \quad \sum_{i=1}^m \alpha_i y_i = 0 \\ & \quad 0 \le \alpha_i \le C, \quad i=1, \dots, m \end{aligned}$$

---

## 4. KKT Conditions and Parameter Recovery

### 4.1 Complementary Slackness
The Karush-Kuhn-Tucker (KKT) conditions must hold at the optimum:
1.  $\alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i] = 0$
2.  $\mu_i \xi_i = (C - \alpha_i) \xi_i = 0$

### 4.2 Support Vector Roles
* **$\alpha_i = 0$:** The point $x_i$ is correctly classified and lies outside the margin. It does not affect $w$.
* **$0 < \alpha_i < C$:** From condition (2), $\xi_i = 0$. These points lie **exactly on the margin** ($y_i(w^\top x_i + b) = 1$). They are the "Free Support Vectors."
* **$\alpha_i = C$:** From condition (1), these points are "Bounded Support Vectors" and may lie inside the margin or be misclassified ($\xi_i > 0$).

### 4.3 Recovering $b^*$
For any support vector $x_j$ where $0 < \alpha_j < C$, we know $y_j(w^\top x_j + b) = 1$. Multiplying by $y_j$:
$$b^* = y_j - \sum_{i=1}^m \alpha_i y_i (x_i^\top x_j)$$
To increase numerical stability, we typically average this over the set of all such support vectors $M$:
$$b^* = \frac{1}{|M|} \sum_{j \in M} \left( y_j - \sum_{i=1}^m \alpha_i y_i (x_i^\top x_j) \right)$$

---

## 5. The Kernel Trick
The dual objective and the prediction rule depend only on the inner product $x_i^\top x_j$. This allows us to map data into a high-dimensional feature space $\phi(x)$ without explicitly computing the coordinates.



### 5.1 Kernel Function
We define a kernel $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$. Common kernels include:
* **Polynomial:** $K(x, z) = (x^\top z + c)^d$
* **Gaussian (RBF):** $K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$

### 5.2 Mercer's Theorem
A function $K(x, z)$ is a valid kernel if and only if for any finite set of points, the **Kernel Matrix** (Gram Matrix) $G_{ij} = K(x_i, x_j)$ is symmetric and positive semi-definite.

### 5.3 Prediction Rule
The final decision function for a new point $x$ becomes:
$$f(x) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b \right)$$