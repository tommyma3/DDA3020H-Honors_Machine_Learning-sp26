# Crash Course in Optimization

## Terminology

- General Form of an Optimization Problem:
  $$
  \begin{aligned}
  & \text{minimize / maximize} & & f(x) \\
  & \text{subject to} & & x \in \Omega.
  \end{aligned}
  $$

  - $x$: decision variable
  - $f(x)$: objective function
  - $\Omega$: feasible set (constraints)

- **Feasible point:** a decision variable $x \in \Omega$ that satisfies the constraints.

- **Feasible set (or feasible region) $\Omega$:** the set of all feasible points.

- **Optimal solution:** a feasible point $x^* \in \Omega$ whose objective value is at least as that of any other feasible point.

- **Optimal value:**
  $$f^* = \inf_{x \in \Omega} f(x)$$
  If there exists $x^* \in \Omega$ such that $f(x^*) = f^*$, then the optimal value is said to be attainable.

- If $\Omega = \mathbb{R}^n$ (no constraints), the problem is called **unconstrained**; otherwise, it is a **constrained optimization** problem.

- **Local minimizers.** A point $x^* \in \Omega$ is called:

  - a **local minimizer** if there exists $\varepsilon > 0$ such that $f(x) \ge f(x^*)$ for all $x \in \Omega \cap B_\varepsilon(x^*)$;
  - a **strict local minimizer** if $f(x) > f(x^*)$ for all $x \in (\Omega \cap B_\varepsilon(x^*)) \setminus \{x^*\}$.

- **Global minimizers.** A point $x^* \in \Omega$ is called:

  - a **global minimizer** if $f(x) \ge f(x^*)$ for all $x \in \Omega$;
  - a **strict global minimizer** if $f(x) > f(x^*)$ for all $x \in \Omega \setminus \{x^*\}$.

- **Stationary point (or critical point):** a point $x^*$ where the gradient of the objective function is zero, i.e., $\nabla f(x^*) = 0$.

- **Saddle point:** a stationary point that is neither a local minimum nor a local maximum.

## Optimality Conditions

### Unconstrained Optimization

- **FONC (First-Order Necessary Condition):** If $f$ is differentiable and $x^*$ is a local minimizer of $\min_{x \in \mathbf{R}^n} f(x)$, then $\nabla f(x^*) = 0$.

- **SONC (Second-Order Necessary Condition):** If $f$ is twice differentiable and $x^*$ is a local minimizer of $\min_{x \in \mathbf{R}^n} f(x)$, then $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is positive semi-definite.

- **SOSC (Second-Order Sufficient Condition):** If $f$ is twice continuously differentiable and $x^*$ satisfies $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is positive definite, then $x^*$ is a strict local minimizer.

- **Sufficient Condition for Saddle Points:** If $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is indefinite, then $x^*$ is a saddle point.

### Constrained Optimization

- Definitions:
  - **Feasible direction:** Given $x \in \Omega$, a direction $d$ is called a **feasible direction** at $x$ if $\exist \bar{t} > 0 \space \text{s.t.}  \space x + td \in \Omega, \space \forall 0 \leq t \leq \bar{t}$.

  - **Descent direction:** Let $f$ be continuously differentiable. A direction $d$ is called a **descent direction** at $x$ if $\nabla f(x)^\top d < 0$. 
    > *Note*: by Taylor's expansion, $x+td$ will decrease the function value if step size $t$ is small enough.

- **FONC (First-Order Necessary Condition):** Let $x^*$ be a local minimizer of $\min_{x \in \Omega} f(x)$. Then, for any feasible direction $d$ at $x^*$, we have $\nabla f(x^*)^\top d \geq 0$.
  - There are no feasible descent directions at a local minimum.
  - In the unconstrained case, all directions are feasible, so $\nabla f(x^*) = 0$.

- **Lagrangian**
  - Problem setup:
    $$
    \begin{aligned}
    & \min_{x \in \mathbf{R}^n} & & f(x) \\
    & \text{s.t.} & & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
    & & & h_j(x) = 0, \quad j = 1, \ldots, p
    \end{aligned}
    $$
    - Feasible set: $\Omega = \{x \in \mathbf{R}^n : g(x) \leq 0, h(x) =0\}$.
    - Active constraints: $\mathcal{A}(x) = \{i : g_i(x) = 0\}$.
    - Inactive constraints: $\mathcal{I}(x) = \{i : g_i(x) < 0\}$.
  - Lagrangian function:
    $$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \mu_j h_j(x)$$
    where $\lambda_i \geq 0$ are the Lagrange multipliers for the inequality constraints and $\mu_j$ are the Lagrange multipliers for the equality constraints.

- **KKT Conditions:** Let $x$ be a local mimimizer, and assume a constaint qualification holds. Then there exist $\lambda_i \geq 0$ and $\mu_j$ such that:
  - Stationarity: $\nabla f(x) + \sum_{i=1}^m \lambda_i \nabla g_i(x) + \sum_{j=1}^p \mu_j \nabla h_j(x) = 0$.
  - Primal feasibility: $g_i(x) \leq 0$ for all $i$, and $h_j(x) = 0$ for all $j$.
  - Dual feasibility: $\lambda_i \geq 0$ for all $i$.
  - Complementarity: $\lambda_i g_i(x) = 0$ for all $i$.

- **LICQ (Linear Independence Constraint Qualification):** The gradients of the active constraints at $x$ are linearly independent, i.e., $\{\nabla g_i(x) : i \in \mathcal{A}(x)\} \cup \{\nabla h_j(x) : j = 1, \ldots, p\}$ are linearly independent. (This is one of the constraint qualifications.)

- A feasible point satisfying the KKT conditions is called a **KKT point**.

## Convexity

- Definitions:
  - A set $\Omega$ is **convex** if for any $x, y \in \Omega$ and $\alpha \in [0,1]$, we have $\alpha x + (1-\alpha) y \in \Omega$.
  - A function $f$ is **convex** if for any $x, y$ in its domain and $\alpha \in [0,1]$, we have $f(\alpha x + (1-\alpha) y) \leq \alpha f(x) + (1-\alpha) f(y)$.
  - A function $f$ is **concave** if $-f$ is convex.
  - A function is **strictly convex** if the inequality is strict for $x \neq y$ and $\alpha \in (0,1)$.
  - A function is **strongly convex** with parameter $\mu > 0$ if
    $$f(\alpha x + (1-\alpha) y) \leq \alpha f(x) + (1-\alpha) f(y) - \frac{\mu}{2} \alpha (1-\alpha) \|x-y\|^2$$
    for all $x, y$ in its domain and $\alpha \in [0,1]$.

- **Second-Order Characterization:** Let $f$ be twice differentiable on a convex open set $\Omega$. Then $f$ is convex on $\Omega$ if and only if its Hessian $\nabla^2 f(x)$ is positive semi-definite for all $x \in \Omega$.

- **First-Order Characterization:** Let $f$ be differentiable on a convex open set $\Omega$. Then $f$ is convex on $\Omega$ if and only if $\forall x, y \in \Omega$, we have
$$f(y) \geq f(x) + \nabla f(x)^\top (y-x)$$

- **First-Order Characterizations of Convexity**
  - **Strictly Convex Function:** The function $f$ is strictly convex if and only if $\forall x \neq y \in \Omega$, we have
  $$f(y) > f(x) + \nabla f(x)^\top (y-x)$$

  - **Strongly Convex Function:** The function $f$ is strongly convex with parameter $\mu > 0$ if and only if $\forall x, y \in \Omega$, we have
  $$f(y) \geq f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2} \|y-x\|^2$$

- **Second-Order Characterizations of Convexity**
  - **Strictly Convex Function:** The function $f$ is strictly convex if $\nabla^2 f(x)$ is positive definite for all $x \in \Omega$.

  - **Strongly Convex Function:** The function $f$ is strongly convex with parameter $\mu > 0$ if and only if for all $x \in \Omega$, $$\lambda_{\min}(\nabla^2 f(x)) \geq \mu$$

- Convex Calculus:
  - **Sum Rule**: If $a_1, \ldots , a_m \geq 0$ and $f_1, \ldots, f_m$ are convex functions, then $f(x) = \sum_{i=1}^m a_i f_i(x)$ is also convex.
  - **Composition with Linear Functions**: If $f: \mathbb{R}^m \to \mathbb{R}$ is convex and $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}$, then $g(x) = f(Ax + b)$ is convex.
  - **Maximum of Convex Functions**: If $f_1, \ldots, f_m$ are convex functions, then $f(x) = \max \{f_1(x), \ldots, f_m(x)\}$ is also convex.
  - **Composition Rule**: Let $\Omega \subseteq \mathbb{R}^n$ be a convex set. Suppose $g: \Omega \to \mathbb{R}$ is a convex function and $f: \mathbb{R} \to \mathbb{R}$ is convex and non-decreasing on an interval $I \supseteq g(\Omega)$. Then the composition $f \circ g$ is convex.

- Convex Constraint
  - A constraint is called *convex* if the feasible region defined by the constraint is a convex set.
  - Sufficient conditions for convex constraints:
    - Constraints of the form $g(x) \leq 0$ with $g$ convex.
    - Linear constraints: $$Ax \leq b, \quad Ax = b, \quad Ax \geq b$$
    - Any combination of convex constraints.

- **Convex Optimization Problem:** An optimization problem is called a convex optimization problem if the objective function is *convex* or *concave* (for maximization) and the feasible set defined by the constraints is a *convex* set.
  - **Convexity and Global Solutions**: If $f$ is convex on a convex set $\Omega$, then any local minimizer of $\min_{x \in \Omega} f(x)$ is also a global minimizer. 
  - **Stationarity and Optimality**: If $f: \mathbb{R}^n \to \mathbb{R}$ is convex and differentiable, then
    $$\nabla f(x^*) = 0 \iff x^* \text{ is a global minimizer.}$$
    
    - *Remarks*:
        - For unconstraint convex problems, FONC is both necessary and sufficient for global optimality.
        - Finding stationary points is enough to find global solutions.
  
  - **Global Optimality of KKT Points**: 
    
    Consider the convex optimization problem 
    $$\begin{aligned}
    & \min_{x \in \mathbf{R}^n} & & f(x) \\
    & \text{s.t.} & & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
    & & & h_j(x) = 0, \quad j = 1, \ldots, p
    \end{aligned}$$
    where $f$ and $g_i$ are convex functions, and $h_j$ are linear. Then any KKT point is a global minimizer.

    - *Remarks*:
      - No constraint qualification is needed for this theorem.

## Algorithms

### Gradient Descent

Consider the unconstrained optimization problem $\min_{x \in \mathbb{R}^n} f(x)$.

**Gradient descent iteration:**
$$x^{k+1} = x^k - \alpha_k \nabla f(x^k)$$

- The update direction $d^k = -\nabla f(x^k)$ is a descent direction, since $\nabla f(x^k)^\top d^k = -\|\nabla f(x^k)\|^2 < 0$ for $\nabla f(x^k) \neq 0$.

- The step size $\alpha_k$ can be chosen by various methods, such as:
  - **Constant step size**: $\alpha_k = \alpha$ for some fixed $\alpha > 0$.
  - **Diminishing step size**: $\alpha_k \to 0$ as $k \to \infty$, e.g., $\alpha_k = \frac{1}{k}$.
  - **Line search**: choose $\alpha_k$ to minimize $f(x^k - \alpha \nabla f(x^k))$ over $\alpha > 0$.

  - **Adaptive step size**: methods like AdaGrad, RMSProp, Adam, etc., which adjust the step size based on past gradients.

- A commonly used stopping criterion is $\|\nabla f(x^k)\| \leq \varepsilon$, where $\varepsilon > 0$ is a prescribed tolerance level.

**Theorem: Gradient Descent for Strongly Convex Functions**

Assume that $f$ is $\mu$-strongly convex and has $L$-Lipschitz continuous gradients ($\|\nabla f(x)-\nabla f(y) \leq L \|x-y\|$). Using a constant step size $\bar{\alpha} = \frac{2}{L+\mu}$, the gradient descent iterates satisfy
$$\|x^{k+1} - x^*\| \leq \frac{L-\mu}{L+\mu} \|x^k - x^*\|$$
where $x^*$ is the unique global minimizer of $f$.

Consequently, after $K$ iterations, 
$$\|x^K - x^*\| \leq \left(\frac{L-\mu}{L+\mu}\right)^K \|x^0 - x^*\|$$
$$f(x^K) - f(x^*) \leq \frac{L}{2} \|x^K - x^*\|^2 \leq \frac{L}{2} \left(\frac{L-\mu}{L+\mu}\right)^{2K} \|x^0 - x^*\|^2$$

> *Note*: Number of iterations to achieve $\varepsilon$-accuracy (i.e., $\|x - x^*\| \leq \varepsilon$) will be $O\left(\frac{L}{\mu} \log \frac{1}{\varepsilon}\right)$.

- **Convex and Smooth Case**

    Let $f$ be convex and have $L$-Lipschitz continuous gradients. Using a constant step size $\bar{\alpha} = \frac{1}{L}$, the gradient descent iterates satisfy
    $$f(x^K) - f(x^*) \leq \frac{2L}{K} \|x^0 - x^*\|^2$$

    - $O(\frac{1}{K})$ convergence rate
    - To find $x$ such that $f(x) - f(x^*) \leq \varepsilon$, it is enough to run $K=O\left(\frac{1}{\varepsilon} \right)$ iterations.

- **Nonconvex and Smooth Case**

    Let $f$ have $L$-Lipschitz continuous gradients (not necessarily convex). With the same step size $\bar{\alpha} = \frac{1}{L}$, 
    $$\frac{1}{K} \sum_{k=0}^{K-1} \|\nabla f(x^k)\|^2 \leq \frac{2L}{K} (f(x^0) - f^*)$$
    where $f^* = \inf_{x \in \mathbb{R}^n} f(x)$. Consequently, 
    $$\min_{k=0, \ldots, K-1} \|\nabla f(x^k)\|^2 \leq \frac{2L}{K} (f(x^0) - f^*)$$

**Constrained Optimization**

Consider the constrained optimization problem $\min_{x \in \Omega} f(x)$, where $\Omega$ is a convex set. The **projected gradient descent** iteration is given by
$$x^{k+1} = P_\Omega(x^k - \alpha_k \nabla f(x^k))$$
where $P_\Omega(z) = \arg\min_{x \in \Omega} \frac{1}{2}\|x-z\|$ is the projection of $z$ onto $\Omega$.


### Stochastic Gradient Descent (SGD)

**Stochastic Optimization**

Stochastic optimzation problem is formulated as $$\min_{x \in \Omega} f(x) = \mathbb{E}_{\xi}[F(x; \xi)]$$
where $\xi$ is a random variable and $F(x; \xi)$ is the sample-wise loss or cost.\
    *Remark*: If $F(\cdot, \xi)$ is convex for every $\xi$, then $f$ is also convex.

- **Stochastic gradient oracle**: Instead of $\nabla f(x)$, we observe a random vector $g(x, \xi)$ such that $\mathbb{E}_\xi[g(x, \xi)] = \nabla f(x)$.

- **Goal**: Use noisy gradient samples to minimze $f$.

**Stochastic Gradient Descent (SGD)**

Given a stochastic gradient estimate $g(x^k, \xi^k)$ at iteration $k$, the SGD uodate is 
$$x^{k+1} = x^k - \lambda_k g(x^k, \xi^k)$$

- $g(x^k, \xi^k)$: one-sample estimate of the gradient $\nabla f(x^k)$.
- $\lambda_k$: step size (learning rate) at iteration $k$.
- $\xi^k$: i.i.d. random sample.

**Theorem**

- *Assumptions on the Stochastic Gradient Oracle*:
  - Unbiasedness: $\mathbb{E}_\xi[g(x, \xi)] = \nabla f(x)$ for all $x$.
  - Bounded variance: $\mathbb{E}_\xi[\|g(x, \xi) - \nabla f(x)\|^2] \leq \sigma^2$ for all $x$.

- **Strongly Convex Case**: Assume $f$ is strongly convec and the stochastic gradient oracle satisfies the above assumptions. Choose step size $\lambda_k = \frac{\theta}{k+1}, \theta > \frac{1}{2\mu}$. Then, 
  $$\mathbb{E}[\|x^k - x^*\|^2] \leq \frac{c_\theta}{k+1}$$
  where $c_\theta = \max\left\{\frac{2\theta^2 \sigma^2}{2\mu \theta - 1}, \|x^0 - x^*\|^2\right\}$.

  - Convergence rate: $O\left(\frac{1}{k}\right)$.
  - With diminishing step size, SGD converges to the optimal solution exactly.


### Example: GD vs SGD

Consider the objective function $f(x)$ defined as the sum of $n$ component functions:


$$f(x) = \sum_{i=1}^{n} f_i(x)$$

Assumptions:
* **Smoothness & Convexity:** Each $f_i$ is strongly convex and has a Lipschitz continuous gradient.
* **Oracle Cost:** The computation of a single gradient $\nabla f_i(x)$ has a cost of $O(1)$.

**GD**:
updates the parameters using the full gradient at each step:


$$x^{k+1} = x^{k} - \alpha \nabla f(x^k) = x^{k} - \alpha \sum_{i=1}^{n} \nabla f_i(x^k)$$

* **Iteration Complexity:** $O(\log \frac{1}{\epsilon})$ (due to linear convergence in the strongly convex case).
* **Per-iteration Cost:** $O(n)$ (must compute all $n$ gradients).
* **Total Cost:** $O(n \log \frac{1}{\epsilon})$.

**SGD**:
picks an index $i_k$ uniformly at random from $\{1, 2, \dots, n\}$ and approximates the gradient:


$$x^{k+1} = x^k - \alpha \left[ n \nabla f_{i_k}(x^k) \right]$$

The estimator is unbiased because:


$$\mathbb{E}[n \nabla f_{i_k}(x^k)] = n \cdot \left[ \frac{1}{n} \nabla f_1(x^k) + \dots + \frac{1}{n} \nabla f_n(x^k) \right] = \nabla f(x^k)$$

* **Iteration Complexity:** $O(\frac{1}{\epsilon})$ (slower sublinear convergence due to gradient noise).
* **Per-iteration Cost:** $O(1)$ (only one gradient computed).
* **Total Cost:** $O(\frac{1}{\epsilon})$.

**Decision**:
To choose between the two, we compare the total costs:


$$\text{GD: } O(n \log \tfrac{1}{\epsilon}) \quad \text{vs.} \quad \text{SGD: } O(\tfrac{1}{\epsilon})$$

> **Conclusion:** We prefer **SGD** when the number of samples $n$ is very large relative to the desired precision, specifically when $n \gg \frac{1}{\epsilon}$.