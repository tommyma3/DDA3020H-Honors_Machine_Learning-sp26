# Introduction to Machine Learning



## Basic Concepts


### Two Basic Paradigms

| Paradigm | Data | Goal | Typical Tasks |
|----------|------|------|---------------|
| **Supervised Learning** | Input–label pairs $\{(x_i, y_i)\}_{i=1}^n$ | Learn a mapping $x \mapsto y$ | Classification, Regression |
| **Unsupervised Learning** | Unlabeled feature vectors $\{x_i\}_{i=1}^n$ | Discover underlying structure | Clustering, Dimensionality Reduction |



### Target Function, Hypothesis, and Hypothesis Space

| Concept | Definition |
|---------|-----------|
| **Target function** $t : \mathcal{X} \to \mathcal{Y}$ | The unknown ground-truth mapping that generates the data: $y_i = t(x_i) + \text{noise}$. We only observe finite data, not this function. |
| **Hypothesis** $h$ | A candidate function used to approximate $t$. Example: $h(x) = \theta_1 x_1 + \theta_2 x_2 = \boldsymbol{\theta}^\top x$. |
| **Hypothesis space** $\mathcal{H}$ | The set of all allowable hypotheses. Example: $\mathcal{H} = \{h_{\mathbf{w}}(x) = \mathbf{w}^\top x : \mathbf{w} \in \mathbb{R}^n\}$. |

### Cost Function, Training, and Testing

**Cost (objective) function:** measures how well a hypothesis fits the training data. Using squared loss as an example:

$$J(h) = \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_\text{train}} \bigl(h(x_i) - y_i\bigr)^2$$

**Training (learning):** find the best hypothesis in $\mathcal{H}$ by minimizing the cost on the training set (optimization):

$$h^* = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_\text{train}} \bigl(h(x_i) - y_i\bigr)^2$$

**Testing (evaluation):** assess the performance of the learned model $h^*$ on the test set:

$$\text{Test error} = \frac{1}{m} \sum_{(x_i, y_i) \in \mathcal{D}_\text{test}} \bigl(h^*(x_i) - y_i\bigr)^2$$

### General Machine Learning Workflow

1. **Data collection:** gather relevant historical data.
2. **Data preprocessing:** handle missing values, outliers, and data quality issues.
3. **Model design:** choose the hypothesis space $\mathcal{H}$, objective function $J$, and optimization method.
4. **Training:** learn model parameters from training data.
5. **Testing:** evaluate performance on unseen test data.
6. **Model improvement:** refine the model, features, or data to improve performance.

### Other Learning Paradigms

| Paradigm | Description |
|----------|-------------|
| **Reinforcement Learning (RL)** | An agent interacts with an environment over time, taking actions to maximize expected cumulative reward. At each step $t$: observe state $s_t$, take action $a_t$, receive reward $r_t$, transition to $s_{t+1}$. |
| **Semi-supervised learning** | Learning from a mixture of labeled and unlabeled data, typically when labeled data is scarce. |
| **Ensemble learning** | Combining predictions from multiple models to obtain a final decision, often improving robustness and accuracy. |
| **Transfer learning** | Leveraging knowledge learned from a source domain to improve learning in a target domain, especially when target data is limited. |
| **Federated learning** | Training models across distributed clients using local data, sharing only model updates (not raw data) — privacy-preserving. |
| **Machine unlearning** | Removing the influence of specific training samples from a trained model, motivated by privacy and regulatory requirements. |

---

## Review of Probability

### Random Experiments, Sample Spaces, and Events

- **Random experiment:** a well-defined procedure whose outcome cannot be predicted with certainty in advance.  
  *Example:* tossing a coin twice.

- **Sample space $S$:** the set of all possible outcomes.  
  *Example:* $S = \{(H,H),\, (H,T),\, (T,H),\, (T,T)\}$.

- **Event $A$:** any subset of the sample space, $A \subseteq S$.  
  *Example:* "at least one head" $= A = \{(H,H),(H,T),(T,H)\} \subseteq S$.

### Probability of Events

A probability function $P(\cdot)$ over events in $S$ satisfies:

1. **Non-negativity:** $P(A) \geq 0$
2. **Normalization:** $P(S) = 1$
3. **Additivity:**
   - If $A \cap B = \emptyset$: $\quad P(A \cup B) = P(A) + P(B)$
   - In general: $\quad P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### Random Variables

A **random variable** is a function mapping outcomes to real values:

$$X : S \to \mathbb{R}$$

*Example (two coin tosses):* Let $X$ = number of tails.

$$X(H,H)=0,\quad X(H,T)=1,\quad X(T,H)=1,\quad X(T,T)=2.$$

The **state space (support)** of $X$ is $\mathcal{X} = \{0, 1, 2\}$.

Two types:
- **Discrete** random variables — countable state space.
- **Continuous** random variables — uncountable state space.

### Probability Mass Function (PMF) — Discrete

For a discrete random variable $X$ with state space $\mathcal{X}$, the **PMF** is $P(X = x)$ for $x \in \mathcal{X}$.

**Properties:**
- Non-negativity: $P(X = x) \geq 0$ for all $x \in \mathcal{X}$
- Normalization: $\displaystyle\sum_{x \in \mathcal{X}} P(X = x) = 1$

### Joint, Marginal, and Conditional Probabilities

For two discrete random variables $X$ and $Y$:

- **Joint probability:** $P(X = x,\, Y = y)$

- **Marginalization:**
$$P(X = x) = \sum_{y \in \mathcal{Y}} P(X = x,\, Y = y), \qquad P(Y = y) = \sum_{x \in \mathcal{X}} P(X = x,\, Y = y)$$

- **Conditional probability:**
$$P(X = x \mid Y = y) = \frac{P(X = x,\, Y = y)}{P(Y = y)}$$

### Bayes' Rule

$$\boxed{P(X = x \mid Y = y) = \frac{P(X = x)\, P(Y = y \mid X = x)}{\displaystyle\sum_{x' \in \mathcal{X}} P(X = x')\, P(Y = y \mid X = x')}}$$



### Independent Random Variables

$X$ and $Y$ are **independent** if:

$$P(X, Y) = P(X)\, P(Y)$$

### Expectation and Variance

For a discrete random variable $X$:

$$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x\, P(X = x)$$

$$\text{Var}(X) = \mathbb{E}\bigl[(X - \mathbb{E}[X])^2\bigr] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

$$\text{Std}(X) = \sqrt{\text{Var}(X)}$$

### Common Distributions

#### Bernoulli Distribution

Models a single binary trial (e.g., one coin flip).

$$p(x \mid \mu) = \mu^x (1-\mu)^{1-x}, \quad x \in \{0, 1\}$$

where $\mu = P(X = 1)$. Mean and variance:

$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = \mu(1 - \mu)$$

#### Binomial Distribution

Models the number of successes in $N$ independent Bernoulli trials.

$$\text{Bin}(m \mid N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m}$$

Mean and variance:

$$\mathbb{E}[m] = N\mu, \qquad \text{Var}(m) = N\mu(1-\mu)$$

### Continuous Random Variables

For a continuous random variable $X$, $P(X = x) = 0$ for any single point. Probabilities are described by the **probability density function (PDF)** $p_X(x)$:

$$P(a < X < b) = \int_a^b p_X(x)\, dx$$

The **cumulative distribution function (CDF)** is:

$$F_X(x) = P(X \leq x), \qquad p_X(x) = \frac{d}{dx} F_X(x)$$

### Bivariate Continuous Distributions

For joint PDF $p_{X,Y}(x,y)$ satisfying $\int_{-\infty}^\infty \int_{-\infty}^\infty p_{X,Y}(x,y)\, dx\, dy = 1$:

- **Marginals:**
$$p_X(x) = \int_{-\infty}^\infty p_{X,Y}(x,y)\, dy, \qquad p_Y(y) = \int_{-\infty}^\infty p_{X,Y}(x,y)\, dx$$

- **Conditional:**
$$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$$

- **Independence:** $X \perp Y \iff p_{X,Y}(x,y) = p_X(x)\, p_Y(y)$

### Gaussian (Normal) Distribution

The Gaussian distribution is one of the most important models in machine learning.

**Univariate:**

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean and $\sigma^2$ is the variance.

**Multivariate** ($D$-dimensional vector $\mathbf{x}$):

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\Sigma}$ is the covariance matrix.


## Crash Course in Optimization

### Terminology

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

- **Optimal value:** $$f^* = \inf_{x \in \Omega} f(x) $$
  If there exists $x^* \in \Omega$ such that $f(x^*) = f^*$, then the optimal value is sait to be attainable.

- If $\Omega = \mathbb{R}^n$ (no constraints), the problem is called **unconstrained**; otherwise, it is a **constrained optimization** problem.



- **Local minimizers.** A point $x^* \in \Omega$ is called:

  - a **local minimizer** if there exists $\varepsilon > 0$ such that $f(x) \ge f(x^*)$ for all $x \in \Omega \cap B_\varepsilon(x^*)$;
  - a **strict local minimizer** if $f(x) > f(x^*)$ for all $x \in (\Omega \cap B_\varepsilon(x^*)) \setminus \{x^*\}$.

- **Global minimizers.** A point $x^* \in \Omega$ is called:

  - a **global minimizer** if $f(x) \ge f(x^*)$ for all $x \in \Omega$;
  - a **strict global minimizer** if $f(x) > f(x^*)$ for all $x \in \Omega \setminus \{x^*\}$.

- **Stationary point (or critical point):** a point $x^*$ where the gradient of the objective function is zero, i.e., $\nabla f(x^*) = 0$.

- **Saddle point:** a stationary point that is neither a local minimum nor a local maximum. 

### Optimality Conditions

#### Unconstrained Optimization

- **FONC (First-Order Necessary Condition):** If $f$ is differentiable and $x^*$ is a local minimizer of $\min_{x \in \mathbf{R}^n} f(x)$, then $\nabla f(x^*) = 0$.

- **SONC (Second-Order Necessary Condition):** If $f$ is twice differentiable and $x^*$ is a local minimizer of $\min_{x \in \mathbf{R}^n} f(x)$, then $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is positive semi-definite.

- **SOSC (Second-Order Sufficient Condition):** If $f$ is twice continuously differentiable and $x^*$ satisfies $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is positive definite, then $x^*$ is a strict local minimizer.

- **Sufficient Condition for Saddle Points:** If $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*)$ is indefinite, then $x^*$ is a saddle point.

#### Constrained Optimization

- Definitions:
  - **Feasible direction:** Given $x \in \Omega$, a direction $d$ is called a **feasible direction** at $x$ if $\exist \bar{t} > 0 \space \text{s.t.}  \space x + td \in \Omega, \space \forall 0 \leq t \leq \bar{t}$.

  - **Descent direction:** Let $f$ be continuously differentiable. A direction $d$ is called a **descent direction** at $x$ if $\nabla f(x)^\top d < 0$. (Note: by Taylor's expansion, $x+td$ will decrease the function value if step size $t$ is small enough).

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

#### Convexity

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