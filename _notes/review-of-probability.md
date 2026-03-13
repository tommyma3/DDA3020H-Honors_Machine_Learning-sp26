# Review of Probability

## Random Experiments, Sample Spaces, and Events

- **Random experiment:** a well-defined procedure whose outcome cannot be predicted with certainty in advance.
  *Example:* tossing a coin twice.

- **Sample space $S$:** the set of all possible outcomes.
  *Example:* $S = \{(H,H),\, (H,T),\, (T,H),\, (T,T)\}$.

- **Event $A$:** any subset of the sample space, $A \subseteq S$.
  *Example:* "at least one head" $= A = \{(H,H),(H,T),(T,H)\} \subseteq S$.

## Probability of Events

A probability function $P(\cdot)$ over events in $S$ satisfies:

1. **Non-negativity:** $P(A) \geq 0$
2. **Normalization:** $P(S) = 1$
3. **Additivity:**
   - If $A \cap B = \emptyset$: $P(A \cup B) = P(A) + P(B)$
   - In general: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

## Random Variables

A **random variable** is a function mapping outcomes to real values:

$$X : S \to \mathbb{R}$$

*Example (two coin tosses):* Let $X$ = number of tails.

$$X(H,H)=0,\quad X(H,T)=1,\quad X(T,H)=1,\quad X(T,T)=2.$$ 

The **state space (support)** of $X$ is $\mathcal{X} = \{0, 1, 2\}$.

Two types:
- **Discrete** random variables — countable state space.
- **Continuous** random variables — uncountable state space.

## Probability Mass Function (PMF) — Discrete

For a discrete random variable $X$ with state space $\mathcal{X}$, the **PMF** is $P(X = x)$ for $x \in \mathcal{X}$.

**Properties:**
- Non-negativity: $P(X = x) \geq 0$ for all $x \in \mathcal{X}$
- Normalization: $\displaystyle\sum_{x \in \mathcal{X}} P(X = x) = 1$

## Joint, Marginal, and Conditional Probabilities

For two discrete random variables $X$ and $Y$:

- **Joint probability:** $P(X = x,\, Y = y)$

- **Marginalization:**
$$P(X = x) = \sum_{y \in \mathcal{Y}} P(X = x,\, Y = y), \qquad P(Y = y) = \sum_{x \in \mathcal{X}} P(X = x,\, Y = y)$$

- **Conditional probability:**
$$P(X = x \mid Y = y) = \frac{P(X = x,\, Y = y)}{P(Y = y)}$$

## Bayes' Rule

$$\boxed{P(X = x \mid Y = y) = \frac{P(X = x)\, P(Y = y \mid X = x)}{\displaystyle\sum_{x' \in \mathcal{X}} P(X = x')\, P(Y = y \mid X = x')}}$$

## Independent Random Variables

$X$ and $Y$ are **independent** if:

$$P(X, Y) = P(X)\, P(Y)$$

## Expectation and Variance

For a discrete random variable $X$:

$$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x\, P(X = x)$$

$$\text{Var}(X) = \mathbb{E}\bigl[(X - \mathbb{E}[X])^2\bigr] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

$$\text{Std}(X) = \sqrt{\text{Var}(X)}$$

## Common Distributions

### Bernoulli Distribution

Models a single binary trial (e.g., one coin flip).

$$p(x \mid \mu) = \mu^x (1-\mu)^{1-x}, \quad x \in \{0, 1\}$$

where $\mu = P(X = 1)$. Mean and variance:

$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = \mu(1 - \mu)$$

### Binomial Distribution

Models the number of successes in $N$ independent Bernoulli trials.

$$\text{Bin}(m \mid N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m}$$

Mean and variance:

$$\mathbb{E}[m] = N\mu, \qquad \text{Var}(m) = N\mu(1-\mu)$$

## Continuous Random Variables

For a continuous random variable $X$, $P(X = x) = 0$ for any single point. Probabilities are described by the **probability density function (PDF)** $p_X(x)$:

$$P(a < X < b) = \int_a^b p_X(x)\, dx$$

The **cumulative distribution function (CDF)** is:

$$F_X(x) = P(X \leq x), \qquad p_X(x) = \frac{d}{dx} F_X(x)$$

## Bivariate Continuous Distributions

For joint PDF $p_{X,Y}(x,y)$ satisfying $\int_{-\infty}^\infty \int_{-\infty}^\infty p_{X,Y}(x,y)\, dx\, dy = 1$:

- **Marginals:**
$$p_X(x) = \int_{-\infty}^\infty p_{X,Y}(x,y)\, dy, \qquad p_Y(y) = \int_{-\infty}^\infty p_{X,Y}(x,y)\, dx$$

- **Conditional:**
$$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$$

- **Independence:** $X \perp Y \iff p_{X,Y}(x,y) = p_X(x)\, p_Y(y)$

## Gaussian (Normal) Distribution

The Gaussian distribution is one of the most important models in machine learning.

**Univariate:**

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean and $\sigma^2$ is the variance.

**Multivariate** ($D$-dimensional vector $\mathbf{x}$):

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\Sigma}$ is the covariance matrix.
