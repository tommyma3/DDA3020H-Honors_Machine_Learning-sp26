# Introduction to Machine Learning


---

## Outline

1. [What Is Machine Learning?](#1-what-is-machine-learning)
2. [Supervised Learning](#2-supervised-learning)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Basic Concepts in Machine Learning](#4-basic-concepts-in-machine-learning)
5. [Review of Probability](#5-review-of-probability)

---

## 1. What Is Machine Learning?

### Definitions

**Arthur Samuel (1959):**

> "Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed."

**Tom Mitchell (1997):**

> "A computer program is said to learn from **experience E** with respect to a class of **tasks T** and a **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E."

### Example: Spam Filter

| Mitchell's Component | Spam Filter |
|----------------------|-------------|
| **Task T** | Classify emails as spam or not spam |
| **Experience E** | Your labels indicating which emails are spam or not |
| **Performance P** | Fraction of emails correctly classified |

### ML and Artificial Intelligence

- **Artificial Intelligence (AI):** The study and development of machines or systems that can perform tasks requiring human-like intelligence — perception, reasoning, learning, decision making.
- **Machine Learning:** A subfield of AI that focuses on algorithms enabling systems to **learn from data** and improve performance through experience.

ML is interdisciplinary, drawing from statistics, mathematics, computer science, and domain-specific fields.

### Two Basic Paradigms

| Paradigm | Data | Goal | Typical Tasks |
|----------|------|------|---------------|
| **Supervised Learning** | Input–label pairs $\{(x_i, y_i)\}_{i=1}^n$ | Learn a mapping $x \mapsto y$ | Classification, Regression |
| **Unsupervised Learning** | Unlabeled feature vectors $\{x_i\}_{i=1}^n$ | Discover underlying structure | Clustering, Dimensionality Reduction |

---

## 2. Supervised Learning

### Setup

In supervised learning, the dataset consists of $N$ labeled examples:

$$\{(x_i, y_i)\}_{i=1}^N$$

Each input $x_i \in \mathbb{R}^D$ is a **feature vector** with $D$ dimensions:

$$x_i = \bigl[x_i^{(1)}, \ldots, x_i^{(j)}, \ldots, x_i^{(D)}\bigr]^\top, \quad i = 1, \ldots, N.$$

The label $y_i$ indicates the desired output:

- **Classification:** $y_i \in \{1, 2, \ldots, C\}$ (discrete class)
- **Regression:** $y_i \in \mathbb{R}$ (continuous value)

### How Supervised Learning Works

1. **Data collection:** obtain labeled data $\{(x_i, y_i)\}_{i=1}^N$.
2. **Training:** learn model parameters from the training data.
3. **Inference (testing):** use the trained model to predict the output for unseen input $x$.

**Goal:** learn a function that **generalizes well** to unseen data.

### Regression vs. Classification

| Type | Output | Example |
|------|--------|---------|
| **Regression** | Continuous value $y \in \mathbb{R}$ | Predicting diamond price from its mass |
| **Classification** | Discrete label $y \in \{1, \ldots, C\}$ | Classifying tumor as malignant/benign |

**Quick check:** 
- *Predicting how many items will sell over the next 3 months* → **Regression** (continuous output).
- *Deciding whether a customer account has been compromised* → **Classification** (discrete output).

---

## 3. Unsupervised Learning

### Setup

In unsupervised learning, the dataset consists of $n$ **unlabeled** examples:

$$\{x_i\}_{i=1}^n$$

Each $x_i$ is a feature vector. The goal is to learn a model that captures **useful structure or patterns** in the data without access to labels.

### What Unsupervised Models Learn

- **Clustering:** group similar data points together.
- **Dimensionality reduction:** transform data into a new, lower-dimensional representation.

**Main purpose:** understand data structure and support future inference or decision making.

### Clustering

- **Task:** partition unlabeled data points into groups (clusters).
- **Performance criteria:**
  - Points within the same cluster are *close* to each other.
  - Points from different clusters are *far* from each other.
  - Clusters provide reasonable coverage of the data.
- **Key question:** how do we define and measure "close" and "far"? (Answered in later lectures.)

### Dimensionality Reduction

- **Task:** map high-dimensional data $x \in \mathbb{R}^D$ to a lower-dimensional representation $z \in \mathbb{R}^d$, where $d \ll D$.
- **Purposes:**
  - *Data simplification:* nonlinear structure → simpler representation.
  - *Data visualization:* high-dimensional data → 2D or 3D.
  - *Feature selection / representation learning:* reduce redundancy for downstream prediction.

### Supervised vs. Unsupervised Learning

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Labels | Required | Not required |
| Goal | Learn a mapping $x \mapsto y$ | Discover structure in $x$ |
| Tasks | Classification, Regression | Clustering, Dim. Reduction |
| Evaluation | Accuracy on labeled test set | Cluster quality, reconstruction error |

---

## 4. Basic Concepts in Machine Learning

### Data and Assumptions

**For supervised learning:**

- **Training set:** $\mathcal{D}_\text{train} = \{(x_i, y_i)\}_{i=1}^n$, where $x_i \in \mathcal{X}$ (input features) and $y_i \in \mathcal{Y}$ (target labels).
- **Test set:** $\mathcal{D}_\text{test} = \{(x_i, y_i)\}_{i=1}^m$, used to evaluate the trained model.

**For unsupervised learning:**

- **Training set:** $\mathcal{D}_\text{train} = \{x_i\}_{i=1}^n$, unlabeled feature vectors.
- **Test set:** $\mathcal{D}_\text{test} = \{x_i\}_{i=1}^m$, used to assess learned structure or representations.

**I.I.D. Assumption:** Samples are assumed to be drawn **independently and identically distributed (i.i.d.)** from the same underlying distribution. Training and test sets follow the same distribution.

### Target Function, Hypothesis, and Hypothesis Space

| Concept | Definition |
|---------|-----------|
| **Target function** $t : \mathcal{X} \to \mathcal{Y}$ | The unknown ground-truth mapping that generates the data: $y_i = t(x_i) + \text{noise}$. We only observe finite data, not this function. |
| **Hypothesis** $h$ | A candidate function used to approximate $t$. Example: $h(x) = \theta_1 x_1 + \theta_2 x_2 = \boldsymbol{\theta}^\top x$. |
| **Hypothesis space** $\mathcal{H}$ | The set of all allowable hypotheses. Example: $\mathcal{H} = \{h_{\mathbf{w}}(x) = \mathbf{w}^\top x : \mathbf{w} \in \mathbb{R}^n\}$. |

### Cost Function, Training, and Testing

**Cost (objective) function:** measures how well a hypothesis fits the training data. Using squared loss as an example:

$$J(h) = \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_\text{train}} \bigl(h(x_i) - y_i\bigr)^2$$

**Training (learning):** find the best hypothesis in $\mathcal{H}$ by minimizing the cost on the training set:

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

## 5. Review of Probability

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

**Medical diagnosis example:**  
Let $x = 1$ denote a positive test result, $y = 1$ denote having the disease.

Given:
- $P(x=1 \mid y=1) = 0.8$ (true positive rate)
- $P(x=1 \mid y=0) = 0.1$ (false positive rate)
- $P(y=1) = 0.13$ (disease prevalence)

**Question:** What is $P(y=1 \mid x=1)$?

$$P(y=1 \mid x=1) = \frac{0.8 \times 0.13}{0.8 \times 0.13 + 0.1 \times 0.87} \approx 0.54$$

Even with a positive test, there is only about a 54% chance of actually having the disease — because the disease is relatively rare.

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

---

## Summary

| Topic | Key Takeaways |
|-------|--------------|
| **What is ML?** | Learning from experience; T–E–P framework (Mitchell). Subfield of AI. |
| **Supervised learning** | Labeled data, learn $x \mapsto y$. Regression (continuous) vs. Classification (discrete). |
| **Unsupervised learning** | Unlabeled data, discover structure (clustering, dim. reduction). |
| **Core concepts** | Target function, hypothesis space, cost function, training, testing. |
| **Probability** | PMF, PDF, Bayes' rule, independence, Gaussian distribution. |

**Next lecture:** Review of Optimization.
