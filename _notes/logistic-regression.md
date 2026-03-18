
# Logistic Regression

## 1. Generalization Error

**Definition**: The generalization error of a model $f$ is the expected loss on new data:

$$E_{\text{gen}}(f) = \mathbb{E}_{(x,y)\sim P}[(f(x) - y)^2]$$

This measures the model's true predictive performance (also called population risk, true risk, or out-of-sample error).

## 2. Binary Classification Setup

For binary classification with labels $y \in \{0, 1\}$:
- $y = 0$: negative class (e.g., not spam, benign)
- $y = 1$: positive class (e.g., spam, malignant)

We need a model that outputs probabilities in $[0, 1]$ rather than arbitrary real numbers.

## 3. Logistic Regression Model

**Sigmoid Function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Properties:
- $\sigma(z) \in (0, 1)$
- $\sigma(0) = 0.5$
- As $z \to \infty$, $\sigma(z) \to 1$
- As $z \to -\infty$, $\sigma(z) \to 0$

**Model Hypothesis**:
$$f_w(x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}$$

where $w \in \mathbb{R}^{d+1}$ are the parameters and $x \in \mathbb{R}^{d+1}$ is the input feature vector (with $x_0 = 1$ for bias term).

## 4. Maximum Likelihood Estimation (MLE) Interpretation

**Assumptions**:
- $\mu(x; w) = \sigma(w^\top x)$
- $y|x; w \sim \text{Bernoulli}(\mu(x; w))$

**Probability Mass Function**:
$$P(y|x; w) = \begin{cases} \mu(x; w) & \text{if } y = 1 \\ 1 - \mu(x; w) & \text{if } y = 0 \end{cases}$$

This can be written compactly as:
$$P(y|x; w) = \mu(x; w)^y \cdot (1 - \mu(x; w))^{1-y}$$

**Log-Likelihood** for training set $\{(x_i, y_i)\}_{i=1}^m$:
$$L(w) = \sum_{i=1}^m [y_i \log(\mu_i) + (1-y_i) \log(1-\mu_i)]$$
where $\mu_i = \sigma(w^\top x_i)$

## 5. Cross-Entropy Loss

**Per-example Loss**:
$$\text{cost}(y_i, f_w(x_i)) = \begin{cases} -\log(f_w(x_i)) & \text{if } y_i = 1 \\ -\log(1 - f_w(x_i)) & \text{if } y_i = 0 \end{cases}$$

This is the **cross-entropy** between two Bernoulli distributions:
- Ground-truth: $P(y=1|x_i, y_i) = y_i$
- Predicted: $P(y=1|x_i; w) = f_w(x_i)$

**Cross-Entropy Formula**:
$$H(P, Q) = -[P(y=1) \log Q(y=1) + P(y=0) \log Q(y=0)]$$
$$= -[y_i \log(f_w(x_i)) + (1-y_i) \log(1 - f_w(x_i))]$$

## 6. Training Objective

**Minimize** (average) cross-entropy loss:
$$J(w) = \frac{1}{m} \sum_{i=1}^m [y_i \log(f_w(x_i)) + (1-y_i) \log(1 - f_w(x_i))]$$

Or equivalently, maximize the log-likelihood.

## 7. Gradient Descent for Logistic Regression

**Gradient**:
$$\nabla_w J(w) = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i$$

**Proof**: Let's derive this step by step.

Given $J(w) = -\frac{1}{m} \sum_i [y_i \log(\sigma_i) + (1-y_i) \log(1-\sigma_i)]$ where $\sigma_i = \sigma(w^\top x_i)$

First, note that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (property of sigmoid)

For a single example $(x, y)$ with $\sigma = \sigma(w^\top x)$:
$$\begin{align*}\frac{\partial}{\partial w}[y \log(\sigma) + (1-y) \log(1-\sigma)] &= y \cdot \frac{1}{\sigma} \cdot \frac{\partial \sigma}{\partial w} + (1-y) \cdot \frac{1}{1-\sigma} \cdot \left(-\frac{\partial \sigma}{\partial w}\right) \\ &= \left[\frac{y}{\sigma} - \frac{1-y}{1-\sigma}\right] \cdot \frac{\partial \sigma}{\partial w} \\ &= \frac{y(1-\sigma) - (1-y)\sigma}{\sigma(1-\sigma)} \cdot \frac{\partial \sigma}{\partial w} \\ &= \frac{y - \sigma}{\sigma(1-\sigma)} \cdot \frac{\partial \sigma}{\partial w} \\ &= \frac{y - \sigma}{\sigma(1-\sigma)} \cdot \sigma(1-\sigma) \cdot x \\ &= (y - \sigma) \cdot x\end{align*}$$

Therefore:
$$\nabla_w J(w) = \frac{1}{m} \sum_{i=1}^m (\sigma(w^\top x_i) - y_i) x_i = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i$$

**Update Rule** (Gradient Descent):
$$w \leftarrow w - \alpha \nabla_w J(w)$$
where $\alpha$ is the learning rate.

**Stochastic Gradient Descent (SGD)**:
For a randomly selected training example $(x_i, y_i)$:
$$w \leftarrow w - \alpha (f_w(x_i) - y_i) x_i$$

## 8. Linear Regression vs. Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| Task | Regression | Classification |
| Hypothesis | $w^\top x \in \mathbb{R}$ | $\sigma(w^\top x) \in [0, 1]$ |
| Objective | Minimize MSE: $\frac{1}{2m}\sum(y_i - w^\top x_i)^2$ | Minimize Cross-Entropy |
| Solution | Closed-form or GD | GD (no closed form) |
| Output | Continuous value | Probability |


