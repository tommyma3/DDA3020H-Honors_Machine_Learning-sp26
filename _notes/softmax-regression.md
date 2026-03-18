# Softmax Regression

## 1. Regularized Logistic Regression

**Problem**: Overfitting when model is too complex or data is limited.

**L2 Regularization** (Ridge):
$$J_\lambda(w) = J(w) + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$

Note: Bias term $w_0$ is typically NOT regularized.

**L1 Regularization** (Lasso):
$$J_\lambda(w) = J(w) + \frac{\lambda}{m} \sum_{j=1}^d |w_j|$$

### Gradient for L2-Regularized Logistic Regression:
$$\nabla_w J_\lambda(w) = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i + \frac{\lambda}{m} w$$

(ignoring $w_0$ in the regularization term)

## 2. MAP Interpretation of Regularization

**Bayesian View**: Treat parameters $w$ as random variables with prior $P(w)$.

**Maximum A Posteriori (MAP) Estimation**:
$$w_{\text{MAP}} = \arg\max_w P(w|X, y) = \arg\max_w P(y|X; w)P(w)$$

Taking logs:
$$w_{\text{MAP}} = \arg\max_w [\log P(y|X; w) + \log P(w)]$$

**L2 Regularization = Gaussian Prior**:
$$w_j \sim \mathcal{N}(0, \sigma^2) \implies \log P(w) = -\frac{1}{2\sigma^2} \sum_j w_j^2 + \text{constant}$$

This yields L2-regularized logistic regression where $\lambda = 1/\sigma^2$.

**L1 Regularization = Laplace Prior**:
$$P(w_j) \propto \exp\left(-\frac{|w_j|}{b}\right) \implies \log P(w) = -\frac{1}{b} \sum_j |w_j| + \text{constant}$$

## 3. Multi-Class Classification

**Problem**: $y \in \{1, 2, \ldots, C\}$ where $C > 2$

Examples:
- Weather: $\{\text{sunny}, \text{cloudy}, \text{rain}, \text{snow}\}$
- Email tags: $\{\text{work}, \text{family}, \text{friends}, \text{hobby}\}$

## 4. One-vs-All (OvA) Approach

**Method**:
For each class $j \in \{1, \ldots, C\}$, train a binary classifier $f_{w_j}(x)$ where:
- Class $j$: positive examples
- All other classes: negative examples

**Prediction**:
$$\hat{y} = \arg\max_j f_{w_j}(x)$$

**Pros**: Simple, easy to implement, can parallelize

**Cons**:
- Requires training C separate models
- Predicted probabilities don't necessarily sum to 1
- Class imbalance in each binary problem

## 5. Softmax Regression (Multinomial Logistic Regression)

**Idea**: Directly model class probabilities $P(y=j|x)$ for all $j$ simultaneously.

### Model Definition:

For each class $j$, define a linear score:
$$z_j = w_j^\top x$$

**Parameter Matrix**: $W = [w_1, w_2, \ldots, w_C] \in \mathbb{R}^{(d+1)\times C}$

**Softmax Function**: Converts scores to probabilities
$$f_W^{(j)}(x) = P(y=j|x; W) = \frac{\exp(z_j)}{\sum_{c=1}^C \exp(z_c)} = \frac{\exp(w_j^\top x)}{\sum_{c=1}^C \exp(w_c^\top x)}$$

**Properties**:
- $f_W^{(j)}(x) \in [0, 1]$ for all $j$
- $\sum_{j=1}^C f_W^{(j)}(x) = 1$

**Prediction Rule**:
$$\hat{y} = \arg\max_j f_W^{(j)}(x) = \arg\max_j w_j^\top x$$

### Cross-Entropy Loss for Softmax:

For training example $(x_i, y_i)$, define:
- Ground-truth (one-hot): $P(y=j|x_i, y_i) = \mathbb{I}(y_i = j)$
- Predicted: $P(y=j|x_i; W) = f_W^{(j)}(x_i)$

**Per-example Loss**:
$$\text{cost}_i = -\sum_{j=1}^C \mathbb{I}(y_i = j) \cdot \log(f_W^{(j)}(x_i)) = -\log(f_W^{(y_i)}(x_i))$$

**Overall Objective**:
$$J(W) = \frac{1}{m} \sum_{i=1}^m \sum_{j=1}^C \mathbb{I}(y_i = j) \cdot \log(f_W^{(j)}(x_i)) = -\frac{1}{m} \sum_{i=1}^m \log(f_W^{(y_i)}(x_i))$$

### Convexity Proof Sketch:

The softmax objective is convex because:

1. The log-sum-exp function $g(z) = \log(\sum_c \exp(z_c))$ is convex
2. The softmax loss can be written as: $\text{cost}_i = -w_{y_i}^\top x_i + \log(\sum_c \exp(w_c^\top x_i))$
3. This is an affine function plus a convex function (log-sum-exp)
4. Therefore, the overall objective is convex in $W$

## 6. Gradient Descent for Softmax Regression

**Gradient** with respect to $w_j$:
$$\frac{\partial J(W)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m [f_W^{(j)}(x_i) - \mathbb{I}(y_i = j)] x_i$$

**Proof**:

From $J(W) = -\frac{1}{m} \sum_i \log(f_W^{(y_i)}(x_i))$, where:
$$f_W^{(y_i)}(x_i) = \frac{\exp(w_{y_i}^\top x_i)}{\sum_c \exp(w_c^\top x_i)}$$

Let $\sigma_j = f_W^{(j)}(x_i)$ for brevity.

For a single example $(x, y)$ with $z_j = w_j^\top x$:
$$\begin{align*}-\frac{\partial}{\partial w_j} \log(\sigma_y) &= -\frac{\partial}{\partial w_j} \left[z_y - \log\left(\sum_c \exp(z_c)\right)\right] \\ &= -\left[\frac{\partial z_y}{\partial w_j} - \frac{1}{\sum_c \exp(z_c)} \cdot \frac{\partial}{\partial w_j} \sum_c \exp(z_c)\right]\end{align*}$$

Case 1: $j = y$ (true class):
$$\begin{align*}-\frac{\partial}{\partial w_y} \log(\sigma_y) &= -\left[x - \frac{1}{\sum_c \exp(z_c)} \cdot \exp(z_y) \cdot x\right] \\ &= -[x - \sigma_y x] \\ &= (\sigma_y - 1)x\end{align*}$$

Case 2: $j \neq y$:
$$\begin{align*}-\frac{\partial}{\partial w_j} \log(\sigma_y) &= -\left[0 - \frac{1}{\sum_c \exp(z_c)} \cdot \exp(z_j) \cdot x\right] \\ &= \sigma_j x\end{align*}$$

Combining both cases using indicator function:
$$-\frac{\partial}{\partial w_j} \log(\sigma_y) = (\sigma_j - \mathbb{I}(y=j)) x$$

**Update Rule** (SGD):
For randomly sampled $(x_i, y_i)$:
$$w_j \leftarrow w_j - \alpha [f_W^{(j)}(x_i) - \mathbb{I}(y_i = j)] x_i$$

**Remark**: This is a natural extension of logistic regression gradient descent. For $C=2$, it reduces to the logistic regression update.

