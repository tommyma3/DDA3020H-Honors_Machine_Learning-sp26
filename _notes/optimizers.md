# Optimizers for Neural Networks

## 1. The Optimization Problem

Neural network training is formulated as empirical risk minimization:

$$\min_{\theta} L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(\theta), \quad \text{where } \ell_i(\theta) = \ell(f_\theta(x_i), y_i)$$

* **Finite-sum structure:** The objective decomposes over training samples.
* **Non-convexity:** Due to nonlinear activation functions, the loss landscape is generally non-convex.
* **Gradient computation:** Training requires computing $\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(\theta)$, done efficiently via backpropagation.

---

## 2. Gradient Descent and Stochastic Gradient Descent

### Gradient Descent (GD)

$$\theta_{k+1} = \theta_k - \alpha_k \nabla L(\theta_k)$$

Each iteration requires computing gradients for all $n$ training samples.

### Stochastic Gradient Descent (SGD)

When $n$ is large, computing the full gradient is expensive. Instead, sample an index $i_k \in \{1, \dots, n\}$ uniformly and use the stochastic gradient $\nabla \ell_{i_k}(\theta_k)$:

$$\theta_{k+1} = \theta_k - \alpha_k \nabla \ell_{i_k}(\theta_k)$$

Since $\mathbb{E}_{i_k}[\nabla \ell_{i_k}(\theta_k)] = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(\theta_k) = \nabla L(\theta_k)$, SGD uses an **unbiased estimate** of the full gradient.

### Mini-batch SGD

In practice, we use mini-batches $B_k \subseteq \{1, \dots, n\}$ of size $b$:

$$\theta_{k+1} = \theta_k - \alpha_k \frac{1}{b} \sum_{i \in B_k} \nabla \ell_i(\theta_k)$$

* **Epoch:** One complete pass through the entire dataset.
* **Iterations per epoch:** $\frac{n}{b}$.
* **Random reshuffling:** Before each epoch, the dataset is often shuffled and divided into consecutive mini-batches. Empirically, this converges faster than independent sampling.

---

## 3. SGD with Momentum

### Heavy-ball Momentum

$$m_{k+1} = \beta m_k + g_k, \quad \theta_{k+1} = \theta_k - \alpha_k m_{k+1}$$

where $0 \le \beta < 1$ is the momentum parameter (typically $\beta = 0.9$).

* $m_k$ accumulates information from previous gradients.
* Momentum acts like a ball rolling downhill, keeping movement in directions that consistently decrease the loss.
* Expanding the recursion: $m_{k+1} = g_k + \beta g_{k-1} + \beta^2 g_{k-2} + \cdots$, so recent gradients receive larger weight.

### Exponential Averaging Form

Sometimes written as:

$$v_{k+1} = \beta v_k + (1 - \beta) g_k$$

Since $\sum_{t=0}^{\infty} (1-\beta)\beta^t = 1$, this is an **exponentially weighted average** of past gradients.

### Nesterov Accelerated Gradient (NAG)

A variant that evaluates the gradient at a look-ahead point:

$$\tilde{\theta}_k = \theta_k - \alpha_k v_k, \quad v_{k+1} = \beta v_k + \nabla L(\tilde{\theta}_k), \quad \theta_{k+1} = \theta_k - \alpha_k v_{k+1}$$

* Uses gradient information from a look-ahead point.
* Less commonly used in nonconvex optimization.

---

## 4. Gradient Normalization and Clipping

### Why Normalize or Clip?

In deep networks, gradients can become very large (exploding gradients), causing unstable updates. A single large gradient can produce an excessively large step, slowing training or causing divergence.

### Gradient Normalization

$$\tilde{g}_k = \frac{g_k}{\|g_k\|_2}, \quad \theta_{k+1} = \theta_k - \alpha_k \tilde{g}_k$$

The update depends only on the direction of the gradient; step length is controlled by the learning rate.

### Gradient Clipping

More commonly used than normalization:

$$\tilde{g}_k = \begin{cases} g_k, & \|g_k\|_2 \le \tau \\ \tau \frac{g_k}{\|g_k\|_2}, & \|g_k\|_2 > \tau \end{cases}$$

If the gradient is small, the update is unchanged. If large, its magnitude is capped at $\tau$ (typically $\tau = 1$ or $0.5$).

* Can be viewed as projecting the gradient onto a ball of radius $\tau$.
* Two hyperparameters: learning rate $\alpha_k$ and clipping threshold $\tau$.

---

## 5. Adaptive Optimizers

Different parameters may have gradients of very different magnitudes. Adaptive methods use a different effective step size for each coordinate based on historical gradients:

* **Smaller steps** for coordinates with large historical gradients.
* **Larger steps** for coordinates with small historical gradients.

### AdaGrad

Accumulates squared gradients:

$$v_{k+1} = v_k + g_k \odot g_k$$

$$\theta_{k+1} = \theta_k - \frac{\alpha \, g_k}{\sqrt{v_{k+1}} + \epsilon}$$

where $\odot$ denotes element-wise multiplication and operations are element-wise.

* Each coordinate has its own adaptive step size.
* Coordinates with large past gradients receive smaller future updates.
* Works well for sparse features.
* **Drawback:** The accumulated sum keeps growing, so the effective learning rate may become too small.

### RMSProp

Improves AdaGrad by using an exponentially weighted average instead of a cumulative sum:

$$v_{k+1} = \rho v_k + (1 - \rho)(g_k \odot g_k)$$

$$\theta_{k+1} = \theta_k - \frac{\alpha \, g_k}{\sqrt{v_{k+1}} + \epsilon}$$

where $0 \le \rho < 1$ (typically $\rho = 0.9$) and $\epsilon \approx 10^{-8}$ for numerical stability.

* The denominator does not grow monotonically.
* Adapts the learning rate based on recent gradients.
* Works well in practice for training neural networks.

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSProp-style scaling:

**First moment estimate:** $m_{k+1} = \beta_1 m_k + (1 - \beta_1) g_k$

**Second moment estimate:** $v_{k+1} = \beta_2 v_k + (1 - \beta_2)(g_k \odot g_k)$

Since $m_0 = v_0 = 0$, the estimates are biased toward zero in early iterations. Apply **bias correction**:

$$\hat{m}_{k+1} = \frac{m_{k+1}}{1 - \beta_1^{k+1}}, \quad \hat{v}_{k+1} = \frac{v_{k+1}}{1 - \beta_2^{k+1}}$$

$$\theta_{k+1} = \theta_k - \frac{\alpha \, \hat{m}_{k+1}}{\sqrt{\hat{v}_{k+1}} + \epsilon}$$

**Typical defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

The correction mainly affects the first few iterations and becomes negligible later.

---

## 6. Weight Decay and AdamW

### L2 Regularization (Weight Decay)

To reduce overfitting, add a regularization term:

$$\min_{\theta} L(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$$

The gradient becomes $\nabla L(\theta) + \lambda \theta$, so the SGD update is:

$$\theta_{k+1} = \theta_k - \alpha_k (g_k + \lambda \theta_k)$$

This is often called **weight decay** as it encourages smaller parameter values and can improve generalization.

### AdamW: Decoupled Weight Decay

For SGD, L2 regularization and weight decay are equivalent. However, for Adam they are not, because the adaptive scaling affects the regularization term. AdamW applies weight decay **separately** from the gradient update:

$$\theta_{k+1} = \theta_k - \frac{\alpha \, \hat{m}_{k+1}}{\sqrt{\hat{v}_{k+1}} + \epsilon} - \alpha \lambda \theta_k$$

* Weight decay is decoupled from the gradient update.
* Works better with adaptive methods like Adam.
* Widely used in modern deep learning.

---

## 7. Learning Rate Schedules

| Schedule | Formula |
|----------|---------|
| **Constant** | $\alpha_k = \alpha_0$ |
| **Exponential decay** | $\alpha_k = \alpha_0 \cdot r^k$ for some $r \in (0, 1)$ |
| **Linear decay** | $\alpha_k = \alpha_0 (1 - \frac{k}{T})$ where $T$ is total epochs |
| **Cosine** | $\alpha_k = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\frac{k}{T}\pi))$ |
| **Reduce on plateau** | Decrease $\alpha$ when validation loss improvement stalls |
| **Warmup** | Start with small $\alpha$, increase over a few epochs to $\alpha_{\max}$, then apply main schedule |

---

## 8. Practical Recommendations

* **Start simple:** Adam/AdamW is often a good default choice.
* **Tune the learning rate first:** it usually matters more than other hyperparameters.
* **Monitor both training and validation performance.**
* If training is unstable: try a smaller learning rate.
* If progress is too slow: try a different optimizer or learning rate schedule.
