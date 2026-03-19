# Backpropagation



## 1. The Optimization Problem
The goal of training a neural network is to minimize the **empirical loss** over a training set of $n$ samples:
$$\min_{\theta} L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_{\theta}(x_i), y_i)$$
* **Finite-Sum Structure:** The objective is a sum of individual sample losses $\ell_i(\theta)$.
* **Non-convexity:** Due to non-linear activation functions, the loss surface is generally non-convex, requiring gradient-based optimization.
* **Gradients:** Training requires computing the gradient of the total loss: $\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(\theta)$.

---

## 2. The Backpropagation Algorithm
Backpropagation is an efficient method for computing the gradient $\nabla \ell_i(\theta)$ using the chain rule. It consists of two primary phases:

### Phase 1: Forward Pass
Input $x$ is fed through the network to compute and store intermediate activations:
1.  **Linear Combination:** $a = Vx$
2.  **Hidden Activation:** $h = \sigma(a)$
3.  **Output Logits:** $z = Wh$
4.  **Prediction:** $\hat{y} = g(z)$
5.  **Loss Computation:** $\ell(\theta) = \|y - \hat{y}\|^2$ (for regression)

### Phase 2: Backward Pass
Gradients are computed layer-by-layer starting from the output:
1.  **Output Error ($\delta^{(2)}$):** Compute the error signal at the output layer:
    $$\delta^{(2)} = 2(g(z) - y) \odot g'(z)$$
2.  **Output Weights Gradient:** $\frac{\partial \ell}{\partial W} = \delta^{(2)}h^\top$
3.  **Hidden Error ($\delta^{(1)}$):** Backpropagate the error to the hidden layer:
    $$\delta^{(1)} = (W^\top \delta^{(2)}) \odot \sigma'(a)$$
4.  **Input Weights Gradient:** $\frac{\partial \ell}{\partial V} = \delta^{(1)}x^\top$


---

## 3. Computational Efficiency
The total cost of a single training step is the sum of the forward and backward passes:
* **Forward Pass:** $O(md + km)$ where $d, m, k$ are the dimensions of the input, hidden, and output layers respectively.
* **Backward Pass:** $O(km + md)$.
* **Key Insight:** The backward pass has the same order of complexity as the forward pass, making backpropagation highly efficient for large-scale models.

---

## 4. Stochastic Gradient Descent (SGD)
To scale to massive datasets, optimizers use subsets of data rather than the full batch.

### Mini-batch SGD
Instead of computing the full gradient $\nabla L(\theta)$, we sample a mini-batch $B_k$ of size $b$:
$$\theta_{k+1} = \theta_k - \alpha \frac{1}{b} \sum_{i \in B_k} \nabla \ell_i(\theta_k)$$
* **Learning Rate ($\alpha$):** Determines the step size of the update.
* **Batch Size ($b$):** Common values include 64, 128, or 256.
* **Epoch:** One complete pass through the entire dataset.
* **Iterations per Epoch:** $\frac{n}{b}$, where $n$ is the total number of training samples.


---

## 5. Model Summary: NN vs. Ensemble Methods
Lecture 13 also concludes the discussion on tree-based ensembles:
* **Bagging:** Reduces variance by training multiple trees on bootstrapped samples and averaging results.
* **Random Forest:** Improves Bagging by selecting a **random subset of $m$ features** at each split to de-correlate the trees.
    * **Classification:** $m = \sqrt{N}$.
    * **Regression:** $m = N/3$.