# Key Takeaways

1. **Logistic Regression**: Models binary classification probabilities using sigmoid function
2. **Cross-Entropy Loss**: Natural loss for probability models, derived from MLE
3. **Gradient**: For logistic regression, $\nabla J(w) = \frac{1}{m} \sum(f_w(x_i) - y_i)x_i$
4. **Regularization**: L2 corresponds to Gaussian prior; prevents overfitting
5. **Softmax**: Generalization of logistic regression to $C > 2$ classes
6. **Convexity**: Both logistic and softmax regression have convex objectives → gradient descent finds global optimum
7. **Connection**: All these models are part of the Generalized Linear Model family

---

# Mathematical Proofs Summary

## Proof 1: Sigmoid Derivative
$$\sigma(z) = \frac{1}{1+e^{-z}}$$
$$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

**Proof**:
$$\begin{align*}\sigma'(z) &= \frac{\partial}{\partial z} (1+e^{-z})^{-1} \\ &= -1 \cdot (1+e^{-z})^{-2} \cdot (-e^{-z}) \\ &= \frac{e^{-z}}{(1+e^{-z})^2} \\ &= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} \\ &= \sigma(z) \cdot \left[\frac{1+e^{-z} - 1}{1+e^{-z}}\right] \\ &= \sigma(z) \cdot \left[1 - \frac{1}{1+e^{-z}}\right] \\ &= \sigma(z) \cdot (1-\sigma(z))\end{align*}$$

## Proof 2: Logistic Loss Gradient
As shown in Section 6.7, the gradient is:
$$\nabla J(w) = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i)x_i$$

## Proof 3: Softmax Convexity
The softmax objective is convex because:
1. log-sum-exp is convex (can be shown via Hessian positivity)
2. Affine composition preserves convexity
3. Sum of convex functions is convex

## Proof 4: Softmax Gradient
As shown in Section 7.6:
$$\frac{\partial J(W)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m [f_W^{(j)}(x_i) - \mathbb{I}(y_i=j)]x_i$$
