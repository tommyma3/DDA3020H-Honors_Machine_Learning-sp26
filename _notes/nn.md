# Neural Networks


## 1. Introduction to Neural Networks
Neural Networks are non-linear statistical models used for both regression and classification. They are inspired by biological neurons and consist of interconnected processing units (neurons) organized into layers.

* **Input Layer:** Receives the raw feature vector $\mathbf{x} \in \mathbb{R}^d$.
* **Hidden Layer(s):** Intermediate layers that extract increasingly abstract features.
* **Output Layer:** Produces the final prediction $\hat{y}$.


---

## 2. The Artificial Neuron (Perceptron)
A single neuron computes a weighted sum of its inputs, adds a bias, and applies a non-linear **activation function**.

### Mathematical Formulation
For a neuron with input vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^\top$:
1.  **Linear Combination:** $$z = \sum_{j=1}^{d} w_j x_j + b = \mathbf{w}^\top \mathbf{x} + b$$
2.  **Non-linear Activation:**
    $$a = \sigma(z)$$
    where $\mathbf{w}$ is the weight vector, $b$ is the bias, and $\sigma(\cdot)$ is the activation function.

### Common Activation Functions
Activation functions introduce non-linearity, allowing the network to approximate complex functions.
* **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$ (Maps to $(0, 1)$)
* **Hyperbolic Tangent (Tanh):** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ (Maps to $(-1, 1)$)
* **Rectified Linear Unit (ReLU):** $\sigma(z) = \max(0, z)$ (Standard for deep networks)


---

## 3. Multilayer Perceptrons (MLP)
In a feedforward network, neurons are arranged in layers where the output of one layer becomes the input of the next.

### Forward Propagation
For a network with $L$ layers, let $a^{(l)}$ be the activation of layer $l$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$
where $W^{(l)}$ is the weight matrix for layer $l$ and $a^{(0)} = \mathbf{x}$.

---

## 4. Training and Loss Functions
To train the network, we define a cost function $J(\theta)$ that measures the discrepancy between the prediction $\hat{y}$ and the ground truth $y$.

* **Mean Squared Error (Regression):**
    $$J(\theta) = \frac{1}{2N} \sum_{i=1}^{N} \|\hat{y}^{(i)} - y^{(i)}\|^2$$
* **Cross-Entropy Loss (Classification):**
    $$J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

---

## 5. Backpropagation and Optimization
Optimization is performed using **Gradient Descent** to minimize the loss function by updating weights and biases:
$$w \leftarrow w - \eta \frac{\partial J}{\partial w}$$

### The Chain Rule
Backpropagation uses the chain rule to compute the gradient of the loss with respect to each parameter by moving backward through the network.
For a weight $w_{ij}^{(l)}$ in layer $l$:
$$\frac{\partial J}{\partial w_{ij}^{(l)}} = \frac{\partial J}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial w_{ij}^{(l)}}$$


---

## 6. Feature Learning
A key advantage of Neural Networks over traditional methods (like Decision Trees) is their ability to learn **hierarchical representations**.
* **Early Layers:** Typically learn low-level features (e.g., edges, blobs in images).
* **Deeper Layers:** Combine low-level features into complex patterns (e.g., shapes, objects).

In tasks like handwritten digit classification ($28 \times 28$ pixels), hidden units act as feature detectors. Visualizing the weight vectors $\mathbf{w}_j$ as images often reveals that the network has learned to respond strongly to specific strokes or orientations.

| **Property** | **Neural Networks** | **Decision Trees** |
| :--- | :--- | :--- |
| **Feature Construction** | Learned automatically via hidden layers. | Manual feature engineering or axis-aligned splits. |
| **Boundaries** | Smooth and non-linear. | Piecewise constant (box-like). |
| **Scalability** | High (with GPU/Big Data). | Becomes complex/unstable with very high dimensionality. |