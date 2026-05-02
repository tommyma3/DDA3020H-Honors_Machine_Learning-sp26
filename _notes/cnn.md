# Convolutional Neural Networks

## 1. Motivation

Fully connected neural networks for images have two key issues:
* **Lose spatial structure:** Flattening an image into a vector destroys the 2D spatial relationships between pixels.
* **Too many parameters:** For a $32 \times 32 \times 3$ image, a single fully connected layer to 1000 neurons has $32 \times 32 \times 3 \times 1000 \approx 3$ million weights.

CNNs preserve spatial structure and use weight sharing to dramatically reduce parameters.

---

## 2. Convolutional Layer

### Core Operation

A convolutional layer applies a set of learnable filters (kernels) to the input. Each filter slides spatially over the input and computes dot products.

For a filter of size $F \times F \times C$ applied to an input of size $W_1 \times H_1 \times C$:
* At each spatial location, compute: $w^\top x + b$ (a dot product over the $F \times F \times C$ patch).
* The result is a single number in the output activation map.

### Hyperparameters

A convolutional layer has 4 hyperparameters:

| Hyperparameter | Symbol | Description |
|---------------|--------|-------------|
| **Number of filters** | $K$ | How many different filters to learn |
| **Filter size** | $F$ | Spatial size of each filter (e.g., $3 \times 3$, $5 \times 5$) |
| **Stride** | $S$ | Step size when sliding the filter |
| **Zero padding** | $P$ | Number of pixels to pad around the border |

### Output Size Formula

Given input $W_1 \times H_1 \times C$:

$$W_2 = \frac{W_1 - F + 2P}{S} + 1, \quad H_2 = \frac{H_1 - F + 2P}{S} + 1$$

The output volume is $W_2 \times H_2 \times K$.

**Common settings:**
* $F = 3$, $S = 1$, $P = 1$ (preserves spatial size)
* $F = 5$, $S = 1$, $P = 2$ (preserves spatial size)
* $F = 5$, $S = 2$, $P$ chosen to fit
* $F = 1$, $S = 1$, $P = 0$

### Number of Parameters

Each filter has $F \times F \times C$ weights plus 1 bias. With $K$ filters:

$$\text{Total parameters} = K \cdot (F^2 \cdot C + 1)$$

**Example:** Input $32 \times 32 \times 3$, $K = 10$ filters of size $5 \times 5$, stride 1, pad 2.
* Output size: $32 \times 32 \times 10$
* Parameters per filter: $5 \times 5 \times 3 + 1 = 76$
* Total parameters: $10 \times 76 = 760$

### Intuition

Each filter learns to detect a specific pattern:
* Early layers: edges, blobs, colors
* Deeper layers: textures, shapes, object parts

When a patch of the image matches the filter pattern, the dot product yields a large value.

---

## 3. Pooling Layer

Pooling reduces spatial dimensions, making representations smaller and more manageable.

### Max Pooling

For each $2 \times 2$ region (typical), output the maximum value.

* Operates independently on each activation map.
* No learnable parameters.
* Introduces translation invariance.

Common setup: $2 \times 2$ filters with stride 2, which halves the spatial dimensions.

---

## 4. Fully Connected Layer

After several convolutional and pooling layers, the final feature maps are flattened and fed into one or more fully connected layers.

* Connects every neuron to all activations in the previous layer.
* The final layer's size and activation depend on the task (e.g., softmax for classification).

---

## 5. Typical CNN Architecture

A CNN is typically a sequence of:

$$\text{[CONV + ReLU]} \rightarrow \text{[POOL]} \rightarrow \text{[CONV + ReLU]} \rightarrow \text{[POOL]} \rightarrow \cdots \rightarrow \text{[FC]} \rightarrow \text{[Output]}$$

Key properties:
* **Hierarchical feature learning:** Early layers extract low-level features; deeper layers combine them into high-level patterns.
* **Parameter efficiency:** Convolution layers share weights spatially, drastically reducing parameters compared to fully connected networks.
* **Translation equivariance:** Shifting the input shifts the output correspondingly (before pooling).

---

## 6. Classic Architectures (Overview)

| Network | Year | Key Innovation |
|---------|------|---------------|
| **LeNet** | 1998 | First successful CNN for digit recognition |
| **AlexNet** | 2012 | Reinvigorated deep learning with GPU training, ReLU, dropout |
| **VGGNet** | 2014 | Showed that depth matters; used small $3 \times 3$ filters exclusively |
| **ResNet** | 2015 | Introduced skip connections to enable very deep networks |
