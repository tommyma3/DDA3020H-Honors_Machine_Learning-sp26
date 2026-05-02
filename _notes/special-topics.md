# Special Topics

## 1. Deep Clustering

### Motivation

PCA is a classical linear method for dimensionality reduction. For high-dimensional data with complicated nonlinear structure (images, text, speech), it is often beneficial to first learn a nonlinear feature representation before clustering.

**Deep clustering** combines deep representation learning with clustering. The main idea is to learn a representation in which clustering becomes easier.

### Two-Stage Deep Clustering

**Approach:**
1. Learn a low-dimensional representation with an autoencoder:
   $$z^{(n)} = f_\theta(x^{(n)})$$
2. Apply a classical clustering algorithm (e.g., K-means) to the learned features.

The K-means objective in the embedded space:

$$\min_{\{r^{(n)}_k\}, \{\mu_k\}} \sum_{n=1}^N \sum_{k=1}^K r^{(n)}_k \|z^{(n)} - \mu_k\|^2$$

subject to $r^{(n)}_k \in \{0, 1\}$ and $\sum_k r^{(n)}_k = 1$.

**Advantages:**
* Easy to implement
* Can use any representation learner and any clustering algorithm

**Limitation:** The learned representation is not directly optimized for clustering.

### Joint Optimization: Deep Embedded Clustering (DEC)

DEC learns the feature extractor and cluster assignments simultaneously.

**Soft assignment.** DEC first maps each sample into a latent feature $z^{(n)} = f_\theta(x^{(n)})$ and maintains $K$ cluster centers $\{\mu_1, \dots, \mu_K\}$. The soft assignment of sample $n$ to cluster $k$ is:

$$q^{(n)}_k = \frac{\left(1 + \|z^{(n)} - \mu_k\|^2/\alpha\right)^{-\frac{\alpha+1}{2}}}{\sum_{j=1}^K \left(1 + \|z^{(n)} - \mu_j\|^2/\alpha\right)^{-\frac{\alpha+1}{2}}}$$

where $\alpha > 0$ is a parameter (often set to 1). This is a Student's t-distribution kernel measuring similarity.

**Target distribution.** DEC does not directly optimize $q^{(n)}_k$. Instead it constructs an auxiliary target distribution $p^{(n)}_k$ from $q^{(n)}_k$:

$$p^{(n)}_k = \frac{(q^{(n)}_k)^2 / f_k}{\sum_{j=1}^K (q^{(n)}_j)^2 / f_j}, \quad \text{where } f_k = \sum_{n=1}^N q^{(n)}_k$$

This target distribution:
* Sharpens confident assignments (squaring emphasizes high probabilities)
* Reduces the influence of very large clusters through normalization by $f_k$

**Objective.** DEC minimizes the KL divergence between the target distribution $P$ and the soft assignments $Q$:

$$\mathcal{L}_{\text{DEC}} = \text{KL}(P \parallel Q) = \sum_{n=1}^N \sum_{k=1}^K p^{(n)}_k \log \frac{p^{(n)}_k}{q^{(n)}_k}$$

Minimizing this with respect to $\theta$ and $\{\mu_k\}$ refines both the representation and the clustering structure simultaneously.

**DEC Algorithm:**
1. **Initialization:** Train an autoencoder to obtain an initial encoder $f_\theta$. Compute latent features and initialize cluster centers by K-means.
2. **Iterative refinement:** Repeat until convergence:
   * Compute soft assignments $q^{(n)}_k$
   * Construct target distribution $p^{(n)}_k$
   * Update $\theta$ and $\{\mu_k\}$ by minimizing $\text{KL}(P \parallel Q)$
3. **Final clustering:** Assign each sample to $\arg\max_k q^{(n)}_k$

---

## 2. Contrastive Learning

### Motivation

Supervised learning requires massive amounts of labeled data, which is expensive to collect. Unlabeled data is often vast. The goal of **self-supervised learning** is to learn a general-purpose feature extractor $f_\theta(x)$ from unlabeled data without human annotations. These learned features can then be used for downstream tasks with only a small number of labeled examples.

### Key Idea

The main principle of contrastive learning is:

> Similar examples should have similar representations.

To achieve this:
* Bring together representations of similar examples (positive pairs)
* Push apart representations of dissimilar examples (negative examples)

### Constructing Positive and Negative Pairs

In unsupervised contrastive learning, we do not have labels. Instead, we construct positive pairs from the same image using data augmentation:

$$x \text{ and } \tilde{x}$$

where $\tilde{x}$ is an augmented view of $x$. All other images in the batch are treated as negative examples.

Common augmentations include random cropping, color distortion, Gaussian blur, and horizontal flipping.

### Triplet Loss

A basic contrastive loss is the **triplet loss**. For each anchor $x$, we have:
* A positive example $x^+$ (similar to $x$)
* A negative example $x^-$ (dissimilar to $x$)

$$\mathcal{L}_{\text{triplet}} = \max\left(0, \|f_\theta(x) - f_\theta(x^+)\|^2 - \|f_\theta(x) - f_\theta(x^-)\|^2 + \epsilon\right)$$

where $\epsilon > 0$ is a margin parameter. The loss encourages the anchor to be closer to the positive than to the negative by at least $\epsilon$.

Compared with an autoencoder, contrastive learning wants similar/dissimilar data to have similar/dissimilar representations, rather than pixel-wise reconstruction.

### N-way Contrastive Loss

Triplet loss considers one positive and one negative at a time. A more general approach treats contrastive learning as an $N$-way classification problem: for an anchor representation $z = f_\theta(x)$, identify its positive pair $z^+$ among many negatives $\{z_i\}$.

$$\mathcal{L}_{N\text{-way}}(\theta) = -\log \frac{\exp(-d(z, z^+))}{\sum_i \exp(-d(z, z_i)) + \exp(-d(z, z^+))}$$

where $d(\cdot, \cdot)$ is a distance function (e.g., cosine distance).

### SimCLR Algorithm

SimCLR is a simple framework for contrastive learning of visual representations.

**Algorithm:**
1. Sample a minibatch of $N$ examples $x_1, \dots, x_N$
2. Augment each example twice to get $\tilde{x}_1, \dots, \tilde{x}_N, \tilde{x}_{N+1}, \dots, \tilde{x}_{2N}$
3. Embed examples with $f_\theta$ to get $\tilde{z}_1, \dots, \tilde{z}_{2N}$
4. Compute pairwise cosine distances:
   $$d(z_i, z_j) = -\frac{z_i^\top z_j}{\|z_i\| \|z_j\|}$$
5. Update $\theta$ with respect to the $N$-way loss:
   $$\mathcal{L}_{N\text{-way}}(\theta) = \sum_{i=1}^N -\log \frac{\exp(-d(\tilde{z}_i, \tilde{z}_{N+i}))}{\sum_{j \neq i} \exp(-d(\tilde{z}_i, \tilde{z}_j))}$$

The key idea is to compare augmentations of the data itself, treating the two augmented views of the same image as a positive pair and all other pairs as negatives.
