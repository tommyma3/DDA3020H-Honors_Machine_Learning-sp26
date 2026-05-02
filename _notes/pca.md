# Principal Component Analysis

## Problem Setup

Given data $\{x^{(n)}\}_{n=1}^N$ with $x^{(n)} \in \mathbb{R}^D$, find a $K$-dimensional subspace $\mathcal{S}$ ($K < D$) spanned by orthonormal basis $\{u_1, \dots, u_K\}$ that preserves the important structure.

Let $U = [u_1 \cdots u_K] \in \mathbb{R}^{D \times K}$ with $U^\top U = I_K$.

The projection of a centered point onto the subspace is:

$$z = U^\top (x - \mu) \in \mathbb{R}^K$$

The reconstruction is:

$$\tilde{x} = \mu + Uz = \mu + UU^\top(x - \mu)$$

**Orthogonality of projection error:** $U^\top(x - \tilde{x}) = 0$, so the error is orthogonal to the subspace.

---

## Derivation 1: Maximum Variance

Choose the subspace such that the variance of the reconstructions is maximized. Since $\tilde{\mu} = \mu$:

$$\max_{U: U^\top U = I} \frac{1}{N} \sum_{n=1}^N \|\tilde{x}^{(n)} - \mu\|^2$$

Since $\tilde{x}^{(n)} - \mu = Uz^{(n)}$ and $\|Uz^{(n)}\|^2 = \|z^{(n)}\|^2$ (because $U^\top U = I$):

$$\max_{U: U^\top U = I} \frac{1}{N} \sum_{n=1}^N \|z^{(n)}\|^2 = \max_{U: U^\top U = I} \text{Tr}(U^\top S U)$$

where $S$ is the sample covariance matrix:

$$S = \frac{1}{N} \sum_{n=1}^N (x^{(n)} - \mu)(x^{(n)} - \mu)^\top$$

---

## Derivation 2: Minimum Reconstruction Error

Minimize the average squared reconstruction error:

$$\min_{U: U^\top U = I} \frac{1}{N} \sum_{n=1}^N \|x^{(n)} - \tilde{x}^{(n)}\|^2$$

---

## Equivalence of the Two Derivations

For each sample, decompose:

$$x^{(n)} - \mu = (\tilde{x}^{(n)} - \mu) + (x^{(n)} - \tilde{x}^{(n)})$$

The two terms are orthogonal. By Pythagoras:

$$\|x^{(n)} - \mu\|^2 = \|\tilde{x}^{(n)} - \mu\|^2 + \|x^{(n)} - \tilde{x}^{(n)}\|^2$$

Averaging over all samples:

$$\underbrace{\frac{1}{N}\sum_n \|x^{(n)} - \mu\|^2}_{\text{constant}} = \underbrace{\frac{1}{N}\sum_n \|\tilde{x}^{(n)} - \mu\|^2}_{\text{projected variance}} + \underbrace{\frac{1}{N}\sum_n \|x^{(n)} - \tilde{x}^{(n)}\|^2}_{\text{reconstruction error}}$$

Since the left side is independent of $U$:

$$\max \text{projected variance} \iff \min \text{reconstruction error}$$

---

## Lagrangian and Eigendecomposition

The PCA objective is:

$$\max_U \text{Tr}(U^\top S U) \quad \text{subject to } U^\top U = I$$

Lagrangian with multiplier matrix $\Lambda$:

$$\mathcal{L}(U, \Lambda) = \text{Tr}(U^\top S U) - \text{Tr}(\Lambda(U^\top U - I))$$

Stationary condition:

$$\frac{\partial \mathcal{L}}{\partial U} = 2SU - U(\Lambda + \Lambda^\top) = 0$$

Let $M = \Lambda + \Lambda^\top$ (symmetric). Then $SU = UM$.

Since $M$ is symmetric, it has an eigendecomposition $M = RDR^\top$. Let $U_0 = UR$. Then $U_0$ is still feasible and satisfies $SU_0 = U_0 D$, meaning the columns of $U_0$ are **eigenvectors of $S$**.

Let $S = Q\Lambda Q^\top = \sum_{i=1}^D \lambda_i q_i q_i^\top$ with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D \geq 0$.

If we choose $U = [q_{t_1} \cdots q_{t_K}]$, then:

$$\text{Tr}(U^\top S U) = \sum_{j=1}^K \lambda_{t_j}$$

To maximize, choose the **top-$K$ eigenvectors**:

$$U^* = [q_1 \; q_2 \; \cdots \; q_K]$$

---

## Interpretation

| Concept | Meaning |
|---------|---------|
| Eigenvector $q_k$ | Principal direction (basis vector of PCA subspace) |
| Eigenvalue $\lambda_k$ | Variance of the data along that direction |
| Larger $\lambda_k$ | Preserves more information about the original data |

In many datasets, the first few principal components explain most of the total variance.

---

## PCA Algorithm

1. **Compute the sample mean:** $\mu = \frac{1}{N}\sum_{n=1}^N x^{(n)}$
2. **Compute the covariance matrix:** $S = \frac{1}{N}\sum_{n=1}^N (x^{(n)} - \mu)(x^{(n)} - \mu)^\top$
3. **Compute eigendecomposition** of $S$ and sort eigenvalues in descending order
4. **Form $U = [q_1 \cdots q_K]$** using the top-$K$ eigenvectors
5. **Compute low-dimensional representation:** $z^{(n)} = U^\top(x^{(n)} - \mu)$
