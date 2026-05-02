# K-Means Clustering

## Objective

K-means partitions $n$ data points into $K$ clusters by minimizing the total within-cluster squared distance:

$$\min_{\{c_k\}, r} J(c, r) = \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|x_i - c_k\|^2$$

subject to:

$$r_{ik} \in \{0, 1\}, \quad \sum_{k=1}^K r_{ik} = 1$$

where $r_{ik} = 1$ if data point $i$ is assigned to cluster $k$.

## Lloyd's Algorithm

1. **Initialize:** choose $K$ centroids randomly
2. **Assignment step:** assign each point to the nearest centroid
   $$k^* = \arg\min_{k} \|x_i - c_k\|^2, \quad r_{ik^*} = 1$$
3. **Update step:** recompute each centroid as the mean of its assigned points
   $$c_k = \frac{\sum_{i=1}^n r_{ik} x_i}{\sum_{i=1}^n r_{ik}}$$
4. **Repeat** until assignments no longer change

## Optimization Perspective: Block Coordinate Descent

The K-means algorithm is block coordinate descent on the objective $J(c, r)$:

* Fix $c$, optimize $r$ (assignment step)
* Fix $r$, optimize $c$ (update step)

The refitting step has a closed-form solution by setting the derivative to zero:

$$\frac{\partial J}{\partial c_k} = -2\sum_{i=1}^n r_{ik}(x_i - c_k) = 0 \implies c_k = \frac{\sum_{i} r_{ik} x_i}{\sum_{i} r_{ik}}$$

## Convergence

K-means converges in finitely many steps because:
* The assignment step never increases $J$
* The refitting step never increases $J$
* There are only finitely many possible assignments ($K^n$)

However, the objective is **non-convex**, so the algorithm may converge to a **local minimum**. Finding the global optimum is NP-hard in general.

## Initialization and Variants

| Variant | Description |
|---------|-------------|
| **Multiple restarts** | Run K-means multiple times with random initializations; pick the solution with smallest $J$ |
| **K-means++** | Smart initialization: choose first centroid randomly, then choose subsequent centroids with probability proportional to squared distance from nearest existing centroid |
| **Mini-batch K-means** | Update centroids using only a small random subset at each iteration (faster for large datasets) |
| **K-medians** | Use $\ell_1$ distance and median instead of mean (more robust to outliers) |
| **K-medoids** | Represent each cluster by an actual data point (more robust) |
| **Fuzzy C-means** | Allow soft assignments with membership weights |

**Empty clusters:** If no points are assigned to a cluster, reinitialize the centroid to a random data point or the point farthest from its current centroid.

## Performance Evaluation

**Internal metric — Silhouette Coefficient:**

For a given data point, define:
* $a$: average distance to all other points in the same cluster
* $b$: smallest average distance to points in any other cluster

$$s = \frac{b - a}{\max(a, b)} = \begin{cases} 1 - \frac{a}{b}, & a < b \\ 0, & a = b \\ \frac{b}{a} - 1, & a > b \end{cases}$$

$s \in [-1, 1]$: large positive means well-clustered; near 0 means near boundary; negative means possibly wrong cluster.

**External metric — Rand Index (RI):**

Compare two partitions $X$ and $Y$ of $n$ samples. Count over all unordered pairs:

* $a$: same cluster in both partitions
* $b$: different clusters in both partitions
* $c$: same in $X$, different in $Y$
* $d$: different in $X$, same in $Y$

$$\text{RI} = \frac{a + b}{a + b + c + d} = \frac{a + b}{\binom{n}{2}}$$

**Adjusted Rand Index (ARI):** corrects RI for chance agreement.

$$\text{ARI} = \frac{\sum_{i,j} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2}\right]\left[\sum_j \binom{b_j}{2}\right]/\binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2}\right]\left[\sum_j \binom{b_j}{2}\right]/\binom{n}{2}}$$

ARI = 1 means perfect agreement; near 0 means random agreement.

## Choosing K

We cannot simply minimize $J$ since it always decreases as $K$ increases. Common approaches:

* **Elbow method:** plot $J$ vs. $K$ and look for an "elbow"
* **Silhouette score:** choose $K$ with highest average silhouette coefficient
* **External metrics** (if ground-truth labels are available)
* **Domain knowledge**
