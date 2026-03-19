# Decision Trees


## 1. Core Concepts and Terminology
A **Decision Tree** is a hierarchical non-parametric model that partitions the feature space into local regions via recursive splits.
* **Root Node:** The initial node containing all observations.
* **Internal (Decision) Node:** Specifies a test on an attribute; branches represent outcomes.
* **Leaf (Terminal) Node:** Represents the final prediction (class label or continuous value).
* **Univariate Tree:** Each internal node tests only a single input dimension ($x_i$).
* **Binary vs. Multi-way:** Binary trees split into exactly two children, while multi-way trees can split into $V$ subsets based on categorical levels.

---

## 2. Classification Trees: Impurity Measures
To determine the optimal split, we measure the "impurity" of labels at a node $S$. Let $p_i$ be the proportion of instances belonging to class $i$.

### Entropy & Information Gain (ID3)
Entropy measures the uncertainty/disorder in a node:
$$H(S) = -\sum_{i=1}^{C} p_i \log_2 p_i$$
**Information Gain** is the reduction in entropy achieved by partitioning $S$ using attribute $A$:
$$\text{Gain}(S, A) = H(S) - H(S|A)$$
$$H(S|A) = \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

### Gain Ratio (C4.5)
To correct the bias of Information Gain toward attributes with many levels, the **Gain Ratio** normalizes the gain:
$$\text{GainRatio}(A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(A)}$$
$$\text{SplitInfo}(A) = -\sum_{v=1}^{V} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

### Gini Index (CART)
Measures the probability of misclassification:
$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$

---

## 3. Regression Trees
In regression, the model partitions the space to minimize prediction error, typically the **Residual Sum of Squares (RSS)**.

### Tree-Growing Algorithm
1.  **Initialize:** Start with a root node containing all $N$ observations. Compute the mean $\bar{y}$ and total error $S$:
    $$\overline{y} = \frac{1}{N}\sum_{i=1}^{N} y_i, \quad S = \sum_{i=1}^{N}(y_i - \overline{y})^2$$
2.  **Best Split Search:** For a threshold $w_1$ splitting $S$ into $c_{\text{L}}$ and $c_{\text{R}}$, compute the new error $S_{w_1}$:
    $$S_{w_1} = \sum_{i \in c_{\text{L}}}(y_i - \overline{y}_{c_{\text{L}}})^2 + \sum_{i \in c_{\text{R}}}(y_i - \overline{y}_{c_{\text{R}}})^2$$
    Select the split that maximizes the reduction: $w_1^* = \text{argmax}_{w_1} (S - S_{w_1})$.
3.  **Recurse:** Apply the process to child nodes until stopping criteria are met.

---

## 4. Overfitting and Pruning
Trees are prone to overfitting, capturing noise and creating non-smooth boundaries.

### Stopping Criteria (Pre-pruning)
Growth stops if:
* All instances in a node have the same label/predictor values.
* The reduction in error $S$ is below a threshold $\epsilon$.
* The resulting nodes contain fewer than $q$ observations.

### Post-pruning
1.  Grow a deep tree $T_0$ using the training set.
2.  Iteratively remove (prune) nodes that yield the largest improvement in **validation performance**.
3.  **Replacement Logic:** Replace a subtree with a leaf if the expected error of the subtree exceeds that of the single leaf.

---

## 5. Ensemble Methods
To overcome the high variance and instability of single trees, ensemble methods combine multiple diverse trees.

### Bootstrap Aggregating (Bagging)
* **Step 1:** Sample $N$ records with replacement (Bootstrap) to create diverse training sets.
* **Step 2:** Fit an overgrown tree to each set.
* **Step 3:** Aggregate predictions via majority voting (classification) or averaging (regression).

### Random Forest
Improves Bagging by further reducing correlation between trees.
* At each split, only a **random subset $m$** of features is considered.
* **Typical $m$:** $\sqrt{N}$ for classification; $N/3$ for regression.
* **Error Hierarchy:** Random Forest < Bagging < Single Tree.

---

## 6. Model Summary

| **Feature** | **Details** |
| :--- | :--- |
| **Advantages** | Interpretability (rules), handles mixed data types, no feature scaling required, non-parametric. |
| **Disadvantages** | High variance (instability), prone to overfitting, piecewise-constant predictions (non-smooth). |
| **Comparison** | Unlike Neural Networks, which learn features and classifiers jointly (end-to-end), trees rely on direct data partitioning. |

[Image comparing a smooth regression curve to a piecewise-constant tree approximation]