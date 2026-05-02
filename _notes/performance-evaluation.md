# Performance Evaluation

## 1. Model Selection and Hyperparameter Tuning

### Underfitting vs. Overfitting

| Condition | Training Error | Test Error | Cause |
|-----------|---------------|------------|-------|
| **Underfitting** | High | High | Model too simple |
| **Overfitting** | Low | High | Model too complex |

The goal is to choose a model with appropriate complexity that generalizes well to new data.

### Hyperparameter Tuning

Hyperparameters control the learning process or model complexity (e.g., polynomial degree, tree depth, regularization $C$, learning rate). They must be chosen before training.

**Bad ideas:**
1. Tuning on the entire dataset — no independent evaluation left.
2. Tuning on the test set — the test error is no longer an unbiased estimate.

**Good idea:** Split data into training, validation, and test sets.
* Train on the training set.
* Choose hyperparameters using the validation set.
* Evaluate the final model on the test set (used only once).

---

## 2. Cross-Validation

### Hold-out Validation

Split data into training / validation / test sets. The result may depend strongly on the particular split.

### K-fold Cross-Validation

1. Split the training data into $K$ roughly equal folds.
2. For each trial $k = 1, \dots, K$: train on all folds except $k$, validate on fold $k$.
3. Average the validation performance over all $K$ trials.
4. Choose hyperparameters with the best average validation performance.

**Practical considerations:**
* If $K$ is too small: training sets are small, performance estimate may be pessimistic.
* If $K$ is too large: validation sets are tiny, estimate becomes more variable; computation is expensive.
* Common choices: $K = 5$ or $K = 10$.
* In deep learning, a simple hold-out validation set is often used instead due to computational cost.

---

## 3. Evaluation Metrics for Regression

| Metric | Formula | Properties |
|--------|---------|------------|
| **MSE** | $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Penalizes large errors more heavily |
| **MAE** | $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$ | More robust to outliers |

---

## 4. Evaluation Metrics for Classification

### Confusion Matrix

For binary classification (positive / negative):

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

* **Type I error:** False Positive (FP)
* **Type II error:** False Negative (FN)

### Basic Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

However, accuracy can be misleading for imbalanced classes.

### Rate-based Metrics

For the **positive class**:
$$\text{TPR (Recall)} = \frac{TP}{TP + FN}, \quad \text{FNR} = \frac{FN}{TP + FN}$$

For the **negative class**:
$$\text{TNR} = \frac{TN}{FP + TN}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

Note: $\text{TPR} + \text{FNR} = 1$ and $\text{TNR} + \text{FPR} = 1$.

**Balanced accuracy:** $\frac{\text{TPR} + \text{TNR}}{2}$

### Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

* **Precision:** Of all predicted positives, how many are correct?
* **Recall:** Of all actual positives, how many are found?

### Cost-Sensitive Evaluation

When different mistakes have different importance, assign weights $C_{p,p}, C_{n,n}, C_{p,n}, C_{n,p}$:

$$\text{Cost-weighted accuracy} = \frac{C_{p,p} TP + C_{n,n} TN}{C_{p,p} TP + C_{n,n} TN + C_{p,n} FN + C_{n,p} FP}$$

---

## 5. ROC Curve and AUC

### Operating Points and Thresholds

A classifier outputs scores; a decision threshold $\tau$ separates predicted positive from negative. By varying $\tau$, we obtain different trade-offs between FPR and FNR. Each choice of $\tau$ is called an **operating point**.

### DET Curve

Plot FNR (y-axis) vs. FPR (x-axis) for all thresholds. A better classifier has a DET curve closer to the bottom-left corner.

### ROC Curve

Plot TPR = $1 - \text{FNR}$ (y-axis) vs. FPR (x-axis) for all thresholds. A better classifier has an ROC curve closer to the **top-left corner** (high TPR, low FPR).

**Equal Error Rate (EER):** The point where $\text{FPR} = \text{FNR}$.

### AUC (Area Under the ROC Curve)

AUC provides an overall measure of classifier performance across all possible thresholds.

* **Scale-invariant:** Changing the scale of prediction scores does not affect AUC.
* **Threshold-invariant:** Summarizes performance over all thresholds.

**Interpretation:** AUC is the probability that a randomly chosen positive sample receives a higher score than a randomly chosen negative sample.

$$\text{AUC} = P(s^+ > s^-)$$

where $s^+$ and $s^-$ are scores of randomly drawn positive and negative samples.

### Empirical AUC Formula

Given $m_+$ positive samples and $m_-$ negative samples with classifier scores $g(x)$. Define:

$$e_{ij} = g(x_i^+) - g(x_j^-)$$

$$u(e) = \begin{cases} 1, & e > 0 \\ 0.5, & e = 0 \\ 0, & e < 0 \end{cases}$$

$$\widehat{\text{AUC}} = \frac{1}{m_+ m_-} \sum_{i=1}^{m_+} \sum_{j=1}^{m_-} u(e_{ij})$$

This is the fraction of positive-negative pairs that are correctly ranked by the classifier.

**AUC ranges:**
* AUC = 1: perfect ranking
* AUC = 0.5: random guessing
* AUC = 0: completely reversed ranking
