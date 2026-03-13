# Basic Concepts

## Two Basic Paradigms

| Paradigm | Data | Goal | Typical Tasks |
|----------|------|------|---------------|
| **Supervised Learning** | Input–label pairs $\{(x_i, y_i)\}_{i=1}^n$ | Learn a mapping $x \mapsto y$ | Classification, Regression |
| **Unsupervised Learning** | Unlabeled feature vectors $\{x_i\}_{i=1}^n$ | Discover underlying structure | Clustering, Dimensionality Reduction |

## Target Function, Hypothesis, and Hypothesis Space

| Concept | Definition |
|---------|-----------|
| **Target function** $t : \mathcal{X} \to \mathcal{Y}$ | The unknown ground-truth mapping that generates the data: $y_i = t(x_i) + \text{noise}$. We only observe finite data, not this function. |
| **Hypothesis** $h$ | A candidate function used to approximate $t$. Example: $h(x) = \theta_1 x_1 + \theta_2 x_2 = \boldsymbol{\theta}^\top x$. |
| **Hypothesis space** $\mathcal{H}$ | The set of all allowable hypotheses. Example: $\mathcal{H} = \{h_{\mathbf{w}}(x) = \mathbf{w}^\top x : \mathbf{w} \in \mathbb{R}^n\}$. |

## Cost Function, Training, and Testing

**Cost (objective) function:** measures how well a hypothesis fits the training data. Using squared loss as an example:

$$J(h) = \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_\text{train}} \bigl(h(x_i) - y_i\bigr)^2$$

**Training (learning):** find the best hypothesis in $\mathcal{H}$ by minimizing the cost on the training set (optimization):

$$h^* = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_\text{train}} \bigl(h(x_i) - y_i\bigr)^2$$

**Testing (evaluation):** assess the performance of the learned model $h^*$ on the test set:

$$\text{Test error} = \frac{1}{m} \sum_{(x_i, y_i) \in \mathcal{D}_\text{test}} \bigl(h^*(x_i) - y_i\bigr)^2$$

## General Machine Learning Workflow

1. **Data collection:** gather relevant historical data.
2. **Data preprocessing:** handle missing values, outliers, and data quality issues.
3. **Model design:** choose the hypothesis space $\mathcal{H}$, objective function $J$, and optimization method.
4. **Training:** learn model parameters from training data.
5. **Testing:** evaluate performance on unseen test data.
6. **Model improvement:** refine the model, features, or data to improve performance.

## Other Learning Paradigms

| Paradigm | Description |
|----------|-------------|
| **Reinforcement Learning (RL)** | An agent interacts with an environment over time, taking actions to maximize expected cumulative reward. At each step $t$: observe state $s_t$, take action $a_t$, receive reward $r_t$, transition to $s_{t+1}$. |
| **Semi-supervised learning** | Learning from a mixture of labeled and unlabeled data, typically when labeled data is scarce. |
| **Ensemble learning** | Combining predictions from multiple models to obtain a final decision, often improving robustness and accuracy. |
| **Transfer learning** | Leveraging knowledge learned from a source domain to improve learning in a target domain, especially when target data is limited. |
| **Federated learning** | Training models across distributed clients using local data, sharing only model updates (not raw data) — privacy-preserving. |
| **Machine unlearning** | Removing the influence of specific training samples from a trained model, motivated by privacy and regulatory requirements. |
