# Recurrent Neural Networks and Transformers

## 1. Sequential Data

Unlike MLPs and CNNs which process fixed-size inputs, sequential data requires modeling temporal or positional dependencies.

| Task | Input | Output | Example |
|------|-------|--------|---------|
| Time series prediction | Sequence | Scalar | Stock price forecasting |
| Sequence-to-sequence | Sequence | Sequence | Machine translation |
| Audio-to-text | Sequence | Sequence | Speech recognition |
| Image-to-text | Image | Sequence | Image captioning |

---

## 2. Recurrent Neural Networks (RNN)

### Basic RNN Architecture

For a sequence $(x_1, x_2, \dots, x_T)$, the RNN maintains a hidden state $h_t$ that carries information from previous time steps:

$$h_t = f_W(h_{t-1}, x_t), \quad \hat{y}_t = g_{W'}(h_t)$$

where $f_W$ is typically:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

* The **same parameters** $W$ are shared across all time steps.
* Allows processing variable-length inputs.
* Computation is sequential (not parallelizable).

### Training: Backpropagation Through Time (BPTT)

The total loss over a sequence of length $T$:

$$E(\theta) = \frac{1}{T} \sum_{t=1}^{T} L(y_t, \hat{y}_t)$$

Gradients flow backward through the unrolled computational graph. For long sequences, this requires storing all hidden states.

**Truncated BPTT:** Carry hidden states forward in time forever, but only backpropagate for a smaller number of steps.

### Vanishing and Exploding Gradients

The gradient with respect to $h_t$ involves repeated multiplications by $W_{hh}^\top$ and derivatives of $\tanh$:

$$\frac{\partial h_T}{\partial h_t} = \prod_{\tau=t+1}^{T} W_{hh}^\top \, \text{diag}\bigl(\tanh'(\cdot)\bigr)$$

Since $\tanh' \in [0, 1]$:
* **Gradient vanishing:** The product of many small terms becomes extremely small, limiting the network's ability to capture long-range dependencies.
* **Gradient exploding:** If the largest singular value of $W_{hh}$ is large, the product grows exponentially.

**Mitigation:**
* Gradient exploding $\rightarrow$ gradient clipping.
* Gradient vanishing $\rightarrow$ architectural modifications (LSTM, GRU).

---

## 3. Long Short-Term Memory (LSTM)

Introduced by Hochreiter and Schmidhuber (1997) to address vanishing gradients.

### LSTM Cell Equations

At each time step, the LSTM maintains a **cell state** $C_t$ (long-term memory) and a **hidden state** $h_t$ (short-term output):

**Forget gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

**Input gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

**Candidate cell state:** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

**Cell state update:** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

**Output gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

**Hidden state:** $h_t = o_t \odot \tanh(C_t)$

where $\odot$ denotes element-wise multiplication.

### Key Properties

* **Gated structure:** The forget gate controls what information to discard; the input gate controls what new information to store.
* **Additive updates:** The cell state is updated by addition ($C_t = f_t \odot C_{t-1} + \dots$), which helps preserve gradients over long sequences.
* Mitigates (but does not completely eliminate) vanishing/exploding gradients.
* Widely adopted in deep learning around 2013–2015.

---

## 4. Gated Recurrent Units (GRU)

A simpler variant introduced by Cho et al. (2014):

**Reset gate:** $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$

**Update gate:** $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$

**Candidate hidden state:** $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$

**Hidden state:** $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

* Similar to LSTM but with fewer parameters (lacks a separate output gate and cell state).
* Often comparable performance to LSTM with less computation.

---

## 5. Limitations of RNNs

| Limitation | Description |
|------------|-------------|
| **Long-term dependencies** | Still struggle with very long sequences due to vanishing gradients |
| **Limited parallelization** | Sequential processing prevents parallel computation, leading to slower training |

These limitations motivated the development of the **Transformer** architecture.

---

## 6. Attention Mechanism

### Core Idea

The goal of attention is to transform input token representations $\{x_1, \dots, x_n\}$ into richer representations $\{y_1, \dots, y_n\}$ where each output depends on **all** tokens:

$$y_i = \sum_{j=1}^{n} a_{ij} x_j$$

where $a_{ij} \ge 0$ and $\sum_{j=1}^{n} a_{ij} = 1$ are attention coefficients.

### Self-Attention

**Idea:** Measure similarity between tokens and convert into probabilities via softmax.

$$a_{ij} = \frac{\exp(x_i^\top x_j)}{\sum_{j'=1}^{n} \exp(x_i^\top x_{j'})}$$

**Matrix form:** Stack outputs into $Y \in \mathbb{R}^{n \times d}$:

$$A = \text{softmax}(XX^\top), \quad Y = AX$$

where softmax is applied row-wise. Each row of $A$ sums to 1.

### Trainable Self-Attention (Scaled Dot-Product)

Introduce learnable linear transformations:

$$Q = XW^{(q)}, \quad K = XW^{(k)}, \quad V = XW^{(v)}$$

where $W^{(q)}, W^{(k)} \in \mathbb{R}^{d \times d_k}$ and $W^{(v)} \in \mathbb{R}^{d \times d_v}$.

$$\text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

* **Query ($Q$):** The token seeking information.
* **Key ($K$):** What each token offers.
* **Value ($V$):** The actual information content.
* **Scaling by $\sqrt{d_k}$:** Prevents large inner products from producing unstable gradients.

### Multi-Head Attention

Different tokens may have different types of relationships. Use $H$ parallel attention heads:

$$H_h = \text{attention}(Q_h, K_h, V_h), \quad h = 1, \dots, H$$

Concatenate and apply an output transformation:

$$Y = [H_1, \dots, H_H] W^{(o)}$$

Usually $H \cdot d_v = d$ so the output dimension matches the input.

---

## 7. Attention Masks

Without restriction, token $i$ may attend to all tokens $j = 1, \dots, n$. We need to restrict invalid interactions.

### Definition

Introduce mask matrix $M \in \mathbb{R}^{n_q \times n_k}$:

$$\text{attention}(Q, K, V; M) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where:

$$M_{ij} = \begin{cases} 0, & \text{allowed} \\ -\infty, & \text{blocked} \end{cases}$$

Since $\exp(-\infty) = 0$, blocked positions receive zero attention weight.

### Padding Mask

In a mini-batch, sequences have different lengths. Shorter sequences are padded. Padded tokens should not affect the output.

Define padding indicator $p_j \in \{0, 1\}$ where $p_j = 1$ if position $j$ is padding. Then:

$$M^{\text{pad}}_{ij} = \begin{cases} -\infty, & p_j = 1 \\ 0, & p_j = 0 \end{cases}$$

### Causal Mask

In autoregressive modeling (next-token prediction), token $i$ must not attend to future tokens $j > i$:

$$M^{\text{causal}}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$

The $i$-th row can only place attention mass on positions $j \le i$. Both masks are combined: $M = M^{\text{causal}} + M^{\text{pad}}$.

---

## 8. Computational Complexity of Attention

| Step | Complexity |
|------|-----------|
| Score matrix $S = QK^\top$ | $O(n^2 d)$ |
| Softmax + storage | $O(n^2)$ time, $O(n^2)$ memory |
| Output $Y = \text{softmax}(S)V$ | $O(n^2 d)$ |
| **Overall** | **$O(n^2 d)$ — quadratic in sequence length $n$** |

This quadratic complexity is a key bottleneck for very long sequences.

---

## 9. Transformer Architecture

### Tokenization and Embedding

* **Tokenization:** Convert text into tokens $(t_1, \dots, t_n)$. Use subword tokenization (e.g., BPE) to balance vocabulary size and sequence length.
* **Embedding:** Map each token $t_i$ to a vector $x_i = E^\top e_{t_i}$ where $E \in \mathbb{R}^{V \times d}$ is a learned embedding matrix and $e_{t_i}$ is a one-hot vector.

### Positional Encoding

Since self-attention processes tokens as a set (order-agnostic), we inject positional information:

$$\tilde{x}_i = x_i + p_i$$

where $p_i$ is a position encoding vector. This can be either learned or fixed (e.g., sinusoidal).

### Transformer Layer

A transformer layer combines four components:

1. **Multi-head self-attention**
2. **Residual connection**
3. **Normalization**
4. **Feed-forward network (MLP)**

**Post-Norm (original Transformer):**
$$Z = \text{Norm}(Y(X) + X), \quad X = \text{Norm}(\text{MLP}(Z) + Z)$$

**Pre-Norm (modern LLMs like GPT, LLaMA):**
$$Z = Y(\text{Norm}(X)) + X, \quad X = \text{MLP}(\text{Norm}(Z)) + Z)$$

Pre-Norm is more stable for very deep networks.

### Layer Normalization (LayerNorm)

For a token representation $x = [x_1, \dots, x_d]$:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$$

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$

where $\gamma, \beta$ are learnable scale and shift parameters. Normalizes each token independently (across the feature dimension).

### RMS Normalization (RMSNorm)

Simpler alternative without mean centering:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}, \quad \hat{x}_i = \frac{x_i}{\text{RMS}(x)}, \quad y_i = \gamma_i \hat{x}_i$$

* Only scaling, no shifting.
* Fewer parameters, more efficient.
* Used in modern architectures like LLaMA.

---

## 10. Decoder-Only Transformer

Used in GPT, LLaMA, etc.:

* Stack multiple masked transformer layers.
* Each layer uses causal self-attention.
* Add embedding layer at input and projection + softmax at output for next-token prediction.

The key insight of the Transformer (Vaswani et al., 2017) is that **attention is all you need** — recurrence and convolutions can be entirely replaced by attention mechanisms, enabling greater parallelization and faster training.
