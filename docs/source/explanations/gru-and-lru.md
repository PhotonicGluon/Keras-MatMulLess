# Recurrent Units Without Matrix Multiplications

```{article-info}
:avatar: https://avatars.githubusercontent.com/u/25820201?v=4
:avatar-link: https://github.com/PhotonicGluon/
:author: "[Ryan Kan](https://github.com/PhotonicGluon/)"
:date: "Jul 4, 2024"
:read-time: "{sub-ref}`wordcount-minutes` min read"
```

This page explains the theory behind `GRUMML` and `LRUMML`.

## Gated Recurrent Units (GRUs)

### A Quick Primer

The Long Short-Term Memory (LSTM) architecture is a staple layer in recurrent neural networks (RNNs), dating back to 1995 in the landmark paper *Long Short Term Memory* by Hochreiter and Schmidhuber. The [main LSTM paper](https://www.researchgate.net/publication/13853244) was subsequently published in 1997 in the journal *Neural Computation*, and since then minor modifications to the original LSTM implementation have been made.

The GRU was a comparatively recent addition to the RNN family by Cho et al. in the paper [*Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation*](https://arxiv.org/pdf/1406.1078v1). An empirical evaluation of GRUs against LSTMs was performed in the follow-up paper [*Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*](https://arxiv.org/pdf/1412.3555v1) by Chung et al. showed that GRUs are comparable to LSTMs on their evaluation tasks. The appeal of GRUs over LSTMs is its (relative) simplicity, using less computation and operations to achieve similar results.

We can formalize the GRU as follows[^gru-notation]. Let $n$ be the input dimension and $d$ be the model dimension (i.e., state dimension). Suppose we have a sequence of $T$ vectors, where $\mathbf{x}_t \in \mathbb{R}^n$ is the input vector at time step $t$ (where $1 \leq t \leq T$). The recurrence is thus
$$
\begin{align*}
    \mathbf{r}_t &= \sigma(\mathbf{x}_t\mathbf{W}_{xr} + \mathbf{h}_{t-1}\mathbf{W}_{hr} + \mathbf{b}_r)\\
    \mathbf{f}_t &= \sigma(\mathbf{x}_t\mathbf{W}_{xf} + \mathbf{h}_{t-1}\mathbf{W}_{hf} + \mathbf{b}_f)\\
    \mathbf{c}_t &= \tau(\mathbf{x}_t\mathbf{W}_{xc} + (\mathbf{r}_t \odot \mathbf{h}_{t-1})\mathbf{W}_{cc} + \mathbf{b}_c)\\
    \mathbf{h}_t &= \mathbf{f}_t\odot\mathbf{h}_{t-1} + (1-\mathbf{f}_t)\odot\mathbf{c}_t\\
    \mathbf{o}_t &= \mathbf{h}_t
\end{align*}
$$
where

- $\mathbf{h}_t \in \mathbb{R}^d$ is the hidden state vector at the timestep $t$ (where we define $\mathbf{h}_0 = \mathbf{0}$);
- $\mathbf{r}_t \in \mathbb{R}^d$ can be called the *reset gate vector*;
- $\mathbf{f}_t \in \mathbb{R}^d$ can be called the *forget gate vector*;
- $\mathbf{c}_t \in \mathbb{R}^d$ can be called the *candidate hidden state*;
- $\mathbf{o}_t \in \mathbb{R}^d$ is the output vector;
- $\mathbf{W}_{xr}$, $\mathbf{W}_{xf}$, and $\mathbf{W}_{xc}$ are $n \times d$ weight matrices relating to the input vector;
- $\mathbf{W}_{hr}$, $\mathbf{W}_{hf}$, and $\mathbf{W}_{vc}$ are $d \times d$ weight matrices relating to the hidden state;
- $\mathbf{b}_r$, $\mathbf{b}_f$, and $\mathbf{b}_c$ are bias vectors in $\mathbb{R}^d$;
- $\sigma$ is the sigmoid activation function;
- $\tau$ is a non-linear activation function (e.g., $\tanh$, $\mathrm{SiLU}$); and
- $\odot$ is the Hadamard product (i.e., element-wise product).

[^gru-notation]: The notation used to describe GRUs differs from the aforementioned paper by Cho et al. (and Chung et al. for that matter). We follow the notation used in [*Scalable MatMul-free Language Modeling*](https://arxiv.org/pdf/2406.02528v1) by Zhu et al. as that would make it easier to describe the changes made to implement a matmul-less version later.

One can see from the equations above that there are a lot of matrix multiplications involved in a GRU. However, note that the recurrence computing $\mathbf{h}_t$ does not involve any matrix multiplications at all, just element-wise multiplication. For a full matmul-less implementation of a GRU, we would like to keep this property while making further changes to truly remove matrix multiplications from the GRU.

### Implementing `GRUMML`

#### Standard Implementation

[*Scalable MatMul-free Language Modeling*](https://arxiv.org/pdf/2406.02528v1) by Zhu et al. proposes a few modifications to the GRU implementation in order to reduce (and then remove) matrix multiplications in the GRU layer.

1. Hidden-state related weights $\mathbf{W}_{hr}$, $\mathbf{W}_{hf}$, and $\mathbf{W}_{cc}$ are removed. This is to allow for parallelization similar to that of the Transformer architecture.
2. A data-dependent output gate $\mathbf{g}_t$ is added between the hidden state computation $\mathbf{h}_t$ and the output vector computation $\mathbf{o}_t$. Specifically, the computation for $\mathbf{o}_t$ is modified to become 
    $$
    \begin{align*}
        \mathbf{g}_t &= \sigma(\mathbf{x}_t\mathbf{W}_{xg} + \mathbf{b}_g)\\
        \mathbf{o}_t' &= \mathbf{g}_t \odot \mathbf{h}_t\\
        \mathbf{o}_t &= \mathbf{o}_t'\mathbf{W}_o + \mathbf{b}_o\\
    \end{align*}
    $$
    where we introduce
    - a new weight matrix $\mathbf{W}_{xg}$ of size $n \times d$;
    - a new weight matrix $\mathbf{W}_o$ of size $d \times d$; and
    - new bias vectors $\mathbf{b}_g, \mathbf{b}_o \in \mathbf{R}^d$.
3. All weight matrices are changed to become ternary weight matrices (i.e., the values in the matrices must be in the set $\{-1, 0, 1\}$), allowing us to use ternary multiplication and remove any matrix multiplication.

<center>
    <img alt="GRUMML" style="width: 75%" src="https://i.ibb.co/LphHHpX/gru.png">
</center>

The resulting GRU architecture (which the authors call $\mathrm{MLGRU}$ but we call `GRUMML`) can be formalized as
$$
\begin{align*}
    \mathbf{f}_t &= \sigma(\mathbf{x}_t\mathbf{W}_f + \mathbf{b}_f)\\
    \mathbf{c}_t &= \tau(\mathbf{x}_t\mathbf{W}_c + \mathbf{b}_c)\\
    \mathbf{g}_t &= \sigma(\mathbf{x}_t\mathbf{W}_g + \mathbf{b}_g)\\
    \mathbf{h}_t &= \mathbf{f}_t\odot\mathbf{h}_{t-1} + (1-\mathbf{f}_t)\odot\mathbf{c}_t\\
    \mathbf{o}_t' &= \mathbf{g}_t \odot \mathbf{h}_t\\
    \mathbf{o}_t &= \mathbf{o}_t'\mathbf{W}_o + \mathbf{b}_o\\
\end{align*}
$$
where

- $\mathbf{W}_f$, $\mathbf{W}_c$, and $\mathbf{W}_g$ are $n \times d$ ternary weight matrices;
- $\mathbf{W}_o$ is a $d \times d$ ternary weight matrix; and
- $\mathbf{b}_f, \mathbf{b}_c, \mathbf{b}_g, \mathbf{b}_o \in \mathbf{R}^d$ are bias vectors.

#### Multi-headed `GRUMML`

The main purpose for Zhu et al. to introduce a matmul-less GRU variant is to use it as a multi-headed attention replacement; instead of using multiple matrices to store information about embeddings, the $\mathrm{MLGRU}$ is a variant that uses recurrence to have a similar effect.

Let $H$ denote the number of heads to use. We will split the features contained in the forget gate $\mathbf{f}_t$ and candidate hidden state $\mathbf{c}_t$ and give it to each of the heads; in particular each head now processes $\frac dH$ features (where we assume that $d$ is a multiple of $H$). The hidden states are also split among the multiple heads. This allows for further parallelization.

## Linear Recurrent Units (LRUs)

### The Why and How of LRUs

A key issue with LSTMs and GRUs is that they are slow to optimize. This is due to their sequential nature &mdash; they rely on previous timesteps to generate the current step's output, and so the gradient propagation is slower.

In [*Resurrecting Recurrent Neural Networks for Long Sequences*](https://arxiv.org/pdf/2303.06349v1) by Orvieto et al., an alternative to the GRU was introduced, called the LRU. Referring to the traditional architecture for RNNs, they realized that replacing $\tau$ with a linear activation (i.e., not having any activation at all) actually resulted in performance *surpassing* non-linear activations for tasks such as text and retrieval tasks in Long Range Arena (LRA). In the words of the authors,

> The empirical result... is surprising, since recurrent nonlinearities are believed to be a key component for the success of RNNs &mdash; both in the theory and in practice.

<center>
    <img alt="LRU" style="width: 75%" src="https://i.ibb.co/QfLZ11X/lru.png">
</center>

To that end, they propose the LRU, using only linear activations. Crucially, the core algorithm uses complex-valued matrices to simplify the recurrence step. We describe the transformations made by the LRU formally[^lru-implementation].

[^lru-implementation]: We use a modified description of the LRU in Appendix A of the aforementioned paper.

Let the input into the layer be the sequence $\{\mathbf{u}_k\}_{k=1}^L$ of length $L$ where $\mathbf{u}_k \in \mathbb{R}^n$ (here $n$ is the input dimension). The output of the layer will be the sequence $\{\mathbf{y}_k\}_{k=1}^L$ where $\mathbf{u}_k \in \mathbb{R}^d$ (here $d$ is the output dimension), where
$$
\begin{align*}
    \mathbf{x}_k &= \mathbf{\Lambda}\mathbf{x}_{k-1} + \exp(\mathbf{\gamma}^{\mathrm{log}}) \odot (\mathbf{B}\mathbf{u}_k)\\
    \mathbf{y}_k &= \Re(\mathbf{C}\mathbf{x}_k) + \mathbf{D}\mathbf{u}_k
\end{align*}
$$

In the description above,

- $\mathbf{x}_k \in \mathbb{C}^m$ is the hidden state at $k$ (where $m$ is the state dimension), where we define $\mathbf{x}_0 = \mathbf{0}$;
- $\mathbf{\Lambda}$ is a complex-valued diagonal matrix of size $m \times m$;
- $\mathbf{\gamma}^{\mathrm{log}} \in \mathbb{R}^m$ is a vector containing the logarithms of the scaling factors (meaning that $\exp(\mathbf{\gamma}^{\mathrm{log}})$ is just the scaling factors matrix);
- $\mathbf{B}$ is a complex-valued matrix of size $n \times m$;
- $\mathbf{C}$ is a complex-valued matrix of size $m \times d$;
- $\mathbf{D}$ is a real-valued matrix of size $n \times d$; and
- $\Re$ refers to taking just the real part of the complex number (i.e., $\Re(\mathbf{C}\mathbf{x}_k)$ means to take the real parts of the values og the vector $\mathbf{C}\mathbf{x}_k$).

The fact that $\mathbf{\Lambda}$ is a diagonal matrix means that the multiplication in "$\mathbf{\Lambda}\mathbf{x}_{k-1}$" is actually element-wise. So the only matrix multiplications that occur are involving the matrices $\mathbf{B}$, $\mathbf{C}$, and $\mathbf{D}$.

### Implementing LRUs In Practice

The trouble with the implementation above is that we are using complex numbers. For some libraries (e.g., [PyTorch](https://pytorch.org/), [Jax](https://jax.readthedocs.io/)), they natively support complex numbers. For others (e.g., [Tensorflow](https://www.tensorflow.org/)), it is not easy to work with. Keras, being a backend-agnostic library that must accommodate all of these different libraries, does *not* support complex numbers. So to implement LRU with complex numbers, we have to split the complex matrices into the real and imaginary parts.

Doing so results in the following equations.
$$
\begin{align*}
    \Re(\mathbf{x}_k) &= \Re(\mathbf{\Lambda})\Re(\mathbf{x}_{k-1}) - \Im(\mathbf{\Lambda})\Im(\mathbf{x}_{k-1}) + \exp(\mathbf{\gamma}^{\mathrm{log}}) \odot (\Re(\mathbf{B})\mathbf{u}_k)\\
    \Im(\mathbf{x}_k) &= \Re(\mathbf{\Lambda})\Im(\mathbf{x}_{k-1}) + \Im(\mathbf{\Lambda})\Re(\mathbf{x}_{k-1}) + \exp(\mathbf{\gamma}^{\mathrm{log}}) \odot (\Im(\mathbf{B})\mathbf{u}_k)\\
    \mathbf{y}_k &= \Re(\mathbf{C})\Re(\mathbf{x}_k) - \Im(\mathbf{C})\Im(\mathbf{x}_k) + \mathbf{D}\mathbf{u}_k
\end{align*}
$$
where $\Re$ refers to the real component of the complex number and $\Im$ refers to the imaginary component of the complex number.

In addition, storing the full values of $\mathbf{\Lambda}$, which are actually the eigenvalues of the linear transformation, is not efficient and could use a lot of memory. In practice we appeal to the fact that every complex number $a + b\mathrm{i}$ has a polar representation $\nu\mathrm{e}^{\mathrm{i}\theta}$ where $\nu \geq 0$ and $\theta \in \mathbb{R}$. So we will use the log of $\nu$ (given by the matrix $\mathbf{\nu}^\mathrm{log}$) and the log of $\theta$ (given by the matrix $\mathbf{\theta}^\mathrm{log}$) as the weights for the network to learn.

### Implementing `LRUMML`

Unlike `GRUMML` which had to use a wholly different architecture to the original GRU, the matmul-less version of LRU simply replaces a few of the matrices with ternary weights. In particular, the values of $\mathbf{B}$, $\mathbf{C}$, and $\mathbf{D}$ can be restricted to be ternary values, and thus we can use ternary multiplications to replace the original matrix multiplications.

## One Last Catch

Despite the above implementations of `GRUMML` and `LRUMML` above being fully matmul-less, there is a problem &mdash; the outputs are also quantized. This limitation means that the output of the two layers are not as good as their normal counterparts.

Thus, the fully matmul-less implementations of both of these layers are **disabled by default**. We only partially use ternary weights for the layers; in particular

- only $\mathbf{W}_f$ and $\mathbf{W}_c$ use ternary weights in `GRUMML`; and
- only $\mathbf{B}$ uses ternary weights in `LRUMML`.

Of course, fully matmul-less implementations of the two layers are still available, but they must be enabled via setting `fully_mml` to `True`.
