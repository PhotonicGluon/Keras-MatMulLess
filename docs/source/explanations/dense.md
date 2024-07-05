# Implementing Matrix Multiplication Free `Dense` Layers

```{article-info}
:avatar: https://avatars.githubusercontent.com/u/25820201?v=4
:avatar-link: https://github.com/PhotonicGluon/
:author: "[Ryan Kan](https://github.com/PhotonicGluon/)"
:date: "Jun 24, 2024"
:read-time: "{sub-ref}`wordcount-minutes` min read"
```

This page explains the theory behind `DenseMML`, as well as important pitfalls one may encounter when using them in an actual model.

## How `DenseMML` Works

The core implementation of `DenseMML` stems from the paper [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/pdf/2402.17764) by Ma and Wang et al.

<center>
    <img alt="Figure 1" style="width: 75%" src="https://i.postimg.cc/SxcSzjCQ/figure-1.png">
</center>

The main idea is to replace (16-bit) floating point values in the weight matrices with *ternary weights*. These weights consist only of the integers $\{-1, 0, 1\}$ and so require way less memory to process (specifically, $\log_2(3) \approx 1.58$ bits). More importantly, when we eventually perform matrix multiplication in `Dense` layers, the operation performed is, in essence, only addition. This dramatically reduces the cost of inference as we do not waste processing power on multiplications.

### Weight Quantization

To describe the weight quantization step, define $\mathbf{W}$ to be the weights matrix with $m$ rows and $n$ columns.

To constrain the weights' values to $-1$, $0$, or $1$, Ma and Wang et al. adopted an *absmean* quantization function. They first computed the mean of the absolute values (i.e., the absmean) of the weight values.
$$
\gamma = \frac1{mn} \sum_{i,j} \left|\mathbf{W}_{i,j}\right|
$$

Then, with this value, they are able to round each weight value in the weight matrix $\mathbf{W}$ to the nearest integer amongst $\{-1, 0, 1\}$ using the formula
$$
\tilde{\mathbf{W}} = \mathrm{RoundClip}\left(\frac1{\gamma + \epsilon}\mathbf{W}, -1, 1\right)
$$
where

- the $\mathrm{RoundClip}$ function is defined by $$\mathrm{RoundClip}(x, a, b) = \max(a, \min(b, \mathrm{round}(x)))$$ for $a \leq b$;
- $\gamma$ is the absmean of the values in $\mathbf{W}$ (see above); and
- $\epsilon$ is a small value (e.g, $10^{-5}$) to avoid division by zero errors.

### Activation Quantization

Quantization of the activation values largely follows previous work from Ma and Wang et al. in [*BitNet: Scaling 1-bit Transformers for
Large Language Models*](https://arxiv.org/pdf/2310.11453v1).

In particular, for the activation values $\mathbf{x}$ being fed into the layer, quantization is performed via the formula
$$
\tilde{\mathbf{x}} = \mathrm{Clip}\left(\frac{Q_b}{\beta}\mathbf{x}, -Q_b+\epsilon, Q_b-\epsilon\right)
$$
where

- $b$ is the number of bits to quantize the activations to, and so $Q_b = 2^{b-1}$;
- the $\mathrm{Clip}$ function is defined by $$\mathrm{Clip}(x, a, b) = \max(a, \min(b, x))$$ for $a \leq b$;
- $\beta$ is the maximum of the absolute values of the components of $\mathbf{x}$; and
- $\epsilon$ is the same small value as above to avoid division by zero errors.

### Computing the Dense Output

With the quantized weights $\tilde{\mathbf{W}}$ and quantized activations $\tilde{\mathbf{x}}$, the result of the dense layer is simply
$$
\mathbf{y} = \tilde{\mathbf{W}}\tilde{\mathbf{x}} \times \frac{\beta\gamma}{Q_b}
$$
where $\beta$, $\gamma$, and $Q_b$ are as described above.

The careful reader might notice that this formulation, as it stands, has a huge issue when attempting to compute the gradients for updating the weights. Since the $\mathrm{Clip}$ and $\mathrm{RoundClip}$ functions do not have well-defined continuous gradients, attempting to perform gradient updates like this would cause serious degradation in performance. Thus, to train the model, a straight-through estimator (STE) trick is employed during backpropagation to bypass the non-differentiable functions during the backward pass.

### Deviations in Our Implementation

One notices that what Ma and Wang describes is tantalizingly close to the actual `Dense` layer used in Keras &mdash; all that we are missing now is ability to customize the activation function and including a bias term.

Adding a bias term is remarkably simple since we just need to add an additional weight to the layer that stores the bias vector. This bias vector will be stored in full precision (i.e., not quantized to 1.58 bits) in order to not reduce the effectiveness of the layer. Activation function is then applied to the combined sum.

## Pitfalls

Although the matrix multiplication free dense layers act as suitable replacements to the regular `Dense` layers in keras, there are a few pitfalls that one needs to be aware of when using them.

### `DenseMML` Layers *Should Not* be the Final Layer(s) of the Model

Since `DenseMML` quantizes the weights to be in the set $\{-1, 0, 1\}$, attempting to use them as the final layer when the output could vary a lot would lead to the model failing to converge properly.

Consider the following classifier.

```python
model = Sequential(
    [
        Input(shape=INPUT_SHAPE),
        Flatten(),
        DenseMML(256),
        DenseMML(256),
        DenseMML(256),
        Dense(NUM_CLASSES, activation="softmax"),  # <--- Classification head
    ]
)
```

If the final layer of the classifier was to be a `DenseMML` layer, then the final output of the model would *likely* be quantized to be around 1.58 bits. This level of precision is woefully insufficient for classification where the precision in the output probabilities matters a lot.

However, there may be some instances that make `DenseMML` suitable to be the last layer, such as when precision does not matter too much. In these cases, it should be *okay* to use `DenseMML` as the last layer.

### `DenseMML` Layers Cannot Be Trained After Saving

When saving `DenseMML` layers, we save the quantized weights in order to reduce memory use. This is very good &mdash; the model size would be reduced by a factor of at least 4. However, this reduction of storage requirements comes at the cost of not being able to be (re-)trained or fine-tuned once the model is saved.
