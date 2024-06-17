# Welcome to Keras-MatMulLess's documentation!

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
api
```

Keras layers without using matrix multiplications.

This is a Keras based implementation of some layers mentioned in the paper ["BitNet: Scaling 1-bit Transformers for Large Language Models"](https://arxiv.org/pdf/2310.11453). Find the documentation [here](https://keras-matmulless.readthedocs.io/en/latest/).

Traditional, matrix multiplication based layers suffer from a few issues.

1. They have high inference and computational costs due to the use of matrix multiplications. This hinders the speed at which inference is performed on GPU-less machines.
2. The memory use for storing full precision weights is very high.
3. The energy costs of running matrix multiplications is very high.

Matrix multiplication free layers addresses these pain points by removing the key source of costs &mdash; matrix multiplications.

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
