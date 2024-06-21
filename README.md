# Keras-MatMulLess

[![CodeCov](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess/graph/badge.svg?token=VKD0CJX1SD)](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess)
[![ReadTheDocs](https://readthedocs.org/projects/keras-matmulless/badge/?version=latest)](https://keras-matmulless.readthedocs.io/en/latest/?badge=latest)

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.
> <div style="text-align: right">&mdash; Noam Shazeer, in <a href="https://arxiv.org/pdf/2002.05202"><em>GLU Variants Improve Transformer</em></a></div>

<!-- start summary -->
Keras layers without using matrix multiplications.

This is a Keras based implementation of some layers mentioned in the papers ["BitNet: Scaling 1-bit Transformers for Large Language Models"](https://arxiv.org/pdf/2310.11453), ["The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"](https://arxiv.org/pdf/2402.17764), and ["Scalable MatMul-free Language Modeling"](https://arxiv.org/pdf/2406.02528). <!-- end summary --> Find the documentation [here](https://keras-matmulless.readthedocs.io/en/latest/).

## Rationale
<!-- start rationale -->

Traditional, matrix multiplication based layers suffer from a few issues.

1. They have high inference and computational costs due to the use of matrix multiplications. This hinders the speed at which inference is performed on GPU-less machines.
2. The memory use for storing full precision weights is very high.
3. The energy costs of running matrix multiplications is very high.

Matrix multiplication free layers addresses these pain points by removing the key source of costs &mdash; matrix multiplications.

<!-- end rationale -->

## Installation
<!-- start installation overview -->

For now, the only way to install Keras-MML is via GitHub.

The requirements for the package are:

- Python 3.9 (and above)

<!-- end installation overview -->

### Installation via GitHub
<!-- start installation GitHub -->

First, clone the repository using

```bash
git clone https://github.com/PhotonicGluon/Keras-MatMulLess.git
cd Keras-MatMulLess
```

We recommend to create a virtual environment to install [Poetry](https://python-poetry.org/) and the other dependencies into.

```bash
python -m venv venv  # If `python` doesn't work, try `python3`
```

Activate the virtual environment using

```bash
source venv/bin/activate
```

or, if you are on Windows,

```bash
venv/Scripts/activate
```

Now we install Poetry.

```bash
pip install poetry
```

Finally, install the development dependencies.

```bash
poetry install --with dev
```

If you have not installed a backend (i.e., [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), or [Jax](https://jax.readthedocs.io/en/latest/index.html)) you can do so here.

```bash
poetry install --with dev,BACKEND_NAME
```

Note that the `BACKEND_NAME` to be specified here is

- `tensorflow` for the Tensorflow backend;
- `torch` for the PyTorch backend; and
- `jax` for the Jax backend.

That's it! You should now have access to the `keras_mml` package.

<!-- end installation GitHub -->

## Quickstart

See the [tutorial notebook](docs/source/getting-started/tutorial.ipynb).
