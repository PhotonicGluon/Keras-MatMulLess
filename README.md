# Keras-MatMulLess

[![CodeCov](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess/graph/badge.svg?token=VKD0CJX1SD)](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess)
[![ReadTheDocs](https://readthedocs.org/projects/keras-matmulless/badge/?version=latest)](https://keras-matmulless.readthedocs.io/en/latest/?badge=latest)

Keras layers without using matrix multiplications.

This is a Keras based implementation of some layers mentioned in the paper ["BitNet: Scaling 1-bit Transformers for Large Language Models"](https://arxiv.org/pdf/2310.11453). Find the documentation [here](https://keras-matmulless.readthedocs.io/en/latest/).

Traditional, matrix multiplication based layers suffer from a few issues.

1. They have high inference and computational costs due to the use of matrix multiplications. This hinders the speed at which inference is performed on GPU-less machines.
2. The memory use for storing full precision weights is very high.
3. The energy costs of running matrix multiplications is very high.

Matrix multiplication free layers addresses these pain points by removing the key source of costs &mdash; matrix multiplications.

## Installation

For now, the only way to install Keras-MML is via GitHub.

The requirements for the package are:

- Python 3.9 (and above)

### Installation via GitHub

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

Now we install Poetry and the development dependencies. If you have not installed a backend (i.e., Tensorflow, PyTorch, or Jax) you can do so here.

```bash
# Installing poetry
pip install poetry

# Without backend
poetry install --with dev

# With backend
poetry install --with dev --extras BACKEND_NAME
```
