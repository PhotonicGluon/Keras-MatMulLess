# Keras-MatMulLess (Keras-MML)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/keras-matmulless?logo=python)](https://pypi.org/project/keras-matmulless/)
[![PyPI - Version](https://img.shields.io/pypi/v/keras-matmulless?label=pypi%20(stable)&logo=pypi)](https://pypi.org/project/keras-matmulless/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-matmulless)](https://pypi.org/project/keras-matmulless/)
[![PyPI - License](https://img.shields.io/pypi/l/keras-matmulless)](LICENSE)

[![Read the Docs - Stable](https://img.shields.io/readthedocs/keras-matmulless?label=docs%20(stable)&logo=readthedocs)](https://keras-matmulless.readthedocs.io/en/stable/)
[![Read the Docs - Latest](https://img.shields.io/readthedocs/keras-matmulless?label=docs%20(latest)&logo=readthedocs)](https://keras-matmulless.readthedocs.io/en/latest/)

[![CodeCov](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess/graph/badge.svg?token=VKD0CJX1SD)](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess)
[![Code Style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](.github/CODE_OF_CONDUCT.md)

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.
> <div style="text-align: right">&mdash; Noam Shazeer, in <a href="https://arxiv.org/pdf/2002.05202v1"><em>GLU Variants Improve Transformer</em></a></div>

<!-- start summary -->
Keras layers without using matrix multiplications.

This is a Keras based implementation of some layers mentioned in the papers [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/pdf/2402.17764v1) and [*Scalable MatMul-free Language Modeling*](https://arxiv.org/pdf/2406.02528v5). <!-- end summary --> Find the documentation [here](https://keras-matmulless.readthedocs.io/).

## Rationale
<!-- start rationale -->

Traditional, matrix multiplication based layers suffer from a few issues.

1. They have high inference and computational costs due to the use of matrix multiplications. This hinders the speed at which inference is performed on GPU-less machines.
2. The memory use for storing full precision weights is very high.
3. The energy costs of running matrix multiplications is very high.

Matrix multiplication free layers addresses these pain points by removing the key source of costs &mdash; matrix multiplications.

<!-- end rationale -->

## Installation
<!-- start installation -->

### Requirements

Keras-MML has a few requirements, namely

- Python 3.9 (or above);
- Keras; and
- the Keras backend (either [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), or [Jax](https://jax.readthedocs.io/en/latest/index.html)).

Instructions on how to install Keras can be found [here](https://keras.io/getting_started/).

### Installation Instructions

#### PyPi

If you use pip, you can install Keras-MML using the command

```bash
pip install keras-matmulless
```

##### Pre-Release Versions

To install pre-release versions, use the command

```bash
pip install --pre keras-matmulless
```

##### Nightly Versions

Nightly releases for Keras-MML are primarily found on the [TestPyPi](https://test.pypi.org/project/keras-matmulless/) page. To install them, use the command

```bash
pip install -i https://test.pypi.org/simple/ keras-matmulless
```

#### Building From Scratch

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

Finally, install the development dependencies. The development dependencies are split into several groups.

- The `test` group contains dependencies that are used to perform testing.
- The `docs` group contains dependencies that are used to generate the documentation.
- The `build` group contains dependencies that are used to create a distributable.
- The `notebook` group is required to run the Jupyter notebooks in the documentation folder.

Simply include the desired groups in the `install.py` call. For example, to install `test`, `docs`, and `build` (the main development dependencies), run the following command.

```bash
python install.py test docs build
```

If you have not installed a backend (i.e., [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), or [Jax](https://jax.readthedocs.io/en/latest/index.html)) you can do so here.

```bash
python install.py test docs build --backend BACKEND_NAME
```

Note that the `BACKEND_NAME` to be specified here is

- `tensorflow` for the Tensorflow backend;
- `torch` for the PyTorch backend; and
- `jax` for the Jax backend.

If you need to install with CUDA support, run

```bash
python install.py test docs build --backend BACKEND_NAME --with-cuda
```

That's it! You should now have access to the `keras_mml` package.

<!-- end installation -->

## Quickstart

Read the [tutorial](https://keras-matmulless.readthedocs.io/en/stable/getting-started/tutorial.html).

## Contributing

We welcome contributions! Please read more about contributing to Keras-MML in the [contribution guidelines](.github/CONTRIBUTING.md).

## License

Keras-MML is licensed under the Apache 2.0 license.
