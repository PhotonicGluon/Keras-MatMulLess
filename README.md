# Keras-MatMulLess (Keras-MML)

[![CodeCov](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess/graph/badge.svg?token=VKD0CJX1SD)](https://codecov.io/gh/PhotonicGluon/Keras-MatMulLess)
[![ReadTheDocs](https://readthedocs.org/projects/keras-matmulless/badge/?version=latest)](https://keras-matmulless.readthedocs.io/en/latest/?badge=latest)

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.
> <div style="text-align: right">&mdash; Noam Shazeer, in <a href="https://arxiv.org/pdf/2002.05202v1"><em>GLU Variants Improve Transformer</em></a></div>

<!-- start summary -->
Keras layers without using matrix multiplications.

This is a Keras based implementation of some layers mentioned in the papers [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/pdf/2402.17764v1) and [*Scalable MatMul-free Language Modeling*](https://arxiv.org/pdf/2406.02528v5). <!-- end summary --> Find the documentation [here](https://keras-matmulless.readthedocs.io/en/latest/).

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

Read the [tutorial](https://keras-matmulless.readthedocs.io/en/latest/getting-started/tutorial.html).

## Using Dev Containers

[![CPU Dev Container](https://img.shields.io/static/v1?label=CPU%20Dev%20Container&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PhotonicGluon/Keras-MatMulLess)

If you already have VS Code and Docker installed, you can click the badge above or [here](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PhotonicGluon/Keras-MatMulLess) to get started. Clicking these links will cause VS Code to automatically install the Dev Containers extension if needed, clone the source code into a container volume, and spin up a dev container for use.

### CUDA

Keras-MML offers a `cuda` dev container for working with CUDA.

> [!IMPORTANT]  
> Edit the [`Dockerfile`](.devcontainer/cuda/Dockerfile) file to set up the architecture properly.
> By default it is using `amd64`. So, if you are on a `arm64` system, **uncomment the appropriate lines in the file**!

## License

Keras-MML is licensed under the MIT license.

```plain
MIT License

Copyright (c) 2024 Ryan Kan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
