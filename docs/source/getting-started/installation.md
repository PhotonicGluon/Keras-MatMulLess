# Installation

For now, the only way to install Keras-MML is via GitHub.

The requirements for the package are:

- Python 3.9 (and above)

## Installation via GitHub

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
poetry install --with dev --extras BACKEND_NAME
```
