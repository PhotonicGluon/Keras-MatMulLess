# Contribution Guidelines

> [!NOTE]
> This document was **largely adapted** from the [Keras Contribution Guidelines](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md). The Keras repository is licensed under the Apache-2.0 license (which is the same as Keras-MML). You can obtain a copy of their license [here](https://github.com/keras-team/keras/blob/master/LICENSE).

**We welcome contributions!**

Contributions can be made in a variety of ways, including coding, enriching documentation, and refining docstrings.

## How to Contribute

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already exist) and discussing your proposed changes. This way, we can give you feedback and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free to open a Pull Request (PR) without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a development environment and run the unit tests. This is covered in the "[Setup Environment](#setup-environment)" section.

### Step 3. Create a pull request

Once the change is ready, [open a pull request](https://github.com/PhotonicGluon/Keras-MatMulLess/pulls) from your branch in your fork to the master branch.

### Step 4. Code Tests and Review

Upon creating a pull request, automated tests will be performed. If the tests fail, look into the error messages and try to fix them.

A reviewer will review the pull request and provide comments. There may be several rounds of comments and code changes before the pull request gets approved by the reviewer.

### Step 5. Merging

Once the pull request is approved, a ready to pull tag will be added to the pull request. A team member will take care of the merging.

## Setup Environment

We provide two ways of setting up a development environment. One is to use a dev container, and the other one is to set up a local environment by installing the dev tools needed.

### Option 1. GitHub Codespaces or Dev Container

We support GitHub Codespaces, Visual Studio Code dev containers and JetBrains dev containers. Please see the [Dev container documentation](../.devcontainer/README.md).

### Option 2. Set Up A Local Environment

Setting up a local environment is similar to [Building From Scratch](../README.md#building-from-scratch). Follow the instructions there.

## Code Organization

### Package Management

Keras-MML uses [Poetry](https://python-poetry.org/) for dependency management. It can be installed in a Python virtual environment with the command

```bash
pip install poetry
```

### Code Style

Keras-MML uses the [Black](https://github.com/psf/black) formatter to help nicely format the code. [isort](https://pycqa.github.io/isort/) is also used to sort the imports in Python files. The [Ruff](https://github.com/astral-sh/ruff) linter is used for fast linting and formatting.

To check that the code is formatted properly, run the following command **at the root directory of the repo**.

```bash
ruff check
```

To then format the code, run

```bash
ruff format
```

### Documentation

We do not have an automated way to check docstrings' style, so if you write or edit any docstring, please make sure to check them manually. Keras-MML docstrings follow the conventions below.

#### Class Docstring

A class's docstring may contain the following items.

- A one-line description of the class.
- Some paragraph(s) of more detailed information.
- An `Attributes` section listing the *public* attributes of the class.
  - Hidden layers and weights should *not* be described.

Documentation for the initialization function (i.e., `__init__()`) should be **included under that function, not in the class's docstring**.

You can check out [`DenseMML`](https://github.com/PhotonicGluon/Keras-MatMulLess/blob/2c98eca8bc254e46b7fa799fe3468dbddcff2b7c/keras_mml/layers/core/dense.py#L22) as an example.

#### Function Docstring

A function's arguments and return values should be annotated with an appropriate type, if possible.

A function's docstring may contain the following items.

- A one-line description of the function.
- Some paragraph(s) of more detailed information.
- An `Args` section for the function arguments.
  - If the function accepts `*args` and/or `**kwargs`, you **must include them in the docstring**.
- An optional `Raises` section for possible errors.
- A `Returns` section for the return values.
- An optional `Examples` section.

You can check out [`decode_ternary_array`](https://github.com/PhotonicGluon/Keras-MatMulLess/blob/2c98eca8bc254e46b7fa799fe3468dbddcff2b7c/keras_mml/utils/array/encoding.py#L94) as an example.

## Running Tests

We use [pytest](http://pytest.org/) to run tests. The tests are located in the `tests` directory.

### Run a Test File

To run the tests in, say, `tests/utils/array/test_encoding.py`, use the following command at the root directory of the repo.

```bash
pytest tests/utils/array/test_encoding.py
```

> [!NOTE]
> If that command does not work, try
> 
> ```bash
> poetry run pytest tests/utils/array/test_encoding.py
> ```
> 
> instead. This will force pytest to be run in the installed Poetry environment.

### Run All Tests

You can run all the tests locally by running the following command in the repo root directory.

```bash
pytest .
```

To speed up the testing process, you can choose to use multiple workers to run the tests by running

```bash
pytest . -n auto
```

To run all tests using a different backend (e.g. `jax` instead of the default `tensorflow`), you can simply specify it on the command line.

```bash
KERAS_BACKEND=jax pytest . -n auto
```

#### Coverage

To measure coverage by the tests, run

```bash
pytest . --cov
```

To generate a report of the coverage (e.g., in `html` format), run

```bash
pytest . --cov --cov-report=html
```
