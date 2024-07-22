# Changelog

All notable changes to Keras-MatMulLess will be documented here.

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [*towncrier*](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found [here](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/main/docs/release/upcoming_changes).

<!-- towncrier release notes start -->

## [0.1.0](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0) - 2024-07-22

No significant changes.


## [0.1.0rc2](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0rc2) - 2024-07-19

### Documentation Changes

- Added missing call conventions for `GRU` and `LRU`.

### Miscellaneous Changes

- Updated the versions of the dependencies listed in `poetry.lock`.


## [0.1.0rc1](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0rc1) - 2024-07-18

### Miscellaneous Changes

- Cleaned up the `install.py` script.
- Fixed `Input 'repository_url' has been deprecated with message: The inputs have been normalized to use kebab-case.` warning in GitHub actions.
- Removed useless setup code for Ruff linting action.


## [0.1.0b1](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0b1) - 2024-07-11

### Changes

- Changed linter from PyLint to Ruff.
- Loosened dependency requirements to allow for more compatibility with other packages.
- Reorganized poetry dependency groups.

### Fixes

- Added missing `scikit-learn` dependency for the `notebook` Poetry group.
- Fixed missing type annotation for `_quantize_kernel()` in `BaseDenseMML`.

### Miscellaneous Changes

- Bump GitHub actions' versions. ([#1](https://github.com/PhotonicGluon/Keras-MatMulLess/issues/1))
- Added a security policy, code of conduct, and contribution guidelines.
- Added new training tests to GitHub actions to ensure that the changes does not affect performance and results of the models.
- Configured dependabot to look out for dependency updates.
- Made devcontainer settings become VSCode workspace settings.
- Renamed and redid some GitHub actions.


## [0.1.0a3](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0a3) - 2024-07-10

### Removals

- Removed Python 3.12 support (to be in line with [Keras' Python supported versions](https://pypi.org/project/keras/3.3.3/))

### New Features

- Added stricter call annotation using the `jaxtyping` package.

### Changes

- Downgraded minimum required version of NumPy from `1.26.4` to `1.23.5`.

### Documentation Changes

- Added new code example on vision transformers.
- Added new style for call convention.

### Miscellaneous Changes

- Fixed `stable-build.yml` GitHub action.
- Added more package information (e.g., classifiers, keywords) into `pyproject.toml`.


## [0.1.0a2](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0a2) - 2024-07-05

### New Features

- Added more parameters for `GRUMML`.
- Added more parameters for `LRUMML`.
- Created `Patches` and `PatchEmbedding` layers for use in vision transformers (ViT).

### Changes

- Improved coding standard to better match PyLint standards.
- Moved `TokenEmbedding` to the `core` package, to be more in line with the Keras organization of the layers.
- Split tests to better run in parallel.

### Performance Improvements

- Combined several internal layers in `GRUMML` into one kernel layer, reducing computational load.

### Documentation Changes

- Added explanation for how the matmul-less recurrent units work.
- Added hover tips to documentation.
- Fixed scrollbar ugliness with the math block rendering on the documentation site.
- Made a new look for the homepage.
- Removed spurious "defaults to" in some layers' documentation.


## [0.1.0a1](https://github.com/PhotonicGluon/Keras-MatMulLess/tree/v0.1.0a1) - 2024-07-03

Initial alpha release of Keras-MML.

### New Features

- Added a matmul-less version of the `Dense` layer in Keras.
- Added a Root Mean Square Normalization (RMSNorm) layer.
- Added a matmul-less version of a Gated Linear Unit (GLU), along with its variants by using different activation functions.
- Added a matmul-less Gated Recurrent Unit (GRU) layer.
- Added a matmul-less Linear Recurrent Unit (LRU) layer.
- Added a matmul-less transformer block layer, along with its associated attention and embedding layers.
