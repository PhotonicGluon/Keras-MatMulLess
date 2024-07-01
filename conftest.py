import pathlib

import pytest
from pytest import Config, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--eager",
        action="store_true",
        default=False,
        help="whether to run all functions eagerly",
    )


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    if config.getoption("--eager"):
        try:
            import tensorflow as tf

            tf.config.experimental_run_functions_eagerly(True)
        except ModuleNotFoundError:
            print("Tensorflow not installed; ignoring `--eager`.")


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool:
    if collection_path.name.startswith("_"):
        return True
