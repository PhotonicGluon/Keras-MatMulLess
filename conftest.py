import os
import pathlib

from pytest import Config, Parser


def pytest_addoption(parser: Parser) -> None:
    """
    Register argparse-style options and ini-style config values, called once at the beginning of a
    test run.
    """

    parser.addoption(
        "--eager",
        action="store_true",
        default=False,
        help="whether to run all functions eagerly",
    )


def pytest_configure(config: Config):
    """
    Allow plugins and conftest files to perform initial configuration.
    """

    if config.getoption("--eager"):
        os.environ["PYTEST_USE_EAGER"] = "true"
        try:
            import tensorflow as tf

            tf.config.experimental_run_functions_eagerly(True)
        except ModuleNotFoundError:
            print("Tensorflow not installed; ignoring `--eager`.")


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool:
    """
    Return ``True`` to ignore this path for collection.

    Return ``None`` to let other plugins ignore the path for collection.

    Returning ``False`` will forcefully not ignore this path for collection, without giving a chance
    for other plugins to ignore this path.

    This hook is consulted for all files and directories prior to calling more specific hooks.
    """

    if collection_path.name.startswith("_"):
        return True
