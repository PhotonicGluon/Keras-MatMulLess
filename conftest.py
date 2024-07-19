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
        os.environ["DISABLE_TORCH_COMPILE"] = "true"
        try:
            import tensorflow as tf

            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()
        except ModuleNotFoundError:
            print("Tensorflow not installed; ignoring `--eager` flag for Tensorflow.")


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
    if collection_path.name == "docs":
        return True
    if collection_path.name == "experiments":
        return True

    try:
        import triton
    except ModuleNotFoundError:
        # Can skip any triton-related modules
        if "triton" in collection_path.name:
            return True
