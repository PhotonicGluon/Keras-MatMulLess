"""
Helper script to install backend-related stuff.

Especially helpful during development.
"""

import os
from argparse import ArgumentParser
from typing import Tuple

REQUIREMENTS_FOLDER = "requirements"


def install(groups: Tuple[str, ...], backends: Tuple[str, ...], with_cuda: bool):
    """
    Dependencies installer.

    WITH_GROUPS are the main dependency groups to install (from `pyproject.toml`).
    """

    # Install the groups first
    if len(groups) != 0:
        os.system(f"poetry install --with {','.join(groups)}")

    # Then install backend dependencies
    requirements_subfolder = "cuda" if with_cuda else "cpu"

    for backend in backends:
        requirements_path = os.path.join(REQUIREMENTS_FOLDER, requirements_subfolder, f"requirements-{backend}.txt")
        os.system(f"poetry run pip install -r {requirements_path}")

    print("\x1b[32mDone!")


if __name__ == "__main__":
    parser = ArgumentParser(prog="install.py", description="Dependencies installer.")

    parser.add_argument(
        "groups",
        nargs="+",
        type=str.lower,
        help="main dependency groups to install (from `pyproject.toml`)",
    )
    parser.add_argument(
        "--backend",
        nargs="*",
        type=str.lower,
        choices=["tensorflow", "torch", "jax"],
        help="backend dependencies to install",
    )
    parser.add_argument(
        "--with-cuda", "--cuda", action="store_true", help="whether the installation is with CUDA support"
    )

    args = parser.parse_args()

    install(args.groups, args.backend if args.backend else [], args.with_cuda)
