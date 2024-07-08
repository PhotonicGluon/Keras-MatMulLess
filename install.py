"""
Helper script to install backend-related stuff.

Especially helpful during development.
"""

import os
from typing import Tuple

import click

REQUIREMENTS_FOLDER = "requirements"


@click.command()
@click.argument("groups", type=str, nargs=-1)
@click.option(
    "--backend",
    "backends",
    type=click.Choice(["tensorflow", "torch", "jax"]),
    required=True,
    multiple=True,
    help="Backend dependencies to install.",
)
@click.option(
    "--cuda",
    "--with-cuda",
    "with_cuda",
    is_flag=True,
    default=False,
    help="Whether the installation is with CUDA support.",
)
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
        os.system(f"pip install -r {requirements_path}")

    click.secho("Done!", fg="green")


if __name__ == "__main__":
    install()
