import pathlib


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool:
    if collection_path.name.startswith("_"):
        return True
