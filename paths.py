"""Convenience helpers for resolving project paths.

This module centralizes the location of the repository's data and model
directories so the rest of the codebase can resolve paths consistently.

The helpers are intentionally small and operate on ``pathlib.Path`` objects,
which makes them easy to reuse across scripts, notebooks, and tests.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
"""Absolute path to the repository root."""

FILES_DIR = BASE_DIR / "files"
"""Absolute path used for input/output artifacts stored under ``files/``."""

FILES_DIR_NAME = "files"
"""Directory name used when resolving relative file paths."""

MODEL_FILES_DIR = BASE_DIR / "model_files"
"""Absolute path used for persisted model artifacts."""

MODEL_FILES_DIR_NAME = "model_files"
"""Directory name used when resolving relative model paths."""


def ensure_files_dir():
    """Create and return the project ``files`` directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the ``files`` directory.
    """
    FILES_DIR.mkdir(exist_ok=True)
    return FILES_DIR


def ensure_model_files_dir():
    """Create and return the project ``model_files`` directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the ``model_files`` directory.
    """
    MODEL_FILES_DIR.mkdir(exist_ok=True)
    return MODEL_FILES_DIR


def resolve_input_path(path):
    """Resolve a user-provided input path.

    Parameters
    ----------
    path : str | pathlib.Path | None
        The input path to resolve. If ``None``, the function returns ``None``.

    Returns
    -------
    str | None
        The resolved path as a string. Relative paths are first looked up under
        the ``files`` directory and then in the current working directory.
    """
    if path is None:
        return None

    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    explicit_path = path_obj.parent != Path(".")
    if explicit_path:
        return str(path_obj)

    files_candidate = Path(FILES_DIR_NAME) / path_obj
    if files_candidate.exists():
        return str(files_candidate)

    if path_obj.exists():
        return str(path_obj)

    return str(files_candidate)


def resolve_output_path(path):
    """Resolve a path intended for writing output artifacts.

    Parameters
    ----------
    path : str | pathlib.Path | None
        The destination path to resolve. If ``None``, the function returns
        ``None``.

    Returns
    -------
    str | None
        The resolved path as a string. Relative paths are written under
        ``files/`` and the directory is created if necessary.
    """
    if path is None:
        return None

    path_obj = Path(path)

    if path_obj.is_absolute() or path_obj.parent != Path("."):
        return str(path_obj)

    ensure_files_dir()
    return str(Path(FILES_DIR_NAME) / path_obj)


def resolve_model_path(path):
    """Resolve a path intended for model artifacts.

    Parameters
    ----------
    path : str | pathlib.Path | None
        The model path to resolve. If ``None``, the function returns ``None``.

    Returns
    -------
    str | None
        The resolved path as a string. Relative paths are first checked under
        ``model_files/`` and then in the current working directory; missing
        directories are created on demand.
    """
    if path is None:
        return None

    path_obj = Path(path)

    if path_obj.is_absolute() or path_obj.parent != Path("."):
        return str(path_obj)

    model_files_candidate = Path(MODEL_FILES_DIR_NAME) / path_obj
    if model_files_candidate.exists():
        return str(model_files_candidate)

    if path_obj.exists():
        return str(path_obj)

    ensure_model_files_dir()
    return str(model_files_candidate)
