from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
FILES_DIR_NAME = "files"
MODEL_FILES_DIR = BASE_DIR / "model_files"
MODEL_FILES_DIR_NAME = "model_files"


def ensure_files_dir():
    FILES_DIR.mkdir(exist_ok=True)
    return FILES_DIR


def ensure_model_files_dir():
    MODEL_FILES_DIR.mkdir(exist_ok=True)
    return MODEL_FILES_DIR


def resolve_input_path(path):
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
    if path is None:
        return None

    path_obj = Path(path)

    if path_obj.is_absolute() or path_obj.parent != Path("."):
        return str(path_obj)

    ensure_files_dir()
    return str(Path(FILES_DIR_NAME) / path_obj)


def resolve_model_path(path):
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
