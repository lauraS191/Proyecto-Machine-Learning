"""Contratos de datos, archivos atómicos e identificación de artefactos."""

import contextlib
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

LABELS = ["negativo", "neutral", "positivo"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "parte2"
DEFAULT_PREPARED = ROOT / "artifacts" / "parte2" / "prepared"
DEFAULT_RUNS = ROOT / "artifacts" / "parte2" / "runs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seal_bundle(directory, metadata):
    directory = Path(directory)
    metadata["files_sha256"] = {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*")) if path.is_file() and path != directory / "bundle.json"
    }
    write_json(directory / "bundle.json", metadata)


def verify_bundle(directory):
    directory = Path(directory)
    metadata = read_json(directory / "bundle.json")
    for relative, expected in metadata["files_sha256"].items():
        path = directory / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Archivo del bundle modificado o ausente: {path}")
    return metadata


def read_csv(path, required):
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(f"CSV vacío o columnas ausentes en {path}. Requeridas: {required}")
    if "id" in frame:
        if frame["id"].eq("").any() or frame["id"].duplicated().any():
            raise ValueError(f"IDs vacíos o repetidos en {path}")
    if "text" in frame and frame["text"].str.strip().eq("").any():
        raise ValueError(f"Textos vacíos en {path}")
    if "label" in frame and not frame["label"].isin(LABELS).all():
        raise ValueError(f"Etiquetas desconocidas en {path}")
    return frame


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive(directory):
    """El lock persiste durante el trabajo, nunca contiene resultados parciales."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Operación activa en {directory}. Si murió el proceso, revisar y retirar {lock}") from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def new_run(root, name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", name):
        raise ValueError("El nombre del experimento debe usar letras, números, _ o -")
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def environment():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commit": commit.stdout.strip(),
        "working_tree_dirty": bool(status.stdout.strip()),
        "packages": {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]},
    }
