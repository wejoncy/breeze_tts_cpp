"""Locate the native artifact from build metadata, then load its dependencies locally."""

import json
import os
import threading
from pathlib import Path
import tvm_ffi
from native_types import register_types

_lock = threading.RLock()
_loaded = {}


def library_path(library=None):
    selected = library if library is not None else os.environ.get("TTS_LIBRARY")
    if selected is None:
        build = Path(__file__).resolve().parents[1] / "build"
        direct = build / "tts-runtime.json"
        manifests = (
            [direct] if direct.is_file() else sorted(build.rglob("tts-runtime.json"))
        )
        if len(manifests) != 1:
            raise RuntimeError(
                "Native build is missing or ambiguous; pass library= or set TTS_LIBRARY "
                "to a native library, its build directory, or tts-runtime.json"
            )
        selected = manifests[0]
    path = Path(selected).expanduser().resolve()
    if path.is_dir():
        path = path / "tts-runtime.json"
    if path.name == "tts-runtime.json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        path = (path.parent / metadata["library"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Native library not found: {path}")
    return path


def load_library(library=None):
    path = library_path(library)
    with _lock:
        if path not in _loaded:
            if _loaded:
                previous = next(iter(_loaded))
                if path.samefile(previous):
                    return _loaded[previous][0]
                raise RuntimeError(
                    "A different native library is already loaded; use the same library "
                    "for logging and engines (or set TTS_LIBRARY before importing)."
                )
            module, directories = _load(path)
            register_types()
            _loaded[path] = (module, directories)
        return _loaded[path][0]


def _load(path):
    directories = []
    if os.name == "nt":
        manifest = path.parent / "tts-runtime.json"
        metadata = (
            json.loads(manifest.read_text(encoding="utf-8"))
            if manifest.is_file()
            else {}
        )
        candidates = [path.parent]
        candidates.extend(
            (path.parent / p).parent for p in metadata.get("runtime_libraries", []) if p
        )
        candidates.extend(
            path.parent / p for p in metadata.get("runtime_directories", []) if p
        )
        try:
            for directory in dict.fromkeys(p.resolve() for p in candidates):
                if directory.is_dir():
                    directories.append(os.add_dll_directory(str(directory)))
        except BaseException:
            for directory in directories:
                directory.close()
            raise
    try:
        library = tvm_ffi.load_module(str(path))
    except BaseException:
        for directory in directories:
            directory.close()
        raise
    # TVM keeps registered code loaded for process lifetime. Match DLL search lifetime.
    return library, directories
