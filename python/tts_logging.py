"""Configure native stderr or rotating file logging."""

from native_library import load_library

LEVELS = ("trace", "debug", "info", "warn", "error", "off")


def configure_logging(level="info", *, file=None, json=False, library=None):
    if level not in LEVELS:
        raise ValueError("log level must be " + ", ".join(LEVELS))
    if not isinstance(json, bool):
        raise ValueError("json must be bool")
    file_path = str(file) if file is not None else ""
    if file is not None and (not file_path or "\0" in file_path):
        raise ValueError("invalid log file path")
    load_library(library).configure_logging(level, file_path, json)
