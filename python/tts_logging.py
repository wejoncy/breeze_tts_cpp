"""Process-wide native logging, forwarded through a TVM FFI callable."""

import atexit
import logging
import threading
from native_library import library_path, load_library

LEVELS = ("trace", "debug", "info", "warn", "error", "off")
_PY_LEVELS = (5, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
_local = threading.local()
_lock = threading.RLock()
_bindings = {}


def check_not_in_callback():
    if getattr(_local, "active", False):
        raise RuntimeError(
            "cannot reenter engine/log configuration from a log callback"
        )


def configure_logging(
    level="info", *, logger=None, file=None, json=False, library=None
):
    """Configure native stderr/rotating file output, or forward to logging.Logger.

    Native filtering precedes formatting. Callbacks expose engine_id, request_id,
    component and timestamp_us. Handler failures cannot fail inference.
    """
    check_not_in_callback()
    if level not in LEVELS:
        raise ValueError("log level must be " + ", ".join(LEVELS))
    if logger is not None and (
        not isinstance(logger, logging.Logger) or file is not None
    ):
        raise ValueError(
            "logger must be logging.Logger and cannot be combined with file"
        )
    if not isinstance(json, bool):
        raise ValueError("json must be bool")
    file_path = str(file) if file is not None else ""
    if file is not None and (not file_path or "\0" in file_path):
        raise ValueError("invalid log file path")
    callback = None
    if logger is not None:

        def callback(level, timestamp_us, engine_id, request_id, component, message):
            _local.active = True
            try:
                severity = _PY_LEVELS[level]
                if not logger.isEnabledFor(severity):
                    return
                record = logger.makeRecord(
                    logger.name,
                    severity,
                    "",
                    0,
                    message,
                    (),
                    None,
                    extra=dict(
                        engine_id=engine_id,
                        request_id=request_id,
                        component=component,
                        timestamp_us=timestamp_us,
                    ),
                )
                record.created = timestamp_us / 1_000_000
                record.msecs = (timestamp_us % 1_000_000) / 1000
                logger.handle(record)
            except BaseException:
                pass
            finally:
                _local.active = False

    with _lock:
        path = library_path(library)
        module = load_library(path)
        module.configure_logging(level, file_path, json, callback)
        _bindings[path] = module


@atexit.register
def _shutdown():
    with _lock:
        for module in _bindings.values():
            module.configure_logging("off", "", False, None)
        _bindings.clear()
