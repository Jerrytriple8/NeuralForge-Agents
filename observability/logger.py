"""
Structured logging with context enrichment.
"""

import json
import time
import logging
import threading
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredLogger:
    """
    Structured JSON logger with context propagation.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self._name = name
        self._level = getattr(logging, level.upper())
        self._context: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._handler = logging.StreamHandler()
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self._level)
        if not self._logger.handlers:
            self._logger.addHandler(self._handler)

    def set_context(self, key: str, value: Any) -> None:
        with self._lock:
            self._context[key] = value

    def clear_context(self) -> None:
        with self._lock:
            self._context.clear()

    def _log(self, level: str, message: str, **kwargs) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self._name,
            "message": message,
            **self._context,
            **kwargs,
        }
        getattr(self._logger, level.lower())(json.dumps(record))

    def debug(self, message: str, **kwargs) -> None:
        if self._level <= logging.DEBUG:
            self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        if self._level <= logging.INFO:
            self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        if self._level <= logging.WARNING:
            self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        if self._level <= logging.ERROR:
            self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log("CRITICAL", message, **kwargs)

    def child(self, name: str) -> "StructuredLogger":
        child = StructuredLogger(f"{self._name}.{name}", logging.getLevelName(self._level))
        child._context = {**self._context}
        return child
