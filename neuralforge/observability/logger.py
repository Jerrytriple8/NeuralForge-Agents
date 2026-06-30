"""Structured JSON logger."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class StructuredLogger:
    """Structured JSON logger that outputs one JSON object per line.

    Args:
        name: Logger namespace (e.g., "neuralforge.engine").
        level: Minimum log level.
        stream: Output stream (default stderr).
    """

    def __init__(
        self,
        name: str = "neuralforge",
        level: int = logging.INFO,
        stream: Any = None,
    ) -> None:
        self._name = name
        self._level = level
        self._stream = stream or sys.stderr

    def _emit(self, level: str, message: str, **kwargs: Any) -> None:
        entry = {
            "ts": time.time(),
            "level": level,
            "logger": self._name,
            "message": message,
        }
        if kwargs:
            entry["extra"] = kwargs
        self._stream.write(json.dumps(entry, default=str) + "\n")
        self._stream.flush()

    def debug(self, msg: str, **kw: Any) -> None:
        if self._level <= logging.DEBUG:
            self._emit("DEBUG", msg, **kw)

    def info(self, msg: str, **kw: Any) -> None:
        if self._level <= logging.INFO:
            self._emit("INFO", msg, **kw)

    def warn(self, msg: str, **kw: Any) -> None:
        if self._level <= logging.WARNING:
            self._emit("WARN", msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        if self._level <= logging.ERROR:
            self._emit("ERROR", msg, **kw)

    def child(self, name: str) -> "StructuredLogger":
        """Create a child logger with a dotted namespace."""
        return StructuredLogger(f"{self._name}.{name}", self._level, self._stream)
