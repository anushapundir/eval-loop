"""Centralized logging configuration.

One place to configure logging so every module logs consistently. Call
``setup_logging`` once at process entry (CLI, API, UI); use ``get_logger``
everywhere else.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Idempotent across repeated calls."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(name)
