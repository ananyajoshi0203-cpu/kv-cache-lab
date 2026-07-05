"""Logging setup. Library modules log via logging.getLogger(__name__); apps call
configure() once to decide verbosity and destination."""

from __future__ import annotations

import logging
import sys

logging.getLogger("kvlab").addHandler(logging.NullHandler())

_CLEAN = "%(message)s"
_VERBOSE = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(level: int | str = logging.INFO, verbose: bool = False, stream=None) -> logging.Logger:
    root = logging.getLogger("kvlab")
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        if not isinstance(h, logging.NullHandler):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(_VERBOSE if verbose else _CLEAN, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("kvlab") else f"kvlab.{name}")
