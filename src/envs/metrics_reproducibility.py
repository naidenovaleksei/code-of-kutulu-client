"""
Helpers to compare Metrics evaluation outputs and INFO logs across runs.

Artifact paths in log lines are normalized so comparisons work across machines.
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

_ARTIFACT_PATH_RE = re.compile(r"Loading agent from '[^']+'")


def normalize_eval_log_line(line: str) -> str:
    """Replace absolute artifact paths so log lines match on different hosts."""
    return _ARTIFACT_PATH_RE.sub("Loading agent from '<ARTIFACT_PATH>'", line)


def normalize_eval_log_lines(lines: Iterable[str]) -> List[str]:
    return [normalize_eval_log_line(line) for line in lines]


def metrics_results_equal(
    a: Dict[str, Union[tuple, float, int, np.number]],
    b: Dict[str, Union[tuple, float, int, np.number]],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Tuple[bool, str]:
    """Deep-compare metrics dicts (tuples of floats use np.isclose)."""
    if set(a) != set(b):
        missing = set(a).symmetric_difference(set(b))
        return False, f"keys differ: {missing}"

    for key in a:
        v1, v2 = a[key], b[key]
        if isinstance(v1, tuple) and isinstance(v2, tuple):
            if len(v1) != len(v2):
                return False, f"{key}: tuple lengths {len(v1)} vs {len(v2)}"
            for i, (x, y) in enumerate(zip(v1, v2)):
                if isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
                    if bool(x) != bool(y):
                        return False, f"{key}[{i}]: {x} vs {y}"
                elif isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)):
                    if int(x) != int(y):
                        return False, f"{key}[{i}]: {x} vs {y}"
                elif isinstance(x, (float, np.floating)) or isinstance(y, (float, np.floating)):
                    if not np.isclose(float(x), float(y), rtol=rtol, atol=atol):
                        return False, f"{key}[{i}]: {x} vs {y}"
                elif x != y:
                    return False, f"{key}[{i}]: {x} vs {y}"
        elif isinstance(v1, (float, np.floating)) or isinstance(v2, (float, np.floating)):
            if not np.isclose(float(v1), float(v2), rtol=rtol, atol=atol):
                return False, f"{key}: {v1} vs {v2}"
        elif v1 != v2:
            return False, f"{key}: {v1} vs {v2}"
    return True, ""


class _FilteredInfoHandler(logging.Handler):
    """Append formatted INFO+ records whose logger name matches substrings."""

    def __init__(self, formatter: logging.Formatter, name_substrings: Tuple[str, ...]):
        super().__init__(level=logging.INFO)
        self.setFormatter(formatter)
        self._name_substrings = name_substrings
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.INFO:
            return
        if not any(sub in record.name for sub in self._name_substrings):
            return
        self.lines.append(self.format(record))


@contextmanager
def capture_src_envs_info_logs(
    name_substrings: Tuple[str, ...] = (
        "kutulu_world",
        "trainer",
        "nn_agent",
        "dqn_agent",
    ),
):
    """
    Capture INFO records from loggers under ``src.envs`` whose names contain
    any of ``name_substrings`` (e.g. trainer rollouts, env seeds, model load paths).
    """
    fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler = _FilteredInfoHandler(fmt, name_substrings)

    env_logger = logging.getLogger("src.envs")
    env_logger.addHandler(handler)
    saved_level = env_logger.level
    env_logger.setLevel(logging.INFO)
    try:
        yield handler.lines
    finally:
        env_logger.removeHandler(handler)
        env_logger.setLevel(saved_level)
