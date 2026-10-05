"""Metrics and resource accounting shared by runners."""

from __future__ import annotations

import math
import resource
import sys
from typing import Any

import numpy as np

from benchmarks.models import CircuitCase


def peak_rss_bytes(*, children: bool = False) -> int:
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def joint_target_successes(observables) -> int:
    """Count shots for which every target projection passes."""
    values = np.asarray(observables, dtype=bool)
    if values.ndim != 2:
        raise ValueError("observables must be a two-dimensional shot matrix")
    return int(np.sum(~np.any(values, axis=1)))


def bernoulli(value: float | None, trials: int) -> dict[str, Any]:
    stderr = None if value is None or not trials else math.sqrt(max(value * (1 - value), 0.0) / trials)
    return {"value": value, "stderr": stderr, "trials": int(trials)}


def quality_metrics(*, attempted: int, accepted: int, target_successes: int | None):
    acceptance = accepted / attempted if attempted else None
    infidelity = None
    if target_successes is not None and accepted:
        infidelity = 1 - target_successes / accepted
    return {
        "acceptance": {
            **bernoulli(acceptance, attempted),
            "definition": "accepted / attempted",
        },
        "target_infidelity": {
            **bernoulli(infidelity, accepted),
            "definition": "1 - accepted_and_all_target_projections_pass / accepted",
        },
    }


def make_result(
    case: CircuitCase,
    *,
    simulator: dict[str, Any],
    configuration: dict[str, Any],
    attempted: int,
    accepted: int,
    target_successes: int | None,
    timing: dict[str, Any],
    rss_bytes: int,
    partial: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "partial" if partial else "ok",
        "case": case.description(),
        "simulator": simulator,
        "configuration": configuration,
        "counts": {
            "attempted": int(attempted),
            "accepted": int(accepted),
            "accepted_and_target_pass": None if target_successes is None else int(target_successes),
        },
        "quality": quality_metrics(
            attempted=attempted,
            accepted=accepted,
            target_successes=target_successes,
        ),
        "timing": timing,
        "memory": {
            "peak_rss_bytes": int(rss_bytes),
            "peak_rss_mb": int(rss_bytes) / (1024 * 1024),
            "metric": "resource.ru_maxrss",
        },
        "partial": bool(partial),
    }


def error_result(
    case: CircuitCase,
    *,
    simulator: str,
    configuration: dict[str, Any],
    error_type: str,
    message: str,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "error",
        "case": case.description(),
        "simulator": {"name": simulator},
        "configuration": configuration,
        "counts": {"attempted": 0, "accepted": 0, "accepted_and_target_pass": None},
        "quality": quality_metrics(attempted=0, accepted=0, target_successes=None),
        "timing": {},
        "memory": {},
        "partial": False,
        "error": {"type": error_type, "message": message, "detail": detail},
    }
