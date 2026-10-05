"""Subprocess adapters that keep optional simulator imports detachable."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from benchmarks.models import CircuitCase

from .common import error_result, make_result


def default_interpreter(backend: str) -> str:
    variable = f"{backend.upper()}_PYTHON"
    return os.environ.get(variable, sys.executable)


def run_external(
    case: CircuitCase,
    *,
    backend: str,
    shots: int,
    seed: int,
    warmup_shots: int = 0,
    build_s: float = 0.0,
    workload: str = "target-scoring",
    interpreter: str | None = None,
    timeout_s: float = 3600.0,
) -> dict[str, Any]:
    if backend not in {"tsim", "xtim", "clifft", "symft"}:
        raise ValueError(f"unknown external backend: {backend!r}")
    interpreter = interpreter or default_interpreter(backend)
    configuration = {
        "p_phys": case.p_phys,
        "noise_model": case.noise_model,
        "target_scoring": case.target_scoring,
        "workload": workload,
        "shots": int(shots),
        "seed": int(seed),
        "warmup_shots": int(warmup_shots),
        "timeout_s": float(timeout_s),
    }
    path = None
    process_started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".stim", encoding="utf-8", delete=False) as handle:
            handle.write(case.circuit)
            path = handle.name
        completed = subprocess.run(
            [
                interpreter,
                "-m",
                "benchmarks.runners.external_driver",
                backend,
                path,
                str(int(shots)),
                str(int(seed)),
                str(int(warmup_shots)),
                str(int(case.n_detectors)),
                str(int(case.n_observables)),
            ],
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        )
        process_wall_s = time.perf_counter() - process_started
    except subprocess.TimeoutExpired as exc:
        return error_result(
            case,
            simulator=backend,
            configuration=configuration,
            error_type="timeout",
            message=f"{backend} exceeded the {timeout_s:g}s timeout",
            detail=str(exc),
        )
    except OSError as exc:
        return error_result(
            case,
            simulator=backend,
            configuration=configuration,
            error_type="interpreter",
            message=f"could not execute {interpreter!r}",
            detail=str(exc),
        )
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    if completed.returncode:
        kind = "missing_dependency" if "ModuleNotFoundError" in completed.stderr else "subprocess"
        return error_result(
            case,
            simulator=backend,
            configuration=configuration,
            error_type=kind,
            message=f"{backend} subprocess exited with status {completed.returncode}",
            detail=completed.stderr.strip()[-4000:],
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return error_result(
            case,
            simulator=backend,
            configuration=configuration,
            error_type="output",
            message=f"{backend} did not emit a valid result",
            detail=f"{exc}: {completed.stdout[-2000:]}",
        )
    attempted = int(payload["attempted"])
    accepted = int(payload["accepted"])
    reported_shape = (int(payload["n_detectors"]), int(payload["n_observables"]))
    expected_shape = (case.n_detectors, case.n_observables)
    if reported_shape != expected_shape:
        return error_result(
            case,
            simulator=backend,
            configuration=configuration,
            error_type="incompatible_output",
            message=f"{backend} returned detector/observable widths {reported_shape}, expected {expected_shape}",
        )
    target_successes = payload.get("target_successes") if case.target_scoring else None
    total_s = float(build_s) + process_wall_s
    timing = {
        "construction_s": float(build_s),
        "compile_s": float(payload["compile_s"]),
        "warmup_s": float(payload["warmup_s"]),
        "sample_s": float(payload["sample_s"]),
        "subprocess_wall_s": process_wall_s,
        "total_wall_s": total_s,
        "sample_s_per_attempted_shot": float(payload["sample_s"]) / attempted if attempted else None,
        "total_s_per_attempted_shot": total_s / attempted if attempted else None,
        "sample_s_per_accepted_shot": float(payload["sample_s"]) / accepted if accepted else None,
    }
    if backend == "clifft":
        interface = "compile(normalize_syndromes=True, postselection_mask=all_detectors) + sample_survivors"
    elif backend == "symft":
        interface = payload["interface"]
    else:
        interface = "compile_detector_sampler(separate_observables=True)"
    simulator = {
        "name": backend,
        "version": payload.get("version"),
        "python": payload.get("python"),
        "interface": interface,
        "noise_sampling": "backend_internal",
    }
    if backend == "xtim":
        simulator["engine"] = payload.get("engine_report")
        simulator["program"] = payload.get("program")
    if backend == "clifft":
        simulator["program"] = {
            "num_actions": payload.get("num_actions"),
            "peak_active_width": payload.get("peak_active_width"),
        }
    if backend == "symft":
        simulator["program"] = payload.get("program")
        simulator["simd_backend"] = payload.get("simd_backend")
        simulator["cuda_enabled"] = payload.get("cuda_enabled")
        simulator["active_cuda_backend"] = payload.get("active_cuda_backend")
    return make_result(
        case,
        simulator=simulator,
        configuration=configuration,
        attempted=attempted,
        accepted=accepted,
        target_successes=None if target_successes is None else int(target_successes),
        timing=timing,
        rss_bytes=int(payload["peak_rss_bytes"]),
    )
