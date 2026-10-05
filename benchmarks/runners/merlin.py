"""In-process adapter for merlin's postselecting CircuitSampler."""

from __future__ import annotations

import importlib.metadata
import sys
import time
from collections.abc import Callable
from typing import Any

from benchmarks.models import CircuitCase

from .common import joint_target_successes, make_result, peak_rss_bytes


def _configuration(case: CircuitCase, *, shots: int, seed: int, warmup_shots: int, workload: str):
    return {
        "p_phys": case.p_phys,
        "noise_model": case.noise_model,
        "target_scoring": case.target_scoring,
        "workload": workload,
        "shots": int(shots),
        "seed": int(seed),
        "warmup_shots": int(warmup_shots),
    }


def run_merlin(
    case: CircuitCase,
    *,
    shots: int,
    seed: int,
    warmup_shots: int = 0,
    build_s: float = 0.0,
    workload: str = "target-scoring",
    save_iter: int = 0,
    callback: Callable[[dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from merlin import CircuitSampler

    shots = int(shots)
    if shots < 0 or warmup_shots < 0 or save_iter < 0:
        raise ValueError("shots, warmup_shots, and save_iter must be nonnegative")
    configuration = _configuration(case, shots=shots, seed=seed, warmup_shots=warmup_shots, workload=workload)
    simulator = {
        "name": "merlin",
        "version": importlib.metadata.version("merlin"),
        "python": sys.version.split()[0],
        "interface": "CircuitSampler(postselect=all_detectors)",
        "noise_sampling": "backend_internal",
    }

    started = time.perf_counter()
    compile_started = time.perf_counter()
    sampler = CircuitSampler(case.circuit, seed=int(seed), postselect=case.detector_indices)
    compile_s = time.perf_counter() - compile_started
    warmup_started = time.perf_counter()
    if warmup_shots:
        sampler.sample(int(warmup_shots))
    warmup_s = time.perf_counter() - warmup_started

    completed = accepted = target_successes = 0
    previous_sample_s = 0.0
    resume_replay_s = 0.0
    if resume and resume.get("partial"):
        counts = resume.get("counts", {})
        completed = min(int(counts.get("attempted", 0)), shots)
        accepted = int(counts.get("accepted", 0))
        saved_target = counts.get("accepted_and_target_pass")
        target_successes = 0 if saved_target is None else int(saved_target)
        previous_sample_s = float(resume.get("timing", {}).get("sample_s", 0.0))
        if completed:
            replay_started = time.perf_counter()
            sampler.sample(completed)
            resume_replay_s = time.perf_counter() - replay_started

    sample_s = previous_sample_s
    chunk_size = save_iter or max(shots - completed, 1)
    last_row = None
    while completed < shots:
        chunk = min(chunk_size, shots - completed)
        sample_started = time.perf_counter()
        result = sampler.sample(chunk)
        detectors = result.detectors
        observables = result.observables
        sample_s += time.perf_counter() - sample_started
        chunk_accepted = int(detectors.shape[0])
        accepted += chunk_accepted
        if case.target_scoring:
            target_successes += joint_target_successes(observables)
        completed += chunk
        partial = completed < shots
        total_s = build_s + compile_s + warmup_s + sample_s
        timing = {
            "construction_s": float(build_s),
            "compile_s": compile_s,
            "warmup_s": warmup_s,
            "sample_s": sample_s,
            "resume_replay_s": resume_replay_s,
            "total_wall_s": total_s,
            "sample_s_per_attempted_shot": sample_s / completed if completed else None,
            "total_s_per_attempted_shot": total_s / completed if completed else None,
            "sample_s_per_accepted_shot": sample_s / accepted if accepted else None,
        }
        last_row = make_result(
            case,
            simulator=simulator,
            configuration=configuration,
            attempted=completed,
            accepted=accepted,
            target_successes=target_successes if case.target_scoring else None,
            timing=timing,
            rss_bytes=peak_rss_bytes(),
            partial=partial,
        )
        if callback and (save_iter or not partial):
            callback(last_row)

    if last_row is None:
        total_s = build_s + compile_s + warmup_s
        last_row = make_result(
            case,
            simulator=simulator,
            configuration=configuration,
            attempted=0,
            accepted=0,
            target_successes=0 if case.target_scoring else None,
            timing={
                "construction_s": float(build_s),
                "compile_s": compile_s,
                "warmup_s": warmup_s,
                "sample_s": 0.0,
                "resume_replay_s": resume_replay_s,
                "total_wall_s": total_s,
                "sample_s_per_attempted_shot": None,
                "total_s_per_attempted_shot": None,
                "sample_s_per_accepted_shot": None,
            },
            rss_bytes=peak_rss_bytes(),
        )
        if callback:
            callback(last_row)
    _ = started
    return last_row
