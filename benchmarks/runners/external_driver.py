"""Small dependency-local driver executed by optional simulator interpreters."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .common import joint_target_successes, peak_rss_bytes


def _sample(sampler, backend: str, shots: int):
    if backend == "tsim":
        return sampler.sample(
            shots=shots,
            batch_size=max(shots, 1),
            separate_observables=True,
            use_detector_reference_sample=True,
            use_observable_reference_sample=True,
        )
    return sampler.sample(shots, separate_observables=True)


def _rows(data, shots: int) -> np.ndarray:
    array = np.asarray(data, dtype=bool)
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array.reshape((0, 0) if shots == 0 else (shots, -1))
    raise ValueError(f"sampler returned an array with shape {array.shape}")


def _version(module):
    value = getattr(module, "__version__", None)
    if value is None:
        value = getattr(module, "version", None)
        if callable(value):
            value = value()
    return None if value is None else str(value)


def _xtim_diagnostics(circuit, sampler) -> dict:
    """Collect public xtim diagnostics without triggering another compilation."""
    engine_report = getattr(sampler, "engine_report", None)
    reference_info = getattr(circuit, "reference_info", None)
    try:
        engine = engine_report() if callable(engine_report) else None
    except Exception as exc:  # diagnostics must not invalidate a completed benchmark
        engine = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        reference = reference_info() if callable(reference_info) else None
    except Exception as exc:  # diagnostics must not invalidate a completed benchmark
        reference = {"error": f"{type(exc).__name__}: {exc}"}

    chi = getattr(sampler, "chi", None)
    metric_source = "compiled_program" if chi is not None else "unavailable"
    if chi is None and isinstance(reference, dict):
        chi = reference.get("chi")
        if chi is not None:
            metric_source = "cached_reference"
    magic_rank = None
    if isinstance(chi, int) and chi > 0 and chi & (chi - 1) == 0:
        magic_rank = chi.bit_length() - 1
    return {
        "engine_report": engine,
        "program": {
            # xtim 2.7 exposes chi on cached references, but not on a sampler
            # compiled from a deduced bare state. Keep the latter explicitly null.
            "stabilizer_term_count": chi,
            "magic_rank": magic_rank,
            "metric_source": metric_source,
        },
        "reference": reference,
    }


def _run_clifft(module, text: str, *, shots: int, seed: int, warmup: int, n_detectors: int):
    compile_started = time.perf_counter()
    program = module.compile(
        text,
        postselection_mask=[1] * n_detectors if n_detectors else None,
        normalize_syndromes=True,
    )
    compile_s = time.perf_counter() - compile_started

    def sample(count: int, sample_seed: int):
        if n_detectors:
            return module.sample_survivors(program, count, seed=sample_seed)
        return module.sample(program, count, seed=sample_seed)

    warmup_started = time.perf_counter()
    if warmup:
        sample(warmup, seed ^ 0x9E3779B9)
    warmup_s = time.perf_counter() - warmup_started
    sample_started = time.perf_counter()
    result = sample(shots, seed)
    sample_s = time.perf_counter() - sample_started

    n_observables = int(program.num_observables)
    if n_detectors:
        attempted = int(result.total_shots)
        accepted = int(result.passed_shots)
        target_successes = accepted - int(result.logical_errors) if n_observables else None
    else:
        attempted = shots
        accepted = shots
        observables = _rows(result.observables, shots)
        target_successes = joint_target_successes(observables) if n_observables else None
    return {
        "compile_s": compile_s,
        "warmup_s": warmup_s,
        "sample_s": sample_s,
        "attempted": attempted,
        "accepted": accepted,
        "target_successes": target_successes,
        "n_detectors": int(program.num_detectors),
        "n_observables": n_observables,
        "num_actions": int(program.num_actions),
        "peak_active_width": int(program.peak_active_width),
    }


def _symft_joint_counts(circuit, measurements: np.ndarray) -> tuple[int, int]:
    """Aggregate detector acceptance and the joint all-observable target rule."""
    rejected = np.zeros(measurements.shape[0], dtype=bool)
    for detector in circuit.detectors:
        records = np.asarray(detector["records"], dtype=np.int64) - 1
        if records.size:
            rejected |= np.bitwise_xor.reduce(measurements[:, records], axis=1)

    observable_values = np.zeros((measurements.shape[0], circuit.num_observables), dtype=bool)
    for observable in circuit.observables:
        records = np.asarray(observable["records"], dtype=np.int64) - 1
        if records.size:
            observable_values[:, int(observable["index"])] ^= np.bitwise_xor.reduce(
                measurements[:, records], axis=1
            )
    accepted_mask = ~rejected
    accepted = int(np.sum(accepted_mask))
    target_successes = joint_target_successes(observable_values[accepted_mask])
    return accepted, target_successes


def _run_symft(module, text: str, *, shots: int, seed: int, warmup: int):
    compile_started = time.perf_counter()
    circuit = module.Circuit(text)
    n_detectors = int(circuit.num_detectors)
    n_observables = int(circuit.num_observables)
    if n_observables <= 1:
        sampler = circuit.compile_counts_sampler(
            batch=True,
            observable=0,
            postselect_detectors=True,
            threads=1,
        )
        interface = "compile_counts_sampler(postselect_detectors=True)"
        program = {
            **dict(sampler.info),
            "preprocessing_timing": dict(sampler.preprocessing_timing),
            "aggregation": "native_counts",
        }

        def sample(count: int, sample_seed: int):
            return sampler.sample(shots=count, stream_id=sample_seed)

        def aggregate(result, count: int):
            accepted = int(result["accepted"])
            successes = accepted - int(result["logical_errors"]) if n_observables else None
            return int(result["shots"]), accepted, successes

    else:
        sampler = circuit.compile_sampler(batch=True)
        interface = "compile_sampler(batch=True) + joint detector/observable aggregation"
        program = {
            "num_qubits": int(sampler.num_qubits),
            "num_measurements": int(sampler.num_measurements),
            "num_detectors": int(sampler.num_detectors),
            "num_observables": n_observables,
            "max_active_qubits": int(sampler.max_active_qubits),
            "aggregation": "python_joint_measurement_records",
        }

        def sample(count: int, sample_seed: int):
            return sampler.sample(shots=count, seed=sample_seed)

        def aggregate(result, count: int):
            measurements = _rows(result, count)
            accepted, successes = _symft_joint_counts(circuit, measurements)
            return count, accepted, successes

    compile_s = time.perf_counter() - compile_started
    warmup_started = time.perf_counter()
    if warmup:
        sample(warmup, seed ^ 0x9E3779B9)
    warmup_s = time.perf_counter() - warmup_started
    sample_started = time.perf_counter()
    result = sample(shots, seed)
    sample_s = time.perf_counter() - sample_started
    attempted, accepted, target_successes = aggregate(result, shots)
    return {
        "compile_s": compile_s,
        "warmup_s": warmup_s,
        "sample_s": sample_s,
        "attempted": attempted,
        "accepted": accepted,
        "target_successes": target_successes,
        "n_detectors": n_detectors,
        "n_observables": n_observables,
        "interface": interface,
        "program": program,
        "simd_backend": str(module.simd_backend()),
        "cuda_enabled": bool(module.cuda_enabled()),
        "active_cuda_backend": str(module.active_cuda_backend()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("tsim", "xtim", "clifft", "symft"))
    parser.add_argument("circuit")
    parser.add_argument("shots", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("warmup", type=int)
    parser.add_argument("n_detectors", type=int)
    parser.add_argument("n_observables", type=int)
    args = parser.parse_args()
    module = __import__(args.backend)
    text = open(args.circuit, encoding="utf-8").read()

    if args.backend == "clifft":
        payload = _run_clifft(
            module,
            text,
            shots=args.shots,
            seed=args.seed,
            warmup=args.warmup,
            n_detectors=args.n_detectors,
        )
    elif args.backend == "symft":
        payload = _run_symft(
            module,
            text,
            shots=args.shots,
            seed=args.seed,
            warmup=args.warmup,
        )
    else:
        compile_started = time.perf_counter()
        circuit = module.Circuit(text)
        sampler = circuit.compile_detector_sampler(seed=args.seed)
        compile_s = time.perf_counter() - compile_started
        warmup_started = time.perf_counter()
        if args.warmup:
            _sample(sampler, args.backend, args.warmup)
        warmup_s = time.perf_counter() - warmup_started
        sample_started = time.perf_counter()
        detectors, observables = _sample(sampler, args.backend, args.shots)
        sample_s = time.perf_counter() - sample_started
        detectors = _rows(detectors, args.shots)
        observables = _rows(observables, args.shots)
        keep = ~np.any(detectors, axis=1) if detectors.shape[1] else np.ones(args.shots, dtype=bool)
        accepted_observables = observables[keep]
        target_successes = int(np.sum(~np.any(accepted_observables, axis=1))) if observables.shape[1] else None
        payload = {
            "compile_s": compile_s,
            "warmup_s": warmup_s,
            "sample_s": sample_s,
            "attempted": args.shots,
            "accepted": int(np.sum(keep)),
            "target_successes": target_successes,
            "n_detectors": int(detectors.shape[1]),
            "n_observables": int(observables.shape[1]),
        }
        if args.backend == "xtim":
            payload.update(_xtim_diagnostics(circuit, sampler))
    if payload["n_detectors"] != args.n_detectors or payload["n_observables"] != args.n_observables:
        raise ValueError(
            "compiled detector/observable widths "
            f"{(payload['n_detectors'], payload['n_observables'])} do not match "
            f"{(args.n_detectors, args.n_observables)}"
        )
    payload.update({
        "version": _version(module),
        "python": sys.version.split()[0],
        "peak_rss_bytes": peak_rss_bytes(),
    })
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
