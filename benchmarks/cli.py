"""Unified command-line interface for the benchmark suite."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import sys
import time
from typing import Any

from .models import CircuitCase
from .plotting import METRIC_SPECS, plot_file
from .protocols import build_code_switching_case, build_cultivation_case, build_distillation_case
from .runners import run_merlin, run_external
from .storage import atomic_write, load_document, new_document, result_key, upsert_result


SIMULATORS = ("merlin", "tsim", "xtim", "clifft", "symft")
WORKLOADS = ("detector-sampling", "target-scoring")
_WORKER_UPDATES = None
WORKLOAD_HELP = (
    "detector-sampling measures detector acceptance, timing, and memory without "
    "target observables; target-scoring additionally scores the desired output "
    "and reports target infidelity"
)


def _csv(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(text).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list must not be empty")
    return values


def _probabilities(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in _csv(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(not 0 <= value <= 1 for value in values):
        raise argparse.ArgumentTypeError("all probabilities must be in [0, 1]")
    return values


def _simulators(text: str) -> tuple[str, ...]:
    values = SIMULATORS if text == "all" else _csv(text)
    unknown = set(values).difference(SIMULATORS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown simulators: {sorted(unknown)}")
    return values


def _workloads(text: str) -> tuple[str, ...]:
    values = WORKLOADS if text == "all" else _csv(text)
    unknown = set(values).difference(WORKLOADS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown workloads: {sorted(unknown)}")
    return values


def _case_token(
    token: str,
    *,
    p_phys: float,
    target_scoring: bool,
    noisy_clifford: bool,
) -> CircuitCase:
    token = token.lower()
    if token == "15to1":
        return build_distillation_case(
            "15to1",
            p_phys=p_phys,
            noisy_clifford=noisy_clifford,
            target_scoring=target_scoring,
        )
    if token == "bh":
        return build_distillation_case(
            "bh",
            k=2,
            p_phys=p_phys,
            noisy_clifford=noisy_clifford,
            target_scoring=target_scoring,
        )
    if token.startswith("bh:"):
        return build_distillation_case(
            "bh",
            k=int(token.split(":", 1)[1]),
            p_phys=p_phys,
            noisy_clifford=noisy_clifford,
            target_scoring=target_scoring,
        )
    if token in {"bt27", "bt81"}:
        return build_code_switching_case(token, p_phys=p_phys, target_scoring=target_scoring)
    if token in {"cultivation-d3-t", "cultivation-d5-t"}:
        return build_cultivation_case(
            3 if token == "cultivation-d3-t" else 5,
            p_phys=p_phys,
            target_scoring=target_scoring,
        )
    raise ValueError(f"unknown benchmark case: {token!r}")


def _shots(args, simulator: str) -> int:
    override = getattr(args, "shots_" + simulator.replace("-", "_"), None)
    return int(args.shots if override is None else override)


def _configuration(case: CircuitCase, args, simulator: str, workload: str) -> dict[str, Any]:
    result = {
        "p_phys": case.p_phys,
        "noise_model": case.noise_model,
        "target_scoring": case.target_scoring,
        "workload": workload,
        "shots": _shots(args, simulator),
        "seed": int(args.seed),
        "warmup_shots": int(args.warmup_shots),
    }
    if simulator != "merlin":
        result["timeout_s"] = float(args.timeout)
    return result


def _resume_row(document, case: CircuitCase, args, simulator: str, workload: str):
    key = _task_key(case, args, simulator, workload)
    return next((row for row in document["results"] if result_key(row) == key), None)


def _task_key(case: CircuitCase, args, simulator: str, workload: str) -> str:
    prototype = {
        "case": case.description(),
        "simulator": {"name": simulator},
        "configuration": _configuration(case, args, simulator, workload),
    }
    return result_key(prototype)


def _is_checkpoint_extension(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    selections = ("cases", "p_grid", "workloads", "simulators")
    if not all(
        isinstance(configuration.get(name), list)
        for configuration in (existing, requested)
        for name in selections
    ):
        return False
    ignored = {*selections, "timeout"}
    for simulator in SIMULATORS:
        name = simulator.replace("-", "_")
        ignored.add(f"shots_{name}")
        # Interpreter paths appeared in older checkpoint configurations.
        ignored.add(f"{name}_python")
    existing_rest = {key: value for key, value in existing.items() if key not in ignored}
    requested_rest = {key: value for key, value in requested.items() if key not in ignored}
    if existing_rest != requested_rest:
        return False
    for simulator in set(existing["simulators"]) & set(requested["simulators"]):
        name = simulator.replace("-", "_")
        if existing.get(f"shots_{name}") != requested.get(f"shots_{name}"):
            return False
    return True


def _merge_checkpoint_configuration(existing: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    merged = {**existing, **requested}
    for name in ("cases", "p_grid", "workloads", "simulators"):
        merged[name] = list(dict.fromkeys([*existing[name], *requested[name]]))
    for simulator in set(existing["simulators"]) - set(requested["simulators"]):
        name = simulator.replace("-", "_")
        merged[f"shots_{name}"] = existing.get(f"shots_{name}")
    return merged


def _prepare_document(args, *, kind: str, invocation: dict[str, Any]):
    if args.out and Path(args.out).exists() and not args.restart:
        document = load_document(args.out)
        existing_configuration = document.get("configuration")
        exact_match = document.get("kind") == kind and existing_configuration == invocation
        checkpoint_extension = (
            document.get("kind") == kind
            and isinstance(existing_configuration, dict)
            and _is_checkpoint_extension(existing_configuration, invocation)
        )
        if not exact_match and not checkpoint_extension:
            raise ValueError("existing checkpoint configuration does not exactly match; use --restart")
        if checkpoint_extension:
            document["configuration"] = _merge_checkpoint_configuration(existing_configuration, invocation)
            atomic_write(args.out, document)
        return document
    document = new_document(kind=kind, configuration=invocation)
    if args.out:
        atomic_write(args.out, document)
    return document


def _report_start(case: CircuitCase, simulator: str, args, workload: str) -> None:
    if args.progress:
        print(
            f"[benchmark] start case={case.case_id} sim={simulator} "
            f"workload={workload} p={case.p_phys:g} shots={_shots(args, simulator)}",
            file=sys.stderr,
            flush=True,
        )


def _report_checkpoint(row: dict[str, Any], case: CircuitCase, simulator: str, args) -> None:
    if args.progress:
        counts = row["counts"]
        print(
            f"[benchmark] case={case.case_id} sim={simulator} "
            f"shots={counts['attempted']}/{_shots(args, simulator)} accepted={counts['accepted']}",
            file=sys.stderr,
            flush=True,
        )


def _report_done(row: dict[str, Any], case: CircuitCase, simulator: str, args) -> None:
    if args.progress:
        suffix = "" if row["status"] == "ok" else f" error={row['error']['type']}"
        print(f"[benchmark] done case={case.case_id} sim={simulator}{suffix}", file=sys.stderr, flush=True)


def _run_task(case, simulator, args, build_s, workload, existing, callback):
    if simulator == "merlin":
        row = run_merlin(
            case,
            shots=_shots(args, simulator),
            seed=args.seed,
            warmup_shots=args.warmup_shots,
            build_s=build_s,
            workload=workload,
            save_iter=args.save_iter,
            callback=callback,
            resume=existing,
        )
    else:
        interpreter = getattr(args, simulator.replace("-", "_") + "_python")
        row = run_external(
            case,
            backend=simulator,
            shots=_shots(args, simulator),
            seed=args.seed,
            warmup_shots=args.warmup_shots,
            build_s=build_s,
            workload=workload,
            interpreter=interpreter,
            timeout_s=args.timeout,
        )
    return row


def _execute(document, *, case, simulator, args, build_s, workload):
    existing = _resume_row(document, case, args, simulator, workload)
    if existing and existing.get("status") == "ok":
        if args.progress:
            print(f"[benchmark] resume {case.case_id} {simulator}: already complete", file=sys.stderr)
        return
    _report_start(case, simulator, args, workload)

    def checkpoint(row):
        upsert_result(document, row)
        if args.out:
            atomic_write(args.out, document)
        _report_checkpoint(row, case, simulator, args)

    row = _run_task(
        case, simulator, args, build_s, workload, existing,
        checkpoint if args.out or args.progress else None,
    )
    upsert_result(document, row)
    if args.out:
        atomic_write(args.out, document)
    _report_done(row, case, simulator, args)


def _init_worker(updates):
    global _WORKER_UPDATES
    _WORKER_UPDATES = updates


def _worker_task(case, simulator, args, build_s, workload, existing, key):
    callback = None if _WORKER_UPDATES is None else lambda row: _WORKER_UPDATES.put((key, row))
    return _run_task(case, simulator, args, build_s, workload, existing, callback)


def _tasks(args):
    for token in args.cases:
        for p_phys in args.p_grid:
            for workload in args.workload:
                started = time.perf_counter()
                case = _case_token(
                    token,
                    p_phys=p_phys,
                    target_scoring=workload == "target-scoring",
                    noisy_clifford=args.noisy_clifford,
                )
                build_s = time.perf_counter() - started
                for simulator in args.sim:
                    yield case, simulator, build_s, workload


def _run_parallel(document, args):
    context = multiprocessing.get_context("spawn")
    existing_order = {result_key(row): index for index, row in enumerate(document["results"])}
    task_order = {}
    completed = set()
    seen = set()
    tasks = iter(_tasks(args))

    def commit(key, row):
        if key in completed:
            return
        upsert_result(document, row)

        def order(item):
            item_key = result_key(item)
            if item_key in existing_order:
                return existing_order[item_key]
            return len(existing_order) + task_order[item_key]

        document["results"].sort(key=order)
        if args.out:
            atomic_write(args.out, document)

    updates = context.Queue() if args.out or args.progress else None
    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context,
        initializer=_init_worker, initargs=(updates,),
    )
    pending = {}

    def drain_updates():
        if updates is None:
            return
        while True:
            try:
                key, row = updates.get_nowait()
            except Empty:
                break
            commit(key, row)
            case, simulator = pending_by_key.get(key, (None, None))
            if case is not None:
                _report_checkpoint(row, case, simulator, args)

    pending_by_key = {}
    try:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * args.workers:
                task = next(tasks, None)
                if task is None:
                    exhausted = True
                    break
                case, simulator, build_s, workload = task
                key = _task_key(case, args, simulator, workload)
                if key in seen:
                    continue
                seen.add(key)
                task_order[key] = len(task_order)
                existing = _resume_row(document, case, args, simulator, workload)
                if existing and existing.get("status") == "ok":
                    if args.progress:
                        print(f"[benchmark] resume {case.case_id} {simulator}: already complete", file=sys.stderr)
                    continue
                _report_start(case, simulator, args, workload)
                future = pool.submit(
                    _worker_task, case, simulator, args, build_s, workload, existing, key
                )
                pending[future] = (key, case, simulator)
                pending_by_key[key] = (case, simulator)
            if pending:
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                drain_updates()
                for future in done:
                    key, case, simulator = pending.pop(future)
                    row = future.result()
                    commit(key, row)
                    completed.add(key)
                    _report_done(row, case, simulator, args)
                    pending_by_key.pop(key)
        drain_updates()
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        while any(not future.done() for future in pending):
            drain_updates()
            time.sleep(0.1)
        pool.shutdown(wait=True, cancel_futures=True)
        drain_updates()
        if updates is not None:
            updates.close()
    return document


def _invocation(args) -> dict[str, Any]:
    return {
        "command": "run",
        "cases": list(args.cases),
        "p_grid": list(args.p_grid),
        "workloads": list(args.workload),
        "noisy_clifford": bool(args.noisy_clifford),
        "simulators": list(args.sim),
        "shots": int(args.shots),
        "shots_merlin": args.shots_merlin,
        "shots_tsim": args.shots_tsim,
        "shots_xtim": args.shots_xtim,
        "shots_clifft": args.shots_clifft,
        "shots_symft": args.shots_symft,
        "seed": int(args.seed),
        "warmup_shots": int(args.warmup_shots),
        "save_iter": int(args.save_iter),
        "timeout": float(args.timeout),
    }


def run_benchmarks(args) -> dict[str, Any]:
    document = _prepare_document(
        args,
        kind="benchmark",
        invocation=_invocation(args),
    )
    if args.workers > 1:
        return _run_parallel(document, args)
    for case, simulator, build_s, workload in _tasks(args):
        _execute(document, case=case, simulator=simulator, args=args, build_s=build_s, workload=workload)
    return document


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=1, help="number of concurrent benchmark tasks (default: 1)")
    parser.add_argument(
        "--sim",
        type=_simulators,
        default=("merlin",),
        help="comma-separated merlin,tsim,xtim,clifft,symft or all",
    )
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--shots-merlin", type=int)
    parser.add_argument("--shots-tsim", type=int)
    parser.add_argument("--shots-xtim", type=int)
    parser.add_argument("--shots-clifft", type=int)
    parser.add_argument("--shots-symft", type=int)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmup-shots", type=int, default=0)
    parser.add_argument("--save-iter", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--tsim-python", default=os.environ.get("TSIM_PYTHON", sys.executable))
    parser.add_argument("--xtim-python", default=os.environ.get("XTIM_PYTHON", sys.executable))
    parser.add_argument("--clifft-python", default=os.environ.get("CLIFFT_PYTHON", sys.executable))
    parser.add_argument("--symft-python", default=os.environ.get("SYMFT_PYTHON", sys.executable))
    parser.add_argument("--out")
    parser.add_argument("--restart", action="store_true")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", default=sys.stderr.isatty())
    progress.add_argument("--no-progress", dest="progress", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run benchmark cases and collect all requested metrics")
    run.add_argument(
        "--cases", type=_csv, required=True,
        help="for example: 15to1,bh:2,bt27,cultivation-d3-t,cultivation-d5-t",
    )
    run.add_argument("--p-grid", type=_probabilities, default=(0.0, 0.001))
    run.add_argument(
        "--workload",
        type=_workloads,
        default=("target-scoring",),
        metavar="{detector-sampling,target-scoring,all}",
        help=f"comma-separated workloads, or all; {WORKLOAD_HELP}",
    )
    run.add_argument("--noisy-clifford", action="store_true", help="use gate depolarization for distillation cases")
    _add_execution_arguments(run)

    plot = commands.add_parser("plot", help="plot a benchmark result document")
    plot.add_argument("results")
    plot.add_argument("--metric", choices=tuple(METRIC_SPECS))
    plot.add_argument("--x", dest="x_axis", choices=("p-phys", "n-phys", "k"))
    plot.add_argument(
        "--workload",
        choices=WORKLOADS,
        help=f"filter results by workload; {WORKLOAD_HELP}",
    )
    plot.add_argument("--sim", dest="simulators", type=_simulators)
    plot.add_argument("--case", dest="cases", type=_csv)
    plot.add_argument("--p-phys", type=float)
    plot.add_argument("--save")
    plot.add_argument("--no-show", action="store_true")
    plot.add_argument("--log-x", action="store_true")
    plot.add_argument("--log-y", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plot":
        plot_file(
            args.results,
            metric=args.metric,
            x_axis=args.x_axis,
            workload=args.workload,
            simulators=args.simulators,
            cases=args.cases,
            p_phys=args.p_phys,
            save=args.save,
            show=not args.no_show,
            log_x=args.log_x,
            log_y=args.log_y,
        )
        return 0
    for name in ("shots", "warmup_shots", "save_iter"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    for simulator in SIMULATORS:
        value = getattr(args, "shots_" + simulator.replace("-", "_"))
        if value is not None and value < 0:
            raise ValueError(f"shot override for {simulator} must be nonnegative")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    document = run_benchmarks(args)
    json.dump(document, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0
