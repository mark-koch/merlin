"""Unified plotting for quality and timing result documents."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import tempfile
from typing import Any

from .storage import load_document


METRIC_SPECS = {
    "acceptance": (("quality", "acceptance", "value"), ("quality", "acceptance", "stderr"), "Acceptance"),
    "target-infidelity": (("quality", "target_infidelity", "value"), ("quality", "target_infidelity", "stderr"), "Target infidelity"),
    "total-time": (("timing", "total_wall_s"), None, "Total time (s)"),
    "sample-time": (("timing", "sample_s"), None, "Sampling time (s)"),
    "sample-time-per-attempt": (
        ("timing", "sample_s_per_attempted_shot"),
        None,
        "Sampling time per shot (s)",
    ),
    "time-per-attempt": (("timing", "total_s_per_attempted_shot"), None, "Total time / attempted shot (s)"),
    "time-per-accepted": (("timing", "sample_s_per_accepted_shot"), None, "Sampling time / accepted shot (s)"),
    "peak-rss": (("memory", "peak_rss_mb"), None, "Peak Memory (MiB)"),
    "chi": (
        ("simulator", "program", "stabilizer_term_count"),
        None,
        "Stabilizer term count χ",
    ),
    "magic-rank": (
        ("simulator", "program", "magic_rank"),
        None,
        "Magic rank r = log₂(χ)",
    ),
}


def _get(row: dict[str, Any], path: tuple[str, ...]):
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _series_label(row: dict[str, Any], x_axis: str) -> str:
    case = row["case"]["id"]
    simulator = row["simulator"]["name"]
    workload = row.get("configuration", {}).get("workload")
    if x_axis == "p-phys":
        parts = (case, simulator, workload)
    else:
        p_phys = row.get("configuration", {}).get("p_phys")
        parts = (simulator,)
    return " / ".join(str(part) for part in parts if part is not None)


def _x_value(row: dict[str, Any], x_axis: str):
    if x_axis == "n-phys":
        return row.get("case", {}).get("n_phys")
    if x_axis == "k":
        return row.get("case", {}).get("k_logical")
    return row.get("configuration", {}).get("p_phys")


def _case_matches(case_id: str, filters: tuple[str, ...] | None) -> bool:
    if not filters:
        return True
    return any(case_id == value or (value == "bh" and case_id.startswith("bh:")) for value in filters)


def plot_document(
    document: dict[str, Any],
    *,
    metric: str | None = None,
    x_axis: str | None = None,
    workload: str | None = None,
    simulators: tuple[str, ...] | None = None,
    cases: tuple[str, ...] | None = None,
    p_phys: float | None = None,
    save: str | None = None,
    show: bool = True,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    if metric is None:
        has_target_metric = any(
            _get(row, ("quality", "target_infidelity", "value")) is not None
            for row in document.get("results", [])
        )
        metric = "target-infidelity" if has_target_metric else "total-time"
    timing_metric = metric not in {"acceptance", "target-infidelity"}
    x_axis = x_axis or ("n-phys" if timing_metric else "p-phys")
    if metric not in METRIC_SPECS:
        raise ValueError(f"unknown metric: {metric!r}")
    if x_axis not in {"n-phys", "p-phys", "k"}:
        raise ValueError(f"unknown x axis: {x_axis!r}")
    value_path, error_path, ylabel = METRIC_SPECS[metric]

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "merlin-benchmark-mpl"))
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 20,
    })

    groups: dict[str, list[tuple[float, float, float | None, dict[str, Any]]]] = defaultdict(list)
    for row in document.get("results", []):
        if row.get("status") != "ok":
            continue
        configuration = row.get("configuration", {})
        case_id = row.get("case", {}).get("id", "")
        if workload is not None and configuration.get("workload") != workload:
            continue
        if simulators and row.get("simulator", {}).get("name") not in simulators:
            continue
        if not _case_matches(case_id, cases):
            continue
        if p_phys is not None and configuration.get("p_phys") != p_phys:
            continue
        x_value = _x_value(row, x_axis)
        y_value = _get(row, value_path)
        if x_value is None or y_value is None:
            continue
        error = _get(row, error_path) if error_path else None
        groups[_series_label(row, x_axis)].append((float(x_value), float(y_value), None if error is None else float(error), row))
    if not groups:
        raise ValueError(f"no successful rows contain metric {metric!r}")

    order = ["tsim", "xtim", "clifft", "symft", "merlin"]
    markers = ["o", "*", "v", "^", "s"]

    figure, axis = plt.subplots(figsize=(5.5, 6))
    for label, values in sorted(groups.items(), key=lambda g: order.index(g[0])):
        marker = markers[order.index(label)]
        values.sort(key=lambda item: item[0])
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        errors = [item[2] or 0.0 for item in values]
        axis.errorbar(xs, ys, yerr=errors if error_path else None, marker=marker, label=label)

    if metric == "target-infidelity" and x_axis == "p-phys":
        clean_15to1 = [
            row
            for values in groups.values()
            for *_unused, row in values
            if row.get("case", {}).get("id") == "15to1"
            and not row.get("case", {}).get("noisy_clifford", False)
        ]
        if clean_15to1:
            xs = sorted({float(row["configuration"]["p_phys"]) for row in clean_15to1})
            axis.plot(xs, [35 * p**3 for p in xs], linestyle="--", label=r"$35p^3$")

    x_labels = {
        "n-phys": "Physical qubits",
        "p-phys": "Physical error probability",
        "k": "Logical outputs k",
    }
    axis.set_xlabel(x_labels[x_axis])
    axis.set_ylabel(ylabel)
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)

    label_params = axis.get_legend_handles_labels() 
    figl, axl = plt.subplots(figsize=(10, 0.5))
    axl.axis(False)
    axl.legend(*label_params, loc="center", ncols=5, frameon=False)
    figl.savefig(save + "-legend.pdf")

    figure.tight_layout()
    if save:
        figure.savefig(save)
    if show:
        plt.show()
    plt.close(figure)


def plot_file(path: str, **kwargs) -> None:
    plot_document(load_document(path), **kwargs)
