"""Reweight cultivation checkpoints and plot errors versus attempts per kept shot."""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom

if __package__:
    from .common import ROOT, load_checkpoint, validate_checkpoint
else:
    from common import ROOT, load_checkpoint, validate_checkpoint


def reweight(data, physical_error_rate):
    if not math.isfinite(physical_error_rate) or not 0 <= physical_error_rate <= 1:
        raise ValueError("Physical error rates must be between 0 and 1")
    groups = validate_checkpoint(data)
    totals = np.array([sum(batch["total"] for batch in batches)
                       for batches in groups.values()], dtype=float)
    if np.any(totals == 0):
        raise ValueError("Require samples at every num_faults from 1 through max_num_faults")
    accepted = np.array([sum(batch["accepted"] for batch in batches)
                         for batches in groups.values()], dtype=float)
    errors = np.array([sum(batch["errors"] for batch in batches)
                       for batches in groups.values()], dtype=float)
    maximum = data["config"]["max_num_faults"]
    sites = data["metadata"]["num_fault_sites"]
    weights = binom.pmf(np.arange(maximum + 1), sites, physical_error_rate)
    # For the supplied circuits, num_faults=0 has acceptance 1 and error 0.
    survival = float(weights[0] + weights[1:] @ (accepted / totals))
    numerator = float(weights[1:] @ (errors / totals))
    fractions = errors / totals
    variance = float(np.sum(weights[1:] ** 2 * fractions * (1 - fractions) / totals))
    tail = float(binom.sf(maximum, sites, physical_error_rate))
    rate = numerator / survival if survival > 0 else math.nan
    # A plug-in variance of zero is not evidence of certainty for rare events.
    interval = (1.96 * math.sqrt(variance) / survival
                if survival > 0 and variance > 0 else math.nan)
    significant_tail = survival == 0 and tail > 0
    if survival > 0:
        precision = max(rate, interval if math.isfinite(interval) else 0)
        significant_tail = tail / survival > max(0.1 * precision, 1e-12)
    return {
        "physical_error_rate": physical_error_rate,
        "survival": survival,
        "attempts_per_kept_shot": 1 / survival if survival > 0 else math.nan,
        "logical_error_rate": rate,
        "ci95_half_width": interval,
        "omitted_probability": tail,
        "significant_tail": significant_tail,
    }


def comparison_datasets(t_path, s_path, clifft_t_path=None, clifft_s_path=None):
    datasets = []
    reference = {}
    for circuit, backend, path in (
        ("t", "merlin", t_path), ("s", "merlin", s_path),
        ("t", "clifft", clifft_t_path), ("s", "clifft", clifft_s_path),
    ):
        if path is None:
            continue
        data = load_checkpoint(path)
        if (data["metadata"]["circuit"] != circuit
                or data.get("backend", {}).get("name") != backend):
            raise ValueError(f"Expected a {circuit.upper()} circuit checkpoint from {backend}: {path}")
        if backend == "merlin":
            reference[circuit] = data["metadata"]
        else:
            for name in ("circuit_sha256", "num_fault_sites", "site_probability",
                         "num_detectors", "num_observables", "num_qubits"):
                if (name not in data["metadata"] or name not in reference[circuit]
                        or data["metadata"][name] != reference[circuit][name]):
                    raise ValueError(f"{circuit.upper()} circuit {name} differs between backends")
        datasets.append((circuit, backend, data))
    return datasets


def make_plot(t_path, s_path, physical_error_rates, output_prefix, *,
              clifft_t_path=None, clifft_s_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    datasets = comparison_datasets(t_path, s_path, clifft_t_path, clifft_s_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    any_zero = False
    positive_rates = []
    for curve_index, (circuit, backend, data) in enumerate(datasets):
        color, marker = ("tab:red", "s") if circuit == "t" else ("tab:blue", "D")
        linestyle = "-" if backend == "merlin" else "--"
        facecolor = color if backend == "merlin" else "none"
        identity = f"{circuit.upper()} ({backend})"
        estimates = [reweight(data, p) for p in sorted(set(physical_error_rates))]
        xs, ys = [], []
        for estimate in estimates:
            rate = estimate["logical_error_rate"]
            attempts = estimate["attempts_per_kept_shot"]
            interval = estimate["ci95_half_width"]
            probability = estimate["physical_error_rate"]
            tail = estimate["omitted_probability"]
            truncated = estimate["significant_tail"]
            uncertainty = f"{interval:.6g}" if math.isfinite(interval) else "unavailable"
            print(f"{identity} p={probability:g}: attempts per kept shot={attempts:.6g}; "
                  f"logical error rate={rate:.6g}; approximate 95% half-width={uncertainty}; "
                  f"omitted probability={tail:.3g}"
                  + ("; WARNING: truncation may materially affect this estimate" if truncated else ""),
                  file=sys.stderr)
            if not math.isfinite(rate):
                print(f"{identity} p={probability:g}: zero survival; estimate undefined",
                      file=sys.stderr)
                xs.append(math.nan)
                ys.append(math.nan)
                continue
            label = f"{probability:g}"
            xs.append(attempts)
            ys.append(rate if rate > 0 else math.nan)
            if rate == 0:
                any_zero = True
                # Axis-relative placement deliberately assigns no numerical error rate.
                vertical = 0.025 + 0.075 * curve_index
                ax.plot(attempts, vertical, marker="v", color=color, markerfacecolor=facecolor,
                        transform=ax.get_xaxis_transform(), linestyle="none")
                ax.annotate(label, (attempts, vertical), xycoords=ax.get_xaxis_transform(),
                            xytext=(4, 7), textcoords="offset points", color=color, fontsize=8)
                print(f"{identity} p={probability:g}: no weighted observed errors; "
                      "confidence interval unavailable", file=sys.stderr)
            else:
                positive_rates.append(rate)
                if math.isfinite(interval):
                    # Clip the normal approximation to the probability range for display.
                    ax.errorbar(attempts, rate,
                                yerr=[[min(interval, rate)],
                                      [min(interval, 1 - rate)]],
                                fmt="none", ecolor=color, capsize=3)
                else:
                    print(f"{identity} p={probability:g}: degenerate plug-in variance; "
                          "confidence interval unavailable", file=sys.stderr)
                offset = 6 if circuit == "t" else -12
                ax.annotate(label, (attempts, rate),
                            xytext=(5 if backend == "merlin" else -5, offset),
                            ha="left" if backend == "merlin" else "right",
                            textcoords="offset points", color=color, fontsize=8)
        ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle,
                markerfacecolor=facecolor, label=f"{circuit.upper()} circuit ({backend})")
    if not positive_rates:
        ax.set_ylim(1e-12, 1)
        ax.text(0.5, 0.5, "No positive logical error estimates.\n"
                "More samples or broader fault-count coverage may be needed.",
                transform=ax.transAxes, ha="center", va="center", color="0.4")
    ax.set_xlabel("Attempts per kept shot")
    ax.set_ylabel("Logical error probability per kept shot")
    ax.set_title("Distance-3 magic state cultivation")
    ax.grid(True, which="both", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if any_zero:
        handles.append(Line2D([], [], color="gray", marker="v", linestyle="none"))
        labels.append("No observed errors (bottom markers have no y value)")
    ax.legend(handles, labels, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("png", "pdf"):
            destination = Path(f"{output_prefix}.{extension}")
            fig.savefig(destination, dpi=200)
            print(f"Plot: {destination}", file=sys.stderr)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-data", type=Path, default=ROOT / "samples_t.json")
    parser.add_argument("--s-data", type=Path, default=ROOT / "samples_s.json")
    parser.add_argument("--clifft-t-data", type=Path, default=ROOT / "samples_clifft_t.json")
    parser.add_argument("--clifft-s-data", type=Path, default=ROOT / "samples_clifft_s.json")
    parser.add_argument("--physical-error-rates", type=float, nargs="+",
                        default=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01])
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "cultivation")
    args = parser.parse_args()
    try:
        make_plot(args.t_data, args.s_data, args.physical_error_rates, args.output_prefix,
                  clifft_t_path=args.clifft_t_data, clifft_s_path=args.clifft_s_data)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
