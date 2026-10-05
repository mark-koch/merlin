"""Print benchmark metrics as CSV tables with cases by simulator.

Run ``python3 benchmarks/results_to_csv.py`` to print all three tables, or
select one CSV table with ``--metric total_wall_s`` (or another metric).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_FILES = (RESULTS_DIR / "other.json",)
METRICS = {
    "total_wall_s": "timing",
    "sample_s_per_attempted_shot": "timing",
    "peak_rss_mb": "memory",
}
SIMULATOR_ORDER = ("tsim", "xtim", "clifft", "symft", "merlin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS, help="print one CSV table")
    args = parser.parse_args()

    results = []
    for path in INPUT_FILES:
        with path.open(encoding="utf-8") as handle:
            results.extend(json.load(handle)["results"])

    cases = sorted({row["case"]["id"] for row in results})
    present_simulators = {row["simulator"]["name"] for row in results}
    simulators = [name for name in SIMULATOR_ORDER if name in present_simulators]
    simulators.extend(sorted(present_simulators - set(SIMULATOR_ORDER)))
    values = {
        (row["case"]["id"], row["simulator"]["name"]): row
        for row in results
        if row.get("status") == "ok"
    }
    writer = csv.writer(sys.stdout)
    for index, metric in enumerate((args.metric,) if args.metric else METRICS):
        if index:
            writer.writerow([])
        writer.writerow([metric, *simulators])
        for case in cases:
            writer.writerow([
                case,
                *(
                    values.get((case, simulator), {}).get(METRICS[metric], {}).get(metric, "")
                    for simulator in simulators
                ),
            ])


if __name__ == "__main__":
    main()
