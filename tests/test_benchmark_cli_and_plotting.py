from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.plotting import plot_document
from benchmarks.protocols.distillation import build_distillation_case
from benchmarks.runners.merlin import run_merlin


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "benchmarks", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_help_describes_both_workloads():
    for command in ("run", "plot"):
        completed = _run(command, "-h")
        assert completed.returncode == 0, completed.stderr
        help_text = " ".join(completed.stdout.split())
        assert "detector-sampling measures detector acceptance" in help_text
        assert "target-scoring additionally scores the desired output" in help_text


def test_cli_json_stdout_progress_stderr_and_exact_resume(tmp_path):
    output = tmp_path / "results.json"
    arguments = (
        "run",
        "--cases",
        "15to1",
        "--p-grid",
        "0",
        "--shots",
        "2",
        "--save-iter",
        "1",
        "--out",
        str(output),
        "--progress",
    )
    first = _run(*arguments)
    assert first.returncode == 0, first.stderr
    document = json.loads(first.stdout)
    assert document["results"][0]["status"] == "ok"
    assert "[benchmark]" in first.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == document

    resumed = _run(*arguments)
    assert resumed.returncode == 0, resumed.stderr
    assert "already complete" in resumed.stderr
    changed_arguments = list(arguments)
    changed_arguments[changed_arguments.index("2")] = "3"
    changed_arguments[-1] = "--no-progress"
    changed = _run(*changed_arguments)
    assert changed.returncode != 0
    assert "does not exactly match" in changed.stderr


def test_checkpoint_can_be_extended_with_an_additional_simulator(tmp_path):
    output = tmp_path / "results.json"
    arguments = [
        "run",
        "--cases",
        "15to1",
        "--p-grid",
        "0",
        "--shots",
        "1",
        "--workload",
        "all",
        "--sim",
        "merlin",
        "--clifft-python",
        "/definitely/missing/python",
        "--timeout",
        "120",
        "--out",
        str(output),
        "--progress",
    ]
    first = _run(*arguments)
    assert first.returncode == 0, first.stderr
    first_document = json.loads(first.stdout)
    first_rows = first_document["results"]
    assert len(first_rows) == 2

    # Simulate a checkpoint written before SymFT-specific options existed.
    first_document["configuration"].pop("shots_symft")
    output.write_text(json.dumps(first_document), encoding="utf-8")

    arguments[arguments.index("merlin")] = "merlin,clifft"
    arguments[arguments.index("all")] = "detector-sampling"
    arguments[arguments.index("120")] = "240"
    extended = _run(*arguments)
    assert extended.returncode == 0, extended.stderr
    document = json.loads(extended.stdout)
    assert document["configuration"]["simulators"] == ["merlin", "clifft"]
    assert document["configuration"]["workloads"] == [
        "detector-sampling",
        "target-scoring",
    ]
    assert document["configuration"]["timeout"] == 240
    assert len(document["results"]) == 3
    assert document["results"][:2] == first_rows
    assert document["results"][2]["simulator"]["name"] == "clifft"
    assert document["results"][2]["error"]["type"] == "interpreter"
    assert "merlin: already complete" in extended.stderr

    arguments[arguments.index("merlin,clifft")] = "merlin"
    selected_subset = _run(*arguments)
    assert selected_subset.returncode == 0, selected_subset.stderr
    assert json.loads(selected_subset.stdout)["results"] == document["results"]


def test_checkpoint_appends_sparse_task_selections_and_preserves_shot_overrides(tmp_path):
    output = tmp_path / "sparse.json"
    common = ("run", "--shots", "1", "--seed", "7", "--out", str(output))
    initial = _run(
        *common, "--cases", "15to1", "--p-grid", "0", "--workload", "target-scoring",
        "--sim", "merlin", "--shots-merlin", "3", "--no-progress",
    )
    assert initial.returncode == 0, initial.stderr
    first_row = json.loads(initial.stdout)["results"][0]

    appended = _run(
        *common, "--cases", "bh:2", "--p-grid", "0.001", "--workload", "detector-sampling",
        "--sim", "clifft", "--shots-clifft", "4", "--clifft-python", "/definitely/missing/python",
        "--timeout", "240", "--workers", "2", "--no-progress",
    )
    assert appended.returncode == 0, appended.stderr
    document = json.loads(appended.stdout)
    assert document["configuration"]["cases"] == ["15to1", "bh:2"]
    assert document["configuration"]["p_grid"] == [0.0, 0.001]
    assert document["configuration"]["workloads"] == ["target-scoring", "detector-sampling"]
    assert document["configuration"]["simulators"] == ["merlin", "clifft"]
    assert document["configuration"]["shots_merlin"] == 3
    assert document["configuration"]["shots_clifft"] == 4
    assert document["configuration"]["timeout"] == 240
    assert len(document["results"]) == 2
    assert document["results"][0] == first_row
    assert document["results"][1]["case"]["id"] == "bh:2"
    assert document["results"][1]["simulator"]["name"] == "clifft"
    assert document["results"][1]["error"]["type"] == "interpreter"

    selected_again = _run(
        *common, "--cases", "15to1", "--p-grid", "0", "--workload", "target-scoring",
        "--sim", "merlin", "--shots-merlin", "3", "--workers", "2", "--progress",
    )
    assert selected_again.returncode == 0, selected_again.stderr
    assert "already complete" in selected_again.stderr
    resumed_document = json.loads(selected_again.stdout)
    assert resumed_document["results"] == document["results"]
    assert resumed_document["configuration"]["shots_clifft"] == 4
    assert json.loads(output.read_text()) == resumed_document

    for changed_option in (("--shots", "2"), ("--seed", "8"), ("--shots-merlin", "4")):
        changed = list(common)
        if changed_option[0] == "--shots-merlin":
            changed.extend(("--shots-merlin", "4"))
        else:
            changed[changed.index(changed_option[0]) + 1] = changed_option[1]
            changed.extend(("--shots-merlin", "3"))
        rejected = _run(
            *changed, "--cases", "15to1", "--p-grid", "0", "--workload", "target-scoring",
            "--sim", "merlin", "--no-progress",
        )
        assert rejected.returncode != 0
        assert "does not exactly match" in rejected.stderr


def test_one_run_collects_quality_and_timing_workloads():
    completed = _run(
        "run",
        "--cases",
        "15to1,bh:2",
        "--p-grid",
        "0",
        "--workload",
        "all",
        "--shots",
        "1",
        "--no-progress",
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["kind"] == "benchmark"
    assert len(document["results"]) == 4
    assert {row["case"]["id"] for row in document["results"]} == {"15to1", "bh:2"}
    assert {row["configuration"]["workload"] for row in document["results"]} == {
        "detector-sampling",
        "target-scoring",
    }
    assert all("total_wall_s" in row["timing"] for row in document["results"])
    scoring = [
        row
        for row in document["results"]
        if row["configuration"]["workload"] == "target-scoring"
    ]
    assert all(row["quality"]["target_infidelity"]["value"] == 0 for row in scoring)


def test_parallel_run_order_checkpoint_resume_and_external_error(tmp_path):
    output = tmp_path / "parallel.json"
    arguments = (
        "run", "--cases", "15to1,bh:2", "--p-grid", "0", "--workload", "all",
        "--sim", "merlin,clifft", "--clifft-python", "/definitely/missing/python",
        "--shots", "2", "--out", str(output), "--no-progress",
    )
    first = _run(*arguments, "--workers", "2")
    assert first.returncode == 0, first.stderr
    document = json.loads(first.stdout)
    assert json.loads(output.read_text()) == document
    assert len(document["results"]) == 8
    assert [(row["case"]["id"], row["configuration"]["workload"], row["simulator"]["name"])
            for row in document["results"]] == [
                (case, workload, simulator)
                for case in ("15to1", "bh:2")
                for workload in ("detector-sampling", "target-scoring")
                for simulator in ("merlin", "clifft")
            ]
    assert all(row["status"] == "ok" for row in document["results"] if row["simulator"]["name"] == "merlin")
    assert all(row["error"]["type"] == "interpreter" for row in document["results"] if row["simulator"]["name"] == "clifft")
    assert "workers" not in document["configuration"]

    resumed = _run(*arguments, "--workers", "1")
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["results"][:1] == document["results"][:1]
    assert json.loads(output.read_text()) == json.loads(resumed.stdout)


def test_parallel_merlin_partial_checkpoint_resume(tmp_path):
    output = tmp_path / "partial.json"
    arguments = (
        "run", "--cases", "15to1,bh:2", "--p-grid", "0.05", "--shots", "5",
        "--seed", "11", "--save-iter", "2", "--out", str(output),
    )
    initial = _run(*arguments)
    assert initial.returncode == 0, initial.stderr
    document = json.loads(initial.stdout)
    partials = []
    expected = run_merlin(
        build_distillation_case("15to1", p_phys=0.05),
        shots=5, seed=11, save_iter=2, callback=partials.append,
    )
    document["results"][0] = partials[0]
    output.write_text(json.dumps(document), encoding="utf-8")

    resumed = _run(*arguments, "--workers", "2", "--progress")
    assert resumed.returncode == 0, resumed.stderr
    rows = json.loads(resumed.stdout)["results"]
    assert rows[0]["counts"] == expected["counts"]
    assert rows[0]["partial"] is False
    assert rows[0]["timing"]["resume_replay_s"] > 0
    assert "already complete" in resumed.stderr
    assert json.loads(output.read_text())["results"] == rows


@pytest.mark.parametrize("workers", ("0", "-1"))
def test_worker_count_must_be_positive(workers):
    result = _run("run", "--cases", "15to1", "--workers", workers)
    assert result.returncode != 0
    assert "--workers must be positive" in result.stderr


def test_cultivation_cases_run_from_cli(tmp_path):
    output = tmp_path / "cultivation.json"
    result = _run(
        "run", "--cases", "cultivation-d3-t,cultivation-d5-t",
        "--p-grid", "0", "--workload", "all", "--sim", "merlin",
        "--shots", "1", "--out", str(output), "--no-progress", timeout=60,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert len(document["results"]) == 4
    assert {row["case"]["id"] for row in document["results"]} == {
        "cultivation-d3-t", "cultivation-d5-t",
    }
    assert {row["configuration"]["workload"] for row in document["results"]} == {
        "detector-sampling", "target-scoring",
    }
    assert all(row["status"] == "ok" for row in document["results"])
    assert all(row["counts"]["accepted"] == 1 for row in document["results"])
    assert json.loads(output.read_text()) == document


def test_old_time_command_is_not_an_alias():
    completed = _run("time", "--cases", "15to1")
    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


@pytest.mark.external
@pytest.mark.skipif(
    not os.environ.get("RUN_EXTERNAL_BENCHMARKS"),
    reason="set RUN_EXTERNAL_BENCHMARKS=1",
)
def test_real_tsim_and_xtim_smoke():
    python = (
        os.environ.get("TSIM_PYTHON") or os.environ.get("XTIM_PYTHON") or sys.executable
    )
    completed = _run(
        "run",
        "--cases",
        "15to1",
        "--p-grid",
        "0",
        "--shots",
        "1",
        "--sim",
        "tsim,xtim",
        "--tsim-python",
        os.environ.get("TSIM_PYTHON", python),
        "--xtim-python",
        os.environ.get("XTIM_PYTHON", python),
        "--no-progress",
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)["results"]
    assert all(row["status"] == "ok" for row in rows)
    assert {row["simulator"]["name"]: row["simulator"]["version"] for row in rows} == {
        "tsim": "0.1.5",
        "xtim": "2.7.0",
    }
    assert all(row["quality"]["target_infidelity"]["value"] == 0 for row in rows)


@pytest.mark.external
@pytest.mark.skipif(
    not os.environ.get("RUN_CLIFFT_BENCHMARKS"), reason="set RUN_CLIFFT_BENCHMARKS=1"
)
def test_real_clifft_smoke():
    completed = _run(
        "run",
        "--cases",
        "15to1",
        "--p-grid",
        "0",
        "--shots",
        "2",
        "--workload",
        "all",
        "--sim",
        "clifft",
        "--clifft-python",
        os.environ.get("CLIFFT_PYTHON", sys.executable),
        "--no-progress",
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)["results"]
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["simulator"]["name"] == "clifft" for row in rows)
    assert all(row["simulator"]["version"] for row in rows)
    assert all(
        row["counts"]["attempted"] == row["counts"]["accepted"] == 2 for row in rows
    )
    scoring = next(
        row for row in rows if row["configuration"]["workload"] == "target-scoring"
    )
    assert scoring["counts"]["accepted_and_target_pass"] == 2
    assert scoring["quality"]["target_infidelity"]["value"] == 0


@pytest.mark.external
@pytest.mark.skipif(
    not os.environ.get("RUN_SYMFT_BENCHMARKS"), reason="set RUN_SYMFT_BENCHMARKS=1"
)
def test_real_symft_smoke():
    completed = _run(
        "run",
        "--cases",
        "15to1,bh:2",
        "--p-grid",
        "0",
        "--shots",
        "2",
        "--workload",
        "all",
        "--sim",
        "symft",
        "--symft-python",
        os.environ.get("SYMFT_PYTHON", sys.executable),
        "--no-progress",
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)["results"]
    assert len(rows) == 4
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["simulator"]["name"] == "symft" for row in rows)
    assert all(row["simulator"]["version"] for row in rows)
    assert all(
        row["counts"]["attempted"] == row["counts"]["accepted"] == 2 for row in rows
    )
    scoring = [
        row for row in rows if row["configuration"]["workload"] == "target-scoring"
    ]
    assert all(row["counts"]["accepted_and_target_pass"] == 2 for row in scoring)
    assert all(row["quality"]["target_infidelity"]["value"] == 0 for row in scoring)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_BENCHMARKS"), reason="set RUN_SLOW_BENCHMARKS=1"
)
@pytest.mark.parametrize("pair", ("bt27", "bt81"))
def test_full_code_switching_zero_noise_smoke(pair):
    completed = _run(
        "run",
        "--cases",
        pair,
        "--p-grid",
        "0",
        "--shots",
        "1",
        "--timeout",
        "840",
        "--no-progress",
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["results"][0]["status"] == "ok"
