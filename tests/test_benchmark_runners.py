from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.plotting import plot_document
from benchmarks.protocols.code_switching import BBTPair, build_code_switching_case
from benchmarks.protocols.css import CSSCode
from benchmarks.protocols.cultivation import build_cultivation_case
from benchmarks.protocols.distillation import build_distillation_case
from benchmarks.runners import common as common_module
from benchmarks.runners import merlin as dcp_runner_module
from benchmarks.runners import external as external_module
from benchmarks.runners.common import joint_target_successes, quality_metrics
from benchmarks.runners.merlin import run_merlin
from benchmarks.runners.external import run_external
from benchmarks.storage import (
    atomic_write,
    load_document,
    new_document,
    result_key,
    upsert_result,
)


def tiny_bbt_pair() -> BBTPair:
    return BBTPair(
        "tiny",
        1,
        1,
        CSSCode(
            "tiny_3d",
            np.eye(3, dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            1,
        ),
        CSSCode(
            "tiny_2d",
            np.eye(2, dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            1,
        ),
        1,
        1,
    )


def test_merlin_zero_noise_and_known_accepted_rm15_fault():
    ideal = run_merlin(build_distillation_case("15to1"), shots=4, seed=3)
    assert ideal["counts"] == {
        "attempted": 4,
        "accepted": 4,
        "accepted_and_target_pass": 4,
    }
    assert ideal["quality"]["target_infidelity"]["value"] == 0
    assert ideal["timing"]["total_wall_s"] >= ideal["timing"]["sample_s"]
    assert ideal["memory"]["peak_rss_bytes"] > 0

    logical_z = run_merlin(
        build_distillation_case("15to1", deterministic_z_faults=(0, 1, 2)),
        shots=4,
        seed=3,
    )
    assert logical_z["counts"]["accepted"] == 4
    assert logical_z["quality"]["target_infidelity"]["value"] == 1


@pytest.mark.parametrize("distance", (3, 5))
def test_cultivation_zero_noise_is_accepted_and_scores_target(distance):
    case = build_cultivation_case(distance, p_phys=0)
    row = run_merlin(case, shots=2, seed=7)
    assert row["counts"] == {
        "attempted": 2,
        "accepted": 2,
        "accepted_and_target_pass": 2,
    }
    detector_only = run_merlin(
        build_cultivation_case(distance, p_phys=0, target_scoring=False),
        shots=1, seed=7, workload="detector-sampling",
    )
    assert detector_only["counts"]["accepted"] == 1
    assert detector_only["counts"]["accepted_and_target_pass"] is None


def test_zero_noise_bh_and_tiny_code_switching_target_infidelity():
    for case in (
        build_distillation_case("bh", k=2),
        build_distillation_case("bh", k=4),
        build_code_switching_case(tiny_bbt_pair()),
    ):
        row = run_merlin(case, shots=4, seed=7)
        assert row["counts"]["accepted"] > 0
        assert row["quality"]["target_infidelity"]["value"] == 0


@pytest.mark.parametrize("pair", ("bt27", "bt81"))
@pytest.mark.slow
def test_bbt_zero_noise_has_compatible_measurements_and_exact_target(pair):
    row = run_merlin(build_code_switching_case(pair), shots=1, seed=7)
    assert row["counts"] == {
        "attempted": 1,
        "accepted": 1,
        "accepted_and_target_pass": 1,
    }
    assert row["quality"]["target_infidelity"]["value"] == 0


def test_tiny_code_switching_known_logical_fault_is_accepted_and_fails_target():
    case = build_code_switching_case(tiny_bbt_pair(), deterministic_z_faults=(3,))
    row = run_merlin(case, shots=3, seed=5)
    assert row["counts"]["accepted"] == 3
    assert row["counts"]["accepted_and_target_pass"] == 0
    assert row["quality"]["target_infidelity"]["value"] == 1


def test_partial_callback_and_exact_rng_resume():
    case = build_distillation_case("15to1", p_phys=0.05)
    saved = []
    complete = run_merlin(case, shots=5, seed=11, save_iter=2, callback=saved.append)
    assert [row["counts"]["attempted"] for row in saved] == [2, 4, 5]
    assert [row["partial"] for row in saved] == [True, True, False]
    resumed = run_merlin(case, shots=5, seed=11, save_iter=2, resume=saved[0])
    assert resumed["counts"] == complete["counts"]
    assert resumed["timing"]["resume_replay_s"] > 0


def test_quality_metrics_and_joint_multi_output_rule():
    metrics = quality_metrics(attempted=4, accepted=3, target_successes=2)
    assert metrics["acceptance"]["value"] == 3 / 4
    assert metrics["acceptance"]["trials"] == 4
    assert metrics["target_infidelity"]["value"] == pytest.approx(1 / 3)
    assert metrics["target_infidelity"]["trials"] == 3

    no_accepts = quality_metrics(attempted=4, accepted=0, target_successes=0)
    assert no_accepts["target_infidelity"]["value"] is None
    assert no_accepts["target_infidelity"]["stderr"] is None

    observables = np.asarray(
        [[False, False], [True, False], [False, True], [True, True]],
        dtype=bool,
    )
    assert joint_target_successes(observables) == 1


def test_dcp_timing_fields_have_deterministic_boundaries(monkeypatch):
    ticks = iter((0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 12.0))
    monkeypatch.setattr(
        dcp_runner_module,
        "time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )
    row = run_merlin(
        build_distillation_case("15to1"),
        shots=2,
        seed=1,
        warmup_shots=1,
        build_s=5.0,
    )
    assert row["timing"] == {
        "construction_s": 5.0,
        "compile_s": 2.0,
        "warmup_s": 3.0,
        "sample_s": 4.0,
        "resume_replay_s": 0.0,
        "total_wall_s": 14.0,
        "sample_s_per_attempted_shot": 2.0,
        "total_s_per_attempted_shot": 7.0,
        "sample_s_per_accepted_shot": 2.0,
    }


def test_external_timing_uses_subprocess_wall_boundary(monkeypatch):
    payload = {
        "compile_s": 0.5,
        "warmup_s": 0.25,
        "sample_s": 1.25,
        "attempted": 2,
        "accepted": 2,
        "target_successes": 2,
        "n_detectors": 14,
        "n_observables": 1,
        "version": "fake",
        "python": "3.13",
        "peak_rss_bytes": 4096,
    }
    monkeypatch.setattr(
        external_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )
    ticks = iter((10.0, 14.0))
    monkeypatch.setattr(
        external_module,
        "time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )
    row = run_external(
        build_distillation_case("15to1"),
        backend="tsim",
        shots=2,
        seed=1,
        build_s=2.0,
        interpreter=sys.executable,
    )
    assert row["timing"] == {
        "construction_s": 2.0,
        "compile_s": 0.5,
        "warmup_s": 0.25,
        "sample_s": 1.25,
        "subprocess_wall_s": 4.0,
        "total_wall_s": 6.0,
        "sample_s_per_attempted_shot": 0.625,
        "total_s_per_attempted_shot": 3.0,
        "sample_s_per_accepted_shot": 0.625,
    }


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", 123), ("linux", 123 * 1024)],
)
def test_peak_rss_platform_unit_conversion(monkeypatch, platform, expected):
    monkeypatch.setattr(
        common_module.resource,
        "getrusage",
        lambda who: SimpleNamespace(ru_maxrss=123),
    )
    monkeypatch.setattr(common_module.sys, "platform", platform)
    assert common_module.peak_rss_bytes() == expected


def test_atomic_document_round_trip_and_upsert(tmp_path):
    path = tmp_path / "results.json"
    document = new_document(kind="quality", configuration={"shots": 2})
    row = run_merlin(build_distillation_case("15to1"), shots=2, seed=1)
    upsert_result(document, row)
    atomic_write(path, document)
    assert load_document(path) == document
    replacement = dict(row)
    replacement["status"] = "replacement"
    upsert_result(document, replacement)
    assert len(document["results"]) == 1
    assert document["results"][0]["status"] == "replacement"


def test_timeout_is_not_part_of_result_identity():
    first = {
        "case": {"id": "15to1"},
        "simulator": {"name": "clifft"},
        "configuration": {"shots": 10, "timeout_s": 120.0},
    }
    retried = {
        "case": {"id": "15to1"},
        "simulator": {"name": "clifft"},
        "configuration": {"shots": 10, "timeout_s": 240.0},
    }
    assert result_key(first) == result_key(retried)
    document = {"results": [first]}
    upsert_result(document, retried)
    assert document["results"] == [retried]


def test_missing_external_interpreter_is_a_structured_error():
    row = run_external(
        build_distillation_case("15to1"),
        backend="xtim",
        shots=1,
        seed=1,
        interpreter="/definitely/missing/python",
    )
    assert row["status"] == "error"
    assert row["error"]["type"] == "interpreter"


def test_missing_external_dependency_is_a_structured_error(monkeypatch):
    monkeypatch.setattr(
        external_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'xtim'",
        ),
    )
    row = run_external(
        build_distillation_case("15to1"),
        backend="xtim",
        shots=1,
        seed=1,
        interpreter=sys.executable,
    )
    assert row["status"] == "error"
    assert row["error"]["type"] == "missing_dependency"


def test_fake_external_adapter_uses_joint_observable_rule(tmp_path, monkeypatch):
    module = """
import numpy as np
__version__ = "fake"
class Circuit:
    def __init__(self, text): self.text = text
    def reference_info(self):
        return {"cached": True, "chi": 8, "n": 15, "source": "fake"}
    def compile_detector_sampler(self, seed=0): return Sampler()
class Sampler:
    def engine_report(self):
        return {"engine": "exact", "reason": "fake exact engine"}
    def sample(self, shots, separate_observables=False, **kwargs):
        det = np.zeros((shots, 14), dtype=bool)
        obs = np.zeros((shots, 1), dtype=bool)
        if shots > 1:
            det[1, 0] = True
            obs[0, 0] = True
        return det, obs
"""
    (tmp_path / "xtim.py").write_text(module, encoding="utf-8")
    old_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + old_path)
    row = run_external(
        build_distillation_case("15to1"),
        backend="xtim",
        shots=3,
        seed=1,
        interpreter=sys.executable,
    )
    assert row["status"] == "ok"
    assert row["simulator"]["version"] == "fake"
    assert row["simulator"]["engine"] == {
        "engine": "exact",
        "reason": "fake exact engine",
    }
    assert row["simulator"]["program"] == {
        "stabilizer_term_count": 8,
        "magic_rank": 3,
        "metric_source": "cached_reference",
    }
    assert row["counts"] == {
        "attempted": 3,
        "accepted": 2,
        "accepted_and_target_pass": 1,
    }
    assert row["quality"]["acceptance"]["value"] == 2 / 3
    assert row["quality"]["target_infidelity"]["value"] == 0.5

    chi_path = tmp_path / "chi-vs-k.png"
    plot_document(
        {"results": [row]},
        metric="chi",
        x_axis="k",
        workload="target-scoring",
        simulators=("xtim",),
        save=str(chi_path),
        show=False,
    )
    assert chi_path.stat().st_size > 0

    magic_rank_path = tmp_path / "magic-rank-vs-k.png"
    plot_document(
        {"results": [row]},
        metric="magic-rank",
        x_axis="k",
        workload="target-scoring",
        simulators=("xtim",),
        save=str(magic_rank_path),
        show=False,
    )
    assert magic_rank_path.stat().st_size > 0


def test_fake_clifft_adapter_uses_native_postselection_and_joint_error_count(
    tmp_path, monkeypatch
):
    module = """
def version(): return "fake-clifft"
class Program:
    num_detectors = 14
    num_observables = 1
    num_actions = 123
    peak_active_width = 7
class Result:
    def __init__(self, shots):
        self.total_shots = shots
        self.passed_shots = shots - 1
        self.logical_errors = 1
def compile(text, postselection_mask=None, normalize_syndromes=False):
    assert len(postselection_mask) == 14
    assert normalize_syndromes is True
    return Program()
def sample_survivors(program, shots, seed=None):
    return Result(shots)
"""
    (tmp_path / "clifft.py").write_text(module, encoding="utf-8")
    old_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + old_path)
    row = run_external(
        build_distillation_case("15to1"),
        backend="clifft",
        shots=3,
        seed=1,
        interpreter=sys.executable,
    )
    assert row["status"] == "ok"
    assert row["simulator"]["version"] == "fake-clifft"
    assert row["simulator"]["program"] == {"num_actions": 123, "peak_active_width": 7}
    assert row["counts"] == {
        "attempted": 3,
        "accepted": 2,
        "accepted_and_target_pass": 1,
    }
    assert row["quality"]["acceptance"]["value"] == 2 / 3
    assert row["quality"]["target_infidelity"]["value"] == 0.5


def test_fake_symft_adapter_uses_native_counts(tmp_path, monkeypatch):
    module = """
__version__ = "fake-symft"
def simd_backend(): return "fake-vector"
def cuda_enabled(): return False
def active_cuda_backend(): return "disabled"
class CountsSampler:
    info = {"max_active_qubits": 4, "backend": "batch"}
    preprocessing_timing = {"plan_s": 0.01}
    def sample(self, shots, stream_id=None):
        return {"shots": shots, "accepted": shots - 1, "logical_errors": 1}
class Circuit:
    num_detectors = 14
    num_observables = 1
    def __init__(self, text): self.text = text
    def compile_counts_sampler(self, **kwargs):
        assert kwargs == {"batch": True, "observable": 0, "postselect_detectors": True, "threads": 1}
        return CountsSampler()
"""
    (tmp_path / "symft.py").write_text(module, encoding="utf-8")
    old_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + old_path)
    row = run_external(
        build_distillation_case("15to1"),
        backend="symft",
        shots=3,
        seed=1,
        interpreter=sys.executable,
    )
    assert row["status"] == "ok"
    assert row["simulator"]["version"] == "fake-symft"
    assert row["simulator"]["simd_backend"] == "fake-vector"
    assert row["simulator"]["program"]["aggregation"] == "native_counts"
    assert row["counts"] == {
        "attempted": 3,
        "accepted": 2,
        "accepted_and_target_pass": 1,
    }


def test_fake_symft_adapter_preserves_joint_multi_observable_rule(
    tmp_path, monkeypatch
):
    module = """
import numpy as np
__version__ = "fake-symft"
def simd_backend(): return "scalar"
def cuda_enabled(): return False
def active_cuda_backend(): return "disabled"
class Sampler:
    num_qubits = 15
    num_measurements = 14
    num_detectors = 12
    max_active_qubits = 3
    def sample(self, shots, seed=None):
        result = np.zeros((shots, 14), dtype=bool)
        result[0, 12] = True
        result[1, 0] = True
        return result
class Circuit:
    num_detectors = 12
    num_observables = 2
    detectors = [{"records": (i + 1,)} for i in range(12)]
    observables = [
        {"index": 0, "records": (13,)},
        {"index": 1, "records": (14,)},
    ]
    def __init__(self, text): self.text = text
    def compile_sampler(self, **kwargs):
        assert kwargs == {"batch": True}
        return Sampler()
"""
    (tmp_path / "symft.py").write_text(module, encoding="utf-8")
    old_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + old_path)
    row = run_external(
        build_distillation_case("bh", k=2),
        backend="symft",
        shots=3,
        seed=1,
        interpreter=sys.executable,
    )
    assert row["status"] == "ok"
    assert (
        row["simulator"]["program"]["aggregation"] == "python_joint_measurement_records"
    )
    assert row["counts"] == {
        "attempted": 3,
        "accepted": 2,
        "accepted_and_target_pass": 1,
    }
    assert row["quality"]["target_infidelity"]["value"] == 0.5


@pytest.mark.external
@pytest.mark.parametrize(
    ("backend", "run_variable", "python_variable"),
    [
        ("tsim", "RUN_TSIM_BENCHMARKS", "TSIM_PYTHON"),
        ("xtim", "RUN_XTIM_BENCHMARKS", "XTIM_PYTHON"),
        ("clifft", "RUN_CLIFFT_BENCHMARKS", "CLIFFT_PYTHON"),
        ("symft", "RUN_SYMFT_BENCHMARKS", "SYMFT_PYTHON"),
    ],
)
def test_real_external_15to1_smoke(backend, run_variable, python_variable):
    if not os.environ.get(run_variable):
        pytest.skip(f"set {run_variable}=1")
    interpreter = os.environ.get(python_variable, sys.executable)
    ideal = run_external(
        build_distillation_case("15to1"),
        backend=backend,
        shots=2,
        seed=7,
        interpreter=interpreter,
    )
    assert ideal["status"] == "ok", ideal.get("error")
    assert ideal["counts"] == {
        "attempted": 2,
        "accepted": 2,
        "accepted_and_target_pass": 2,
    }

    logical_z = run_external(
        build_distillation_case("15to1", deterministic_z_faults=(0, 1, 2)),
        backend=backend,
        shots=2,
        seed=7,
        interpreter=interpreter,
    )
    assert logical_z["status"] == "ok", logical_z.get("error")
    assert logical_z["counts"] == {
        "attempted": 2,
        "accepted": 2,
        "accepted_and_target_pass": 0,
    }
    assert logical_z["quality"]["target_infidelity"]["value"] == 1
