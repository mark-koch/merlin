"""Regression checks for the cultivation experiment workflow."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin import CircuitSampler
from experiments.cultivation.common import (
    inspect_circuit, load_checkpoint, save_checkpoint, validate_checkpoint,
)
from experiments.cultivation.sample import (
    clifft_program, collect, compile_clifft, pending_batches, sample_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def arguments(output, **overrides):
    values = dict(circuit="s", backend="merlin", shots_per_num_faults=7, max_num_faults=3,
                  batch_size=3, workers=1, seed=42, output=output)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("circuit", ["t", "s"])
def test_noiseless_circuits(circuit):
    source, metadata = inspect_circuit(circuit)
    assert (metadata["num_fault_sites"], metadata["num_detectors"],
            metadata["num_observables"], metadata["num_qubits"]) == (518, 20, 1, 15)
    sampler = CircuitSampler(source, seed=42, postselect=list(range(20)))
    result = sampler.sample(100, num_faults=0)
    assert result.accepted_shots == result.total_shots == 100
    assert not result.observables.any()


def test_uniform_noise_validation(monkeypatch, tmp_path):
    from experiments.cultivation import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    path = tmp_path / "circuit_d3_s.stim"
    path.write_text("R 0\nX_ERROR(0.01) 0\nM(0.02) 0\n")
    with pytest.raises(ValueError, match="uniform"):
        inspect_circuit("s")
    path.write_text("R 0\nPAULI_CHANNEL_1(0.1,0.1,0.1) 0\nM 0\n")
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_circuit("s")


@pytest.mark.parametrize("backend", ["merlin", "clifft"])
def test_accepted_error_counting(backend):
    if backend == "clifft":
        pytest.importorskip("clifft")
    # One of the two possible single faults is rejected; the other is a logical error.
    source = "R 0 1\nX_ERROR(0.01) 0 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    metadata = {"circuit_sha256": "test", "num_detectors": 1, "num_observables": 1,
                "num_fault_sites": 2, "site_probability": 0.01}
    result = sample_batch(source, metadata, 42, 1, 0, 100, backend=backend)
    assert result["total"] == 100
    assert 0 < result["accepted"] < 100
    assert result["errors"] == result["accepted"]


@pytest.mark.parametrize("backend", ["merlin", "clifft"])
def test_remainders_resume_and_extension(tmp_path, backend):
    if backend == "clifft":
        pytest.importorskip("clifft")
    path = tmp_path / "samples.json"
    original = collect(arguments(path, backend=backend))
    assert len(original["batches"]) == 9
    assert {batch["num_faults"] for batch in original["batches"]} == {1, 2, 3}
    for batches in validate_checkpoint(original).values():
        assert [batch["total"] for batch in batches] == [3, 3, 1]
    assert collect(arguments(path, backend=backend)) == original
    # Simulate interruption with gaps and batches completed out of order.
    incomplete = copy.deepcopy(original)
    incomplete["batches"] = incomplete["batches"][1::2]
    save_checkpoint(path, incomplete)
    assert collect(arguments(path, backend=backend)) == original
    extended = collect(arguments(path, backend=backend, shots_per_num_faults=11))
    for batches in validate_checkpoint(extended).values():
        assert sum(batch["total"] for batch in batches) == 11
    assert all(batch in extended["batches"] for batch in original["batches"])
    assert not list(pending_batches(extended))


@pytest.mark.parametrize("change", [dict(seed=43), dict(batch_size=4),
                                  dict(max_num_faults=2), dict(circuit="t"),
                                  dict(shots_per_num_faults=6)])
def test_incompatible_resume(tmp_path, change):
    path = tmp_path / "samples.json"
    collect(arguments(path))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        collect(arguments(path, **change))
    assert path.read_bytes() == before


def test_backend_mismatch_and_duplicate_batches(tmp_path):
    path = tmp_path / "samples.json"
    data = collect(arguments(path))
    data["backend"]["version"] = "different"
    save_checkpoint(path, data)
    with pytest.raises(ValueError, match="backend"):
        collect(arguments(path))
    data["batches"].append(data["batches"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_checkpoint(data)


def test_worker_failure_preserves_progress(tmp_path, monkeypatch):
    from experiments.cultivation import sample

    path = tmp_path / "samples.json"
    real_sample = sample.sample_batch

    def failing_sample(*args, **kwargs):
        if args[-2] == 3:
            raise RuntimeError("simulated failure")
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(sample, "sample_batch", failing_sample)
    with pytest.raises(RuntimeError, match="simulated failure"):
        collect(arguments(path))
    assert len(load_checkpoint(path)["batches"]) == 1
    monkeypatch.setattr(sample, "sample_batch", real_sample)
    assert len(collect(arguments(path))["batches"]) == 9


@pytest.mark.parametrize("circuit", ["t", "s"])
@pytest.mark.parametrize("backend", ["merlin", "clifft"])
def test_multicore_cli_reproducibility(tmp_path, circuit, backend):
    if backend == "clifft":
        pytest.importorskip("clifft")
    serial = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json"
    collect(arguments(serial, circuit=circuit, backend=backend))
    command = [sys.executable, str(ROOT / "experiments/cultivation/sample.py"),
               "--circuit", circuit, "--backend", backend,
               "--workers", "2", "--shots-per-num-faults", "7",
               "--max-num-faults", "3", "--batch-size", "3", "--output", str(parallel)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert json.loads(serial.read_text()) == json.loads(parallel.read_text())


def synthetic_checkpoint():
    return {
        "schema_version": 1,
        "metadata": {"circuit": "t", "num_fault_sites": 2, "circuit_sha256": "fixture",
                     "site_probability": 0.01, "num_detectors": 1,
                     "num_observables": 1, "num_qubits": 2},
        "backend": {"name": "merlin", "version": "test"},
        "config": {"max_num_faults": 1, "batch_size": 100, "shots_per_num_faults": 100},
        "batches": [{"num_faults": 1, "offset": 0, "total": 100,
                     "accepted": 50, "errors": 10}],
    }


def test_reweighting():
    pytest.importorskip("scipy")
    from experiments.cultivation.plot import reweight

    estimate = reweight(synthetic_checkpoint(), 0.1)
    assert estimate["survival"] == pytest.approx(0.81 + 0.18 * 0.5)
    assert estimate["logical_error_rate"] == pytest.approx(0.18 * 0.1 / 0.9)
    assert estimate["attempts_per_kept_shot"] == pytest.approx(1 / 0.9)
    assert estimate["omitted_probability"] == pytest.approx(0.01)
    assert estimate["ci95_half_width"] == pytest.approx(1.96 * math.sqrt(0.18**2 * 0.1 * 0.9 / 100) / 0.9)
    assert estimate["significant_tail"]
    data = synthetic_checkpoint()
    data["batches"][0]["errors"] = 0
    estimate = reweight(data, 0.1)
    assert estimate["logical_error_rate"] == 0
    assert math.isnan(estimate["ci95_half_width"])
    estimate = reweight(data, 1)
    assert math.isnan(estimate["logical_error_rate"])
    assert math.isnan(estimate["attempts_per_kept_shot"])
    data["batches"] = []
    with pytest.raises(ValueError, match="every num_faults"):
        reweight(data, 0.1)


def test_headless_plot(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from experiments.cultivation.plot import make_plot

    paths = []
    for circuit in ("t", "s"):
        data = synthetic_checkpoint()
        data["metadata"]["circuit"] = circuit
        if circuit == "s":
            data["batches"][0]["errors"] = 0
        path = tmp_path / f"{circuit}.json"
        save_checkpoint(path, data)
        paths.append(path)
    make_plot(*paths, [0, 0.1, 1], tmp_path / "comparison")
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "comparison.pdf").read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob("*.csv"))


@pytest.mark.parametrize("circuit", ["t", "s"])
def test_clifft_noiseless_metadata_and_cache(circuit):
    pytest.importorskip("clifft")
    source, metadata = inspect_circuit(circuit)
    compile_clifft.cache_clear()
    result = sample_batch(source, metadata, 42, 0, 0, 100, backend="clifft")
    assert result["total"] == result["accepted"] == 100
    assert result["errors"] == 0
    sample_batch(source, metadata, 42, 1, 0, 10, backend="clifft")
    assert compile_clifft.cache_info().misses == 1
    assert compile_clifft.cache_info().hits == 1
    metadata["num_fault_sites"] += 1
    with pytest.raises(ValueError, match="metadata"):
        clifft_program(source, metadata)


def test_clifft_api_contract(monkeypatch):
    clifft = pytest.importorskip("clifft")
    from experiments.cultivation import sample

    source, metadata = inspect_circuit("t")
    program = clifft_program(source, metadata)
    calls = []

    def fake_sample(received_program, **kwargs):
        assert received_program is program
        calls.append(kwargs)
        return SimpleNamespace(total_shots=17, passed_shots=5, logical_errors=2)

    monkeypatch.setattr(clifft, "sample_k_survivors", fake_sample)
    result = sample.sample_batch(source, metadata, 42, 3, 100, 17, backend="clifft")
    assert result == {"num_faults": 3, "offset": 100, "total": 17, "accepted": 5, "errors": 2}
    assert calls[0]["shots"] == 17
    assert calls[0]["k"] == 3
    assert calls[0]["threads"] == 1
    assert calls[0]["keep_records"] is False
    assert 0 <= calls[0]["seed"] < 2**64


@pytest.mark.parametrize("backend", ["merlin", "clifft"])
def test_backend_defaults_and_resume_rejection(tmp_path, monkeypatch, backend):
    pytest.importorskip("clifft")
    from experiments.cultivation import sample

    monkeypatch.setattr(sample, "ROOT", tmp_path)
    data = collect(arguments(None, backend=backend))
    filename = "samples_s.json" if backend == "merlin" else "samples_clifft_s.json"
    path = tmp_path / filename
    assert load_checkpoint(path) == data
    before = path.read_bytes()
    other = "clifft" if backend == "merlin" else "merlin"
    with pytest.raises(ValueError, match="backend"):
        collect(arguments(path, backend=other))
    assert path.read_bytes() == before
    data["backend"]["version"] = "different"
    save_checkpoint(path, data)
    with pytest.raises(ValueError, match="backend"):
        collect(arguments(path, backend=backend))


def comparison_paths(tmp_path):
    paths = []
    for backend in ("merlin", "clifft"):
        for circuit in ("t", "s"):
            data = synthetic_checkpoint()
            data["metadata"]["circuit"] = circuit
            data["backend"]["name"] = backend
            if backend == "clifft":
                data["config"]["max_num_faults"] = 2
                data["config"]["shots_per_num_faults"] = 200
                data["batches"].append(dict(data["batches"][0], num_faults=2))
            if circuit == "s":
                for batch in data["batches"]:
                    batch["errors"] = 0
            path = tmp_path / f"{backend}_{circuit}.json"
            save_checkpoint(path, data)
            paths.append(path)
    return paths


def test_four_curve_plot(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from experiments.cultivation.plot import make_plot

    paths = comparison_paths(tmp_path)
    curves = {}
    original_plot = Axes.plot

    def recording_plot(self, *args, **kwargs):
        if "label" in kwargs:
            curves[kwargs["label"]] = kwargs
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", recording_plot)
    make_plot(*paths[:2], [0, 0.1, 1], tmp_path / "four",
              clifft_t_path=paths[2], clifft_s_path=paths[3])
    assert set(curves) == {f"{circuit} circuit ({backend})"
                           for circuit in ("T", "S") for backend in ("merlin", "clifft")}
    assert curves["T circuit (clifft)"]["linestyle"] == "--"
    assert curves["T circuit (clifft)"]["markerfacecolor"] == "none"
    assert curves["T circuit (merlin)"]["linestyle"] == "-"
    assert (tmp_path / "four.png").exists()
    assert (tmp_path / "four.pdf").exists()
    diagnostics = capsys.readouterr().err
    for backend in ("clifft", "merlin"):
        assert f"T ({backend}) p=0.1: attempts per kept shot=" in diagnostics
    assert "logical error rate=" in diagnostics
    assert "approximate 95% half-width=" in diagnostics
    assert "zero survival" in diagnostics


@pytest.mark.parametrize("field,value", [
    ("circuit", "s"), ("backend", "merlin"), ("circuit_sha256", "different"),
    ("site_probability", 0.02), ("num_fault_sites", 3),
])
def test_comparison_rejects_mismatches(tmp_path, field, value):
    pytest.importorskip("scipy")
    from experiments.cultivation.plot import comparison_datasets

    paths = comparison_paths(tmp_path)
    data = load_checkpoint(paths[2])
    if field == "backend":
        data["backend"]["name"] = value
    else:
        data["metadata"][field] = value
    save_checkpoint(paths[2], data)
    with pytest.raises(ValueError):
        comparison_datasets(*paths)
