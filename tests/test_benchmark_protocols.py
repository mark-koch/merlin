from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest

from benchmarks.protocols.code_switching import (
    BBTPair,
    build_code_switching_case,
    ccz_generator_tensor,
    ccz_logical_tensor,
    ccz_triples_for_pair,
    make_bbt_pair,
    switch_logical_map,
    switch_map_commutes,
)
from benchmarks.protocols.cultivation import build_cultivation_case
from benchmarks.protocols.css import (
    CSSCode,
    f2_rank,
    matrix_15to1,
    matrix_bravyi_haah_3k8,
    triorthogonal_css_code,
    validate_triorthogonal_generator,
)
from benchmarks.protocols.distillation import build_distillation_case
from benchmarks.protocols.gates import append_ccz, append_cs
from merlin import CircuitSampler


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


def _classical_phase(lines: list[str], bits: tuple[int, ...]):
    state = list(bits)
    phase = 0
    for line in lines:
        gate, *targets = line.split()
        targets = [int(target) for target in targets]
        if gate == "CX":
            state[targets[1]] ^= state[targets[0]]
        elif gate == "T":
            phase += state[targets[0]]
        elif gate == "T_DAG":
            phase -= state[targets[0]]
        elif gate == "S":
            phase += 2 * state[targets[0]]
        elif gate == "S_DAG":
            phase -= 2 * state[targets[0]]
    return tuple(state), phase % 8


def _apply_cnot_network(bits, operations):
    result = np.asarray(bits, dtype=np.int64).copy()
    for kind, control, target in operations:
        assert kind == "cnot"
        result[target] ^= result[control]
    return result


def test_triorthogonal_matrices_and_encoder_round_trips():
    cases = [
        (matrix_15to1(), 1),
        (matrix_bravyi_haah_3k8(2), 2),
        (matrix_bravyi_haah_3k8(4), 4),
    ]
    for generator, k in cases:
        validated, actual_k = validate_triorthogonal_generator(generator, k)
        assert actual_k == k
        assert np.array_equal(validated, generator)
        code = triorthogonal_css_code(generator, k_logical=k)
        assert (
            f2_rank(code.hx, n=code.n) + f2_rank(code.hz, n=code.n) + code.k == code.n
        )
        for qubit in range(code.n):
            bits = np.zeros(code.n, dtype=np.int64)
            bits[qubit] = 1
            encoded = _apply_cnot_network(bits, code.encoder_ops())
            decoded = _apply_cnot_network(encoded, code.decoder_ops())
            np.testing.assert_array_equal(decoded, bits)


def test_cs_and_ccz_decompositions_have_exact_basis_phases():
    cs_lines: list[str] = []
    append_cs(cs_lines, 0, 1)
    for value in range(4):
        bits = ((value >> 0) & 1, (value >> 1) & 1)
        output, phase = _classical_phase(cs_lines, bits)
        assert output == bits
        assert phase == (2 if all(bits) else 0)

    ccz_lines: list[str] = []
    append_ccz(ccz_lines, 0, 1, 2)
    assert sum(line.startswith("T ") for line in ccz_lines) == 4
    assert sum(line.startswith("T_DAG ") for line in ccz_lines) == 3
    assert sum(line.startswith("CX ") for line in ccz_lines) == 10
    for value in range(8):
        bits = tuple((value >> qubit) & 1 for qubit in range(3))
        output, phase = _classical_phase(ccz_lines, bits)
        assert output == bits
        assert phase == (4 if all(bits) else 0)


def test_distillation_noise_detectors_and_target_locations():
    clean = build_distillation_case("15to1", p_phys=0.01)
    assert clean.n_phys == 15
    assert clean.n_detectors == 14
    assert clean.n_observables == 1
    assert clean.circuit.count("Z_ERROR(") == 15
    assert "DEPOLARIZE" not in clean.circuit
    assert clean.circuit.count("DETECTOR ") == 14
    assert clean.circuit.rstrip().endswith("OBSERVABLE_INCLUDE(0) rec[-1]")

    noisy = build_distillation_case("bh", k=2, p_phys=0.01, noisy_clifford=True)
    assert noisy.n_phys == 14
    assert noisy.n_observables == 2
    assert "DEPOLARIZE1(0.01)" in noisy.circuit
    assert "DEPOLARIZE2(0.01)" in noisy.circuit
    assert "Z_ERROR(" not in noisy.circuit


@pytest.mark.parametrize("distance,qubits,detectors,sites", [(3, 15, 20, 518), (5, 42, 107, 3564)])
def test_cultivation_case_metadata_noise_and_target_scoring(distance, qubits, detectors, sites):
    case = build_cultivation_case(distance, p_phys=0.002)
    assert case.case_id == f"cultivation-d{distance}-t"
    assert case.family == "cultivation"
    assert (case.n_phys, case.n_detectors, case.n_observables) == (qubits, detectors, 1)
    assert case.metadata["distance"] == distance
    assert "X_ERROR(0.002)" in case.circuit
    assert "MX(0.002)" in case.circuit
    assert "DEPOLARIZE2(0.002)" in case.circuit
    sampler = CircuitSampler(case.circuit, seed=1)
    assert (sampler.num_qubits, sampler.num_detectors,
            sampler.num_observables, sampler.num_fault_sites) == (qubits, detectors, 1, sites)

    detector_only = build_cultivation_case(distance, target_scoring=False)
    assert detector_only.n_observables == 0
    assert "OBSERVABLE_INCLUDE" not in detector_only.circuit
    assert CircuitSampler(detector_only.circuit, seed=1).num_detectors == detectors
    assert CircuitSampler(detector_only.circuit, seed=1).num_observables == 0


def test_cultivation_rejects_invalid_distance_and_probability():
    with pytest.raises(ValueError, match="distance"):
        build_cultivation_case(7)
    with pytest.raises(ValueError, match="p_phys"):
        build_cultivation_case(3, p_phys=-0.1)


def test_bbt_css_logical_map_and_circuit_invariants():
    expected = {
        "bt27": (27, 18, 135, 8, 16),
        "bt81": (81, 54, 405, 26, 52),
    }
    for name, (source_n, target_n, total_n, rank_x, rank_z) in expected.items():
        pair = make_bbt_pair(name)
        assert (pair.source.n, pair.target.n) == (source_n, target_n)
        assert f2_rank(pair.source.hx, n=source_n) == rank_x
        assert f2_rank(pair.source.hz, n=source_n) == rank_z
        assert f2_rank(pair.target.hx, n=target_n) == rank_x
        assert f2_rank(pair.target.hz, n=target_n) == rank_x
        assert switch_map_commutes(pair)
        np.testing.assert_array_equal(
            switch_logical_map(pair),
            np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.int64),
        )
        case = build_code_switching_case(pair, p_phys=0.01, target_scoring=False)
        assert case.n_phys == total_n
        assert case.n_detectors == 3 * (source_n - 3)
        assert case.n_observables == 0
        switch_index = case.circuit.index(f"CX 0 {3 * source_n}")
        noise_index = case.circuit.index("DEPOLARIZE1(")
        assert switch_index < noise_index
        assert noise_index < case.circuit.index("T ", noise_index)
        assert "CX rec[-" in case.circuit
        assert "CCZ" not in case.circuit


def test_bbt_case_reports_physical_non_clifford_size():
    expected = {
        "bt27": (378, 4_129, 126),
        "bt81": (1_134, 24_397, 396),
    }
    for name, (
        non_clifford_rotations,
        instruction_count,
        measurement_count,
    ) in expected.items():
        case = build_code_switching_case(name, target_scoring=False)
        metrics = case.description()["circuit_metrics"]
        assert metrics["non_clifford_rotation_count"] == non_clifford_rotations
        assert metrics["instruction_count"] == instruction_count
        assert metrics["measurement_count"] == measurement_count


def test_bbt_cup_tensor_has_no_mixed_stabilizer_overlaps():
    expected_logical = np.zeros((3, 3, 3), dtype=np.int64)
    for indices in permutations(range(3)):
        expected_logical[indices] = 1

    for name in ("bt27", "bt81"):
        pair = make_bbt_pair(name)
        assert len(ccz_triples_for_pair(pair)) == 6 * pair.cells
        tensor = ccz_generator_tensor(pair)
        expected = np.zeros_like(tensor)
        expected[: pair.source.k, : pair.source.k, : pair.source.k] = expected_logical
        np.testing.assert_array_equal(tensor, expected)
        np.testing.assert_array_equal(ccz_logical_tensor(pair), expected_logical)


def test_inverse_target_scoring_for_ideal_orthogonal_and_nonorthogonal_errors():
    from merlin import CircuitSampler

    ideal = CircuitSampler("RX 0\nT 0\nT_DAG 0\nMX 0", seed=1).sample(100).measurements
    orthogonal = (
        CircuitSampler("RX 0\nT 0\nZ_ERROR(1) 0\nT_DAG 0\nMX 0", seed=1)
        .sample(100)
        .measurements
    )
    coherent = (
        CircuitSampler("RX 0\nT 0\nT 0\nT_DAG 0\nMX 0", seed=2)
        .sample(20_000)
        .measurements
    )
    assert not ideal.any()
    assert orthogonal.all()
    assert abs(float(coherent.mean()) - math.sin(math.pi / 8) ** 2) < 0.015


def test_tiny_code_switching_contains_feedback_and_nine_source_observables():
    case = build_code_switching_case(tiny_bbt_pair(), target_scoring=True)
    assert case.n_phys == 15
    assert case.n_detectors == 0
    assert case.n_observables == 9
    assert case.circuit.count("CX rec[-") == 6
    assert case.circuit.count("OBSERVABLE_INCLUDE(") == 9
