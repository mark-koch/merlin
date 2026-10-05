from __future__ import annotations

import numpy as np
import pytest

from benchmarks.models import CircuitCase
from benchmarks.protocols.code_switching import build_code_switching_case, make_bbt_pair
from benchmarks.protocols.distillation import build_distillation_case
from merlin import CircuitSampler


def _expectation_probe_circuit(case: CircuitCase) -> str:
    """Replace final checks/projections by non-destructive Pauli probes."""
    lines = case.circuit.splitlines()
    remaining = case.n_detectors + case.n_observables
    for index in range(len(lines) - 1, -1, -1):
        fields = lines[index].split()
        if not fields:
            continue
        gate = fields[0].split("(", 1)[0]
        if remaining and gate in {"M", "MX"}:
            targets = fields[1:]
            if len(targets) > remaining:
                raise AssertionError("final measurement boundary splits an instruction")
            pauli = "Z" if gate == "M" else "X"
            lines[index] = "EXP_VAL " + " ".join(pauli + target for target in targets)
            remaining -= len(targets)
    if remaining:
        raise AssertionError(f"could not locate {remaining} final measurements")
    return (
        "\n".join(
            line
            for line in lines
            if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE"))
        )
        + "\n"
    )


def _sample_raw_final_bits(case: CircuitCase, *, shots: int, seed: int) -> np.ndarray:
    result = CircuitSampler(case.circuit, seed=seed).sample(shots)
    assert result.accepted_shots == shots
    width = case.n_detectors + case.n_observables
    assert width > 0
    return result.measurements[:, -width:]


def _assert_ideal_final_state(case: CircuitCase, *, shots: int, seed: int) -> None:
    raw = _sample_raw_final_bits(case, shots=shots, seed=seed)
    np.testing.assert_array_equal(raw, np.zeros_like(raw))

    probes = CircuitSampler(_expectation_probe_circuit(case), seed=seed).sample(shots)
    assert probes.exp_vals.shape == (shots, case.n_detectors + case.n_observables)
    np.testing.assert_allclose(probes.exp_vals, 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("protocol", "k"),
    [("15to1", 1)] + [("bh", k) for k in range(2, 18, 2)],
)
def test_distillation_circuits_have_ideal_raw_checks_and_expectations(protocol, k):
    _assert_ideal_final_state(
        build_distillation_case(protocol, k=k),
        shots=3,
        seed=100 + k,
    )


@pytest.mark.slow
@pytest.mark.parametrize(("pair", "shots"), [("bt27", 5), ("bt81", 1)])
def test_bbt_circuits_have_ideal_raw_checks_and_expectations(pair, shots):
    _assert_ideal_final_state(
        build_code_switching_case(pair),
        shots=shots,
        seed=2718,
    )


@pytest.mark.parametrize(
    ("protocol", "k"),
    [("15to1", 1)] + [("bh", k) for k in range(2, 18, 2)],
)
def test_every_single_distillation_z_fault_triggers_a_raw_check(protocol, k):
    ideal = build_distillation_case(protocol, k=k, target_scoring=False)
    for qubit in range(ideal.n_phys):
        faulty = build_distillation_case(
            protocol,
            k=k,
            target_scoring=False,
            deterministic_z_faults=(qubit,),
        )
        syndrome = _sample_raw_final_bits(faulty, shots=1, seed=qubit)[0]
        assert syndrome.any(), f"{ideal.case_id} failed to detect Z on qubit {qubit}"


def test_bravyi_haah_fixed_logical_fault_is_accepted_and_fails_one_target():
    case = build_distillation_case("bh", k=2, deterministic_z_faults=(0, 1, 8))
    final = _sample_raw_final_bits(case, shots=2, seed=8128)
    syndrome = final[:, : case.n_detectors]
    targets = final[:, case.n_detectors :]
    assert not syndrome.any()
    np.testing.assert_array_equal(targets, [[True, False], [True, False]])


@pytest.mark.slow
def test_bt27_representative_single_faults_trigger_only_their_block_syndrome():
    pair = make_bbt_pair("bt27")
    checks_per_block = pair.source.n - pair.source.k
    for block, qubit in enumerate((0, pair.source.n, 2 * pair.source.n)):
        case = build_code_switching_case(
            pair,
            target_scoring=False,
            deterministic_z_faults=(qubit,),
        )
        syndrome = _sample_raw_final_bits(case, shots=1, seed=block)[0]
        active_blocks = {
            index // checks_per_block for index in np.flatnonzero(syndrome)
        }
        assert active_blocks == {block}


@pytest.mark.slow
def test_bt27_fixed_logical_fault_is_accepted_and_fails_one_target():
    # This is the first source-block logical Z representative in the fixed
    # logical basis chosen by make_bbt_pair("bt27").
    case = build_code_switching_case("bt27", deterministic_z_faults=(2, 4, 6))
    final = _sample_raw_final_bits(case, shots=2, seed=31415)
    syndrome = final[:, : case.n_detectors]
    targets = final[:, case.n_detectors :]
    assert not syndrome.any()
    expected = np.zeros_like(targets)
    expected[:, 0] = True
    np.testing.assert_array_equal(targets, expected)
