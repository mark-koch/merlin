import numpy as np
import pytest

import merlin
from merlin import CircuitSampler


def test_construction_counts_repr_and_all_result_families() -> None:
    source = """
        RX 0
        T 0
        EXP_VAL X0 Y0 Z0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-2]
    """
    sampler = CircuitSampler(source, seed=1)
    assert sampler.num_qubits == 2
    assert sampler.num_measurements == 2
    assert sampler.num_detectors == 2
    assert sampler.num_observables == 3
    assert sampler.num_exp_vals == 3
    assert sampler.num_fault_sites == 0
    assert repr(sampler) == (
        "CircuitSampler(num_qubits=2, num_measurements=2, num_detectors=2, "
        "num_observables=3, num_exp_vals=3, num_fault_sites=0)"
    )
    assert merlin.CircuitSampler is CircuitSampler
    assert not hasattr(merlin, "MeasurementSampler")
    assert not hasattr(merlin, "DetectorSampler")

    result = sampler.sample(2)
    assert merlin.SampleResult is type(result)
    assert repr(result) == "SampleResult(total_shots=2, accepted_shots=2)"
    assert result.total_shots == result.accepted_shots == 2
    assert result.discards == 0
    assert result.measurements.shape == (2, 2)
    assert result.detectors.shape == (2, 2)
    assert result.observables.shape == (2, 3)
    assert result.exp_vals.shape == (2, 3)
    assert result.measurements.dtype == np.bool_
    assert result.exp_vals.dtype == np.float64
    np.testing.assert_allclose(
        result.exp_vals,
        [[2**-0.5, 2**-0.5, 0.0]] * 2,
        atol=1e-12,
    )


def test_expectations_execute_in_place_and_do_not_collapse() -> None:
    sampler = CircuitSampler(
        "RX 0\nT 0\nEXP_VAL X0 Z0\nT_DAG 0\nEXP_VAL X0 !X0 Z0\nMX 0",
        seed=2,
    )
    result = sampler.sample(3)
    np.testing.assert_allclose(
        result.exp_vals,
        [[2**-0.5, 0.0, 1.0, -1.0, 0.0]] * 3,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.measurements, [[False]] * 3)


def test_ccz_and_depolarize3_execute_in_circuit_order() -> None:
    ccz = CircuitSampler("RX 0 1 2\nCCZ 0 1 2\nEXP_VAL X0 X1 X2\nM 0 1 2", seed=20)
    direct = merlin.Simulator(seed=20)
    direct.reset_x(0, 1, 2)
    direct.ccz(0, 1, 2)
    expected_measurements = direct.measure_many(0, 1, 2)
    result = ccz.sample(1)
    np.testing.assert_array_equal(result.measurements, [expected_measurements])
    np.testing.assert_allclose(result.exp_vals, [[0.5, 0.5, 0.5]])

    source = "RX 0 1 2\nDEPOLARIZE3(1) 0 1 2\nM 0 1 2"
    left = CircuitSampler(source, seed=21).sample(20)
    right = CircuitSampler(source, seed=21).sample(20)
    np.testing.assert_array_equal(left.measurements, right.measurements)

    repeated = CircuitSampler(
        "RX 0 1 2\nREPEAT 2 {\nCCZ 0 1 2\nDEPOLARIZE3(0) 0 1 2\n}\nMX 0 1 2",
        seed=22,
    )
    np.testing.assert_array_equal(repeated.sample(1).measurements, [[False] * 3])


def test_boolean_outputs_pack_independently() -> None:
    source = (
        "X_ERROR(1) 0 2 7 8\nM 0 1 2 3 4 5 6 7 8\n"
        "DETECTOR rec[-9]\nDETECTOR rec[-8]\n"
        "OBSERVABLE_INCLUDE(8) rec[-7]\nEXP_VAL Z0"
    )
    result = CircuitSampler(source, seed=3).sample(2, bit_packed=True)
    assert result.measurements.dtype == np.uint8
    assert result.detectors.dtype == np.uint8
    assert result.observables.dtype == np.uint8
    np.testing.assert_array_equal(result.measurements, [[0b10000101, 1]] * 2)
    np.testing.assert_array_equal(result.detectors, [[1]] * 2)
    np.testing.assert_array_equal(result.observables, [[0, 1]] * 2)
    np.testing.assert_array_equal(result.exp_vals, [[-1.0]] * 2)


def test_postselection_omits_entire_result_rows() -> None:
    source = "X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nEXP_VAL Z0"
    result = CircuitSampler(source, seed=4, postselect=[0]).sample(5)
    assert result.total_shots == 5
    assert result.accepted_shots == 0
    assert result.discards == 5
    assert result.measurements.shape == (0, 1)
    assert result.detectors.shape == (0, 1)
    assert result.observables.shape == (0, 0)
    assert result.exp_vals.shape == (0, 1)


def test_mid_circuit_measurement_reset_feedback_and_mpp() -> None:
    sampler = CircuitSampler(
        "X 0\nMR 0\nCX rec[-1] 1\nM 0 1\nRX 2\nZ 2\nMRX 2\nMX 2\nMPP Z0*Z1",
        seed=5,
    )
    np.testing.assert_array_equal(
        sampler.sample(1).measurements,
        [[True, False, True, True, False, True]],
    )


def test_seeded_successive_calls_and_empty_shapes() -> None:
    source = "RX 0\nX_ERROR(.25) 0\nEXP_VAL X0\nM 0"
    split = CircuitSampler(source, seed=6)
    together = CircuitSampler(source, seed=6)
    left = split.sample(7)
    right = split.sample(9)
    whole = together.sample(16)
    np.testing.assert_array_equal(
        np.concatenate([left.measurements, right.measurements]), whole.measurements
    )
    np.testing.assert_allclose(
        np.concatenate([left.exp_vals, right.exp_vals]), whole.exp_vals
    )

    empty = CircuitSampler("", seed=7).sample(3)
    assert empty.measurements.shape == (3, 0)
    assert empty.detectors.shape == (3, 0)
    assert empty.observables.shape == (3, 0)
    assert empty.exp_vals.shape == (3, 0)


def test_validation() -> None:
    for source in [
        "EXP_VAL",
        "EXP_VAL X0*Z0",
        "EXP_VAL X0**Z1",
        "EXP_VAL A0",
        "EXP_VAL X",
    ]:
        with pytest.raises(ValueError):
            CircuitSampler(source)
    for mask in [[-1], [1], [0, 0]]:
        with pytest.raises(ValueError, match="postselected detector"):
            CircuitSampler("M 0\nDETECTOR rec[-1]", postselect=mask)
    with pytest.raises(ValueError, match="shots must be non-negative"):
        CircuitSampler("").sample(-1)


def test_uniform_fixed_fault_sampling() -> None:
    sampler = CircuitSampler("X_ERROR(0) 0 1 2 3\nM 0 1 2 3", seed=30)
    assert sampler.num_fault_sites == 4

    result = sampler.sample(100, num_faults=2)
    assert result.total_shots == result.accepted_shots == 100
    np.testing.assert_array_equal(result.measurements.sum(axis=1), [2] * 100)

    np.testing.assert_array_equal(
        sampler.sample(1, num_faults=0).measurements,
        [[False, False, False, False]],
    )
    np.testing.assert_array_equal(
        sampler.sample(1, num_faults=4).measurements,
        [[True, True, True, True]],
    )


def test_fixed_fault_readout_sites_packing_and_postselection() -> None:
    readout = CircuitSampler("M 0\nM(0) 1\nMPP(0) Z2", seed=31)
    assert readout.num_fault_sites == 2
    result = readout.sample(1, num_faults=2, bit_packed=True)
    assert result.measurements.dtype == np.uint8
    np.testing.assert_array_equal(result.measurements, [[0b110]])

    postselected = CircuitSampler(
        "X_ERROR(0) 0 1\nM 0 1\nDETECTOR rec[-2]",
        seed=32,
        postselect=[0],
    ).sample(200, num_faults=1)
    assert postselected.total_shots == 200
    assert 0 < postselected.accepted_shots < 200
    assert not postselected.detectors.any()


def test_fixed_fault_validation_and_seeded_reproducibility() -> None:
    source = "DEPOLARIZE1(0) 0 1 2\nM 0 1 2"
    left = CircuitSampler(source, seed=33)
    right = CircuitSampler(source, seed=33)
    np.testing.assert_array_equal(
        left.sample(100, num_faults=2).measurements,
        right.sample(100, num_faults=2).measurements,
    )

    for shots, num_faults, message in [
        (-1, 0, "shots"),
        (1, -1, "num_faults"),
        (1, 4, "fault sites"),
    ]:
        with pytest.raises(ValueError, match=message):
            CircuitSampler(source, seed=34).sample(shots, num_faults=num_faults)

    for channel in [
        "PAULI_CHANNEL_1(1,0,0) 0",
        "PAULI_CHANNEL_2(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 1",
    ]:
        sampler = CircuitSampler(channel, seed=35)
        assert sampler.num_fault_sites == 1
        with pytest.raises(ValueError, match="does not support PAULI_CHANNEL"):
            sampler.sample(0, num_faults=0)


def test_reference_and_shot_incompatibility_context() -> None:
    lines = ["RX 0 1 2", "T 0", "T 0", "T 1", "T 2"]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        lines.extend((f"CX {a} {b}", f"T_DAG {b}", f"CX {a} {b}"))
    lines.extend(("CX 0 2", "CX 1 2", "T 2", "CX 1 2", "CX 0 2", "MX 0"))
    with pytest.raises(merlin.MeasurementError, match="reference measurement 0"):
        CircuitSampler("\n".join(lines), seed=8)
