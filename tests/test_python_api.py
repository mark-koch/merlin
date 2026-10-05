import merlin
import numpy as np
import pytest

from merlin import Simulator, MeasurementError


def test_simulator_api_mutates_in_place() -> None:
    sim = Simulator(seed=1)
    assert sim.num_qubits == 0
    assert repr(sim) == "Simulator(num_qubits=0)"
    assert sim.t(2) is None
    assert sim.num_qubits == 3
    assert sim.t_dag(1) is None
    assert sim.cnot(0, 1) is None
    assert sim.cx(1, 2) is None


def test_constructor_is_keyword_only_and_legacy_api_is_removed() -> None:
    with pytest.raises(TypeError):
        Simulator(1)  # type: ignore[misc]
    assert not hasattr(merlin, "DCPState")
    assert not hasattr(merlin, "DcpFrameSimulator")
    assert not hasattr(merlin, "DcpGroupSimulator")
    assert set(merlin.__all__) == {
        "CircuitSampler",
        "Simulator",
        "MeasurementError",
        "SampleResult",
    }
    assert not hasattr(Simulator, "initial")
    for name in ("apply_t", "apply_tdg", "apply_cnot"):
        assert not hasattr(Simulator, name)


def test_variadic_gates_and_cnot_pairs() -> None:
    sim = Simulator(seed=2)
    assert sim.t() is None
    assert sim.t_dag() is None
    assert sim.cnot() is None
    assert sim.cx() is None
    assert sim.num_qubits == 0

    sim.reset_x(0, 2)
    sim.cnot(0, 1, 2, 3)
    assert sim.measure(0) == sim.measure(1)
    assert sim.measure(2) == sim.measure(3)


def test_complete_interactive_gate_set_matches_dense_states() -> None:
    x = Simulator()
    x.x(0)
    np.testing.assert_array_equal(x.state_vector(), [0, 1])

    y = Simulator()
    y.y(0)
    np.testing.assert_array_equal(y.state_vector(), [0, 1])

    z = Simulator()
    z.reset_x(0)
    z.z(0)
    np.testing.assert_allclose(z.state_vector(), np.array([1, -1]) / np.sqrt(2))

    phase = Simulator()
    phase.reset_x(0)
    phase.s(0)
    np.testing.assert_allclose(
        phase.state_vector(), np.array([1, 1j]) / np.sqrt(2), atol=1e-7
    )
    phase.s_dag(0)
    np.testing.assert_allclose(
        phase.state_vector(), np.array([1, 1]) / np.sqrt(2), atol=1e-7
    )

    controlled = Simulator()
    controlled.reset_x(0, 1, 2)
    controlled.cz(0, 1)
    expected_cz = np.ones(8) / np.sqrt(8)
    expected_cz[[3, 7]] *= -1
    np.testing.assert_allclose(controlled.state_vector(), expected_cz, atol=1e-7)

    swapped = Simulator()
    swapped.x(2)
    swapped.swap(0, 2)
    expected_swap = np.zeros(8)
    expected_swap[1] = 1
    np.testing.assert_array_equal(swapped.state_vector(), expected_swap)

    ccz = Simulator()
    ccz.reset_x(0, 1, 2)
    ccz.t(0)
    ccz.s(1)
    ccz.x(2)
    before = ccz.state_vector()
    ccz.ccz(0, 1, 2)
    expected_ccz = before.copy()
    expected_ccz[7] *= -1
    np.testing.assert_allclose(ccz.state_vector(), expected_ccz, atol=1e-7)
    ccz.ccz(0, 1, 2)
    np.testing.assert_allclose(ccz.state_vector(), before, atol=1e-7)


def test_random_gate_sequences_cancel_with_their_adjoint() -> None:
    rng = np.random.default_rng(0xDC0)
    inverse = {
        "x": "x",
        "y": "y",
        "z": "z",
        "s": "s_dag",
        "s_dag": "s",
        "t": "t_dag",
        "t_dag": "t",
        "cnot": "cnot",
        "cz": "cz",
        "swap": "swap",
        "ccz": "ccz",
    }
    gate_names = tuple(inverse)

    def random_operation(name: str, n: int) -> tuple[str, tuple[int, ...]]:
        arity = 3 if name == "ccz" else (2 if name in {"cnot", "cz", "swap"} else 1)
        targets = tuple(int(q) for q in rng.choice(n, size=arity, replace=False))
        return name, targets

    for _ in range(80):
        n = int(rng.integers(3, 7))
        sim = Simulator()
        sim.reset_x(*range(n))
        sim.t(0)
        sim.s(1)
        sim.cnot(0, 1)
        sim.ccz(0, 1, 2)
        before = sim.state_vector()

        sequence = [random_operation(name, n) for name in gate_names]
        sequence.extend(
            random_operation(str(rng.choice(gate_names)), n) for _ in range(40)
        )
        rng.shuffle(sequence)

        for name, targets in sequence:
            getattr(sim, name)(*targets)
        for name, targets in reversed(sequence):
            getattr(sim, inverse[name])(*targets)

        after = sim.state_vector()
        pivot = int(np.argmax(np.abs(before)))
        phase = after[pivot] / before[pivot]
        np.testing.assert_allclose(after, phase * before, atol=2e-6, rtol=2e-6)


def test_gate_batch_validation_is_transactional() -> None:
    sim = Simulator(seed=202)
    for method in (sim.cz, sim.swap):
        with pytest.raises(ValueError, match="even number"):
            method(0)
        with pytest.raises(ValueError, match="different qubits"):
            method(3, 3)
    with pytest.raises(ValueError, match="multiple of three"):
        sim.ccz(0, 1)
    with pytest.raises(ValueError, match="different"):
        sim.ccz(0, 1, 1)
    with pytest.raises(IndexError):
        sim.ccz(0, 1, -1)
    assert sim.num_qubits == 0

    assert sim.x() is sim.y() is sim.z() is None
    assert sim.s() is sim.s_dag() is None
    assert sim.cz() is sim.swap() is sim.ccz() is None
    assert sim.num_qubits == 0

    sim.ccz(0, 1, 2, 2, 3, 4)
    assert sim.num_qubits == 5

    wide = Simulator()
    wide.ccz(64, 127, 191)
    wide.ccz(64, 127, 191)
    assert wide.num_qubits == 192


def test_invalid_variadic_targets_are_rejected_before_growth() -> None:
    sim = Simulator()
    with pytest.raises(IndexError):
        sim.t(-1)
    assert sim.num_qubits == 0
    with pytest.raises(IndexError):
        sim.cnot(0, -1)
    assert sim.num_qubits == 0
    with pytest.raises(ValueError, match="even number"):
        sim.cnot(0)
    assert sim.num_qubits == 0
    with pytest.raises(ValueError, match="different qubits"):
        sim.cnot(4, 4)
    assert sim.num_qubits == 0


def test_measurement_bits_and_measure_many() -> None:
    sim = Simulator(seed=3)
    result = sim.measure(2)
    assert result is False
    assert sim.num_qubits == 3

    sim.reset_x(0, 1)
    outcomes = sim.measure_many(0, 1)
    assert all(type(outcome) is bool for outcome in outcomes)
    assert sim.measure_many(0, 1) == outcomes
    assert sim.measure_many() == []


def test_x_measurement_uses_bool_convention() -> None:
    sim = Simulator(seed=4)
    sim.reset_x(0)
    assert sim.measure_x(0) is False

    sim.t(0, 0, 0, 0)
    assert sim.measure_x(0) is True


def test_probability_queries_are_non_mutating() -> None:
    sim = Simulator(seed=5)
    assert sim.peek_probability(100) == 0.0
    assert sim.peek_x_probability(100) == 0.5
    assert sim.num_qubits == 0

    left = Simulator(seed=1234)
    right = Simulator(seed=1234)
    left.t(7)
    right.t(7)
    assert left.peek_probability(0) == 0.0
    assert left.peek_x_probability(0) == 0.5
    left.reset_x(*range(8))
    right.reset_x(*range(8))
    assert left.measure_many(*range(8)) == right.measure_many(*range(8))


def test_z_measurement_aliases() -> None:
    left = Simulator(seed=51)
    left.reset_x(0)
    right = left.copy(copy_rng=True)
    assert left.peek_z_probability(0) == left.peek_probability(0) == 0.5
    assert left.measure_z(0) == right.measure(0)


@pytest.mark.parametrize("method", ["x_error", "y_error", "z_error"])
def test_pauli_errors_are_variadic_seeded_and_grow(method: str) -> None:
    left = Simulator(seed=52)
    right = Simulator(seed=52)
    getattr(left, method)(0, 3, p=0.4)
    getattr(right, method)(0, 3, p=0.4)
    assert left.num_qubits == right.num_qubits == 4
    np.testing.assert_allclose(left.state_vector(), right.state_vector())

    no_error = Simulator(seed=53)
    getattr(no_error, method)(64, p=0)
    assert no_error.num_qubits == 65


def test_deterministic_pauli_errors_have_expected_actions() -> None:
    x = Simulator()
    x.x_error(0, p=1)
    np.testing.assert_array_equal(
        x.state_vector(), np.array([0, 1], dtype=np.complex64)
    )

    z = Simulator()
    z.reset_x(0)
    z.z_error(0, p=1)
    np.testing.assert_allclose(
        z.state_vector(), np.array([1, -1], dtype=np.complex64) / np.sqrt(2)
    )

    y = Simulator()
    y.y_error(0, p=1)
    np.testing.assert_array_equal(
        y.state_vector(), np.array([0, 1], dtype=np.complex64)
    )


def test_depolarizing_noise_is_seeded_and_paired() -> None:
    left = Simulator(seed=54)
    right = Simulator(seed=54)
    left.reset_x(0, 1, 2)
    right.reset_x(0, 1, 2)
    left.depolarize1(0, 1, 2, p=1)
    right.depolarize1(0, 1, 2, p=1)
    left.depolarize2(0, 1, 1, 2, p=1)
    right.depolarize2(0, 1, 1, 2, p=1)
    left.depolarize3(0, 1, 2, 2, 1, 0, p=1)
    right.depolarize3(0, 1, 2, 2, 1, 0, p=1)
    np.testing.assert_allclose(left.state_vector(), right.state_vector())

    growth = Simulator(seed=54)
    growth.depolarize3(64, 127, 191, p=0)
    assert growth.num_qubits == 192

    skipped = Simulator(seed=55)
    direct = Simulator(seed=55)
    skipped.reset_x(0, 1, 2)
    direct.reset_x(0, 1, 2)
    skipped.depolarize3(0, 1, 2, p=0)
    skipped.depolarize3(0, 1, 2, p=1)
    direct.depolarize3(0, 1, 2, p=1)
    np.testing.assert_allclose(skipped.state_vector(), direct.state_vector())


@pytest.mark.parametrize(
    "method",
    ["x_error", "y_error", "z_error", "depolarize1", "depolarize2", "depolarize3"],
)
@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_noise_rejects_invalid_probabilities_without_growth(
    method: str, probability: float
) -> None:
    sim = Simulator()
    targets = (
        (2, 3)
        if method == "depolarize2"
        else ((2, 3, 4) if method == "depolarize3" else (2,))
    )
    with pytest.raises(ValueError, match="probability"):
        getattr(sim, method)(*targets, p=probability)
    assert sim.num_qubits == 0


def test_depolarize2_validates_all_pairs_before_mutation() -> None:
    sim = Simulator(seed=55)
    with pytest.raises(ValueError, match="even number"):
        sim.depolarize2(0, p=0.5)
    with pytest.raises(ValueError, match="different qubits"):
        sim.depolarize2(4, 4, p=0.5)
    with pytest.raises(IndexError):
        sim.depolarize2(0, -1, p=0.5)
    assert sim.num_qubits == 0


def test_depolarize3_validates_all_triples_before_mutation() -> None:
    sim = Simulator(seed=56)
    control = sim.copy(copy_rng=True)
    with pytest.raises(ValueError, match="multiple of three"):
        sim.depolarize3(0, 1, p=0.5)
    with pytest.raises(ValueError, match="different"):
        sim.depolarize3(4, 4, 5, p=0.5)
    with pytest.raises(IndexError):
        sim.depolarize3(0, 1, -1, p=0.5)
    assert sim.num_qubits == 0
    sim.depolarize3(0, 1, 2, p=1)
    control.depolarize3(0, 1, 2, p=1)
    np.testing.assert_allclose(sim.state_vector(), control.state_vector())


def test_noise_probability_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        Simulator().x_error(0, 0.5)  # type: ignore[misc]


def test_state_vector_matches_stim_endianness_and_is_read_only() -> None:
    sim = Simulator(seed=56)
    sim.reset_x(0)
    sim.cnot(0, 2)
    sim.t(0)
    control = sim.copy(copy_rng=True)

    little = sim.state_vector()
    big = sim.state_vector(endian="big")
    assert little.dtype == np.complex64
    assert little.shape == big.shape == (8,)
    np.testing.assert_allclose(np.linalg.norm(little), 1)
    np.testing.assert_allclose(little, big.reshape(2, 2, 2).transpose().reshape(-1))
    first = little[np.flatnonzero(little)[0]]
    assert first.imag == 0 and first.real > 0
    assert sim.num_qubits == control.num_qubits
    assert sim.measure_many(0, 1, 2) == control.measure_many(0, 1, 2)

    empty = Simulator().state_vector()
    np.testing.assert_array_equal(empty, np.array([1], dtype=np.complex64))
    with pytest.raises(ValueError, match="endian"):
        sim.state_vector(endian="middle")  # type: ignore[arg-type]


def test_non_dyadic_x_probability() -> None:
    sim = Simulator(seed=6)
    sim.reset_x(0)
    sim.t(0)
    expected_true = (1.0 - 1.0 / 2.0**0.5) / 2.0
    assert sim.peek_x_probability(0) == pytest.approx(expected_true)


def test_reset_and_reset_z_prepare_zero() -> None:
    sim = Simulator(seed=7)
    sim.reset_x(0, 1)
    sim.reset(0)
    sim.reset_z(1)
    assert sim.peek_probability(0) == 0.0
    assert sim.peek_probability(1) == 0.0
    assert sim.measure_many(0, 1) == [False, False]


def test_reset_x_prepares_plus_and_grows() -> None:
    sim = Simulator(seed=8)
    sim.reset_x(64)
    assert sim.num_qubits == 65
    assert sim.peek_x_probability(64) == 0.0
    assert sim.measure_x(64) is False


def test_measure_observable_matches_stim_bit_conventions() -> None:
    sim = Simulator(seed=18)
    sim.reset_x(0)
    sim.cnot(0, 1)
    assert sim.measure_observable("XX") is False
    assert sim.measure_observable("YY") is True
    assert sim.measure_observable("-ZZ") is True

    flipped = Simulator(seed=19)
    flipped.reset_x(0)
    assert flipped.measure_observable("X", flip_probability=1) is True
    assert flipped.measure_observable("X") is False


def test_measure_observable_validation_precedes_mutation() -> None:
    sim = Simulator(seed=20)
    for observable in ["", "+", "-", "XZq"]:
        with pytest.raises(ValueError):
            sim.measure_observable(observable)
    with pytest.raises(ValueError, match="probability"):
        sim.measure_observable("Z_____", flip_probability=2)
    assert sim.num_qubits == 0

    assert sim.measure_observable("Z_I") is False
    assert sim.num_qubits == 3


def test_peek_observable_expectation_matches_stim_and_non_clifford_values() -> None:
    bell = Simulator(seed=41)
    bell.reset_x(0)
    bell.cnot(0, 1)
    expected = {
        "XX": 1.0,
        "YY": -1.0,
        "ZZ": 1.0,
        "-ZZ": -1.0,
        "ZI": 0.0,
        "II": 1.0,
        "IIZ": 1.0,
    }
    for observable, value in expected.items():
        assert bell.peek_observable_expectation(observable) == pytest.approx(value)

    non_clifford = Simulator(seed=42)
    non_clifford.reset_x(0)
    non_clifford.t(0)
    assert non_clifford.peek_observable_expectation("X") == pytest.approx(
        1 / np.sqrt(2)
    )
    assert non_clifford.peek_observable_expectation("Y") == pytest.approx(
        1 / np.sqrt(2)
    )
    assert non_clifford.peek_observable_expectation("Z") == 0.0


def test_peek_observable_expectation_is_non_mutating_and_validates() -> None:
    sim = Simulator(seed=43)
    control = sim.copy(copy_rng=True)
    assert sim.peek_observable_expectation("-I") == -1.0
    assert sim.peek_observable_expectation("Z_I") == 1.0
    assert sim.peek_observable_expectation("X_I") == 0.0
    assert sim.num_qubits == 0

    for observable in ["", "+", "-", "Xi"]:
        with pytest.raises(ValueError):
            sim.peek_observable_expectation(observable)
    assert sim.num_qubits == 0
    assert sim.measure_many(0, 1, 2) == control.measure_many(0, 1, 2)


def test_peek_observable_expectation_handles_incompatible_projection() -> None:
    sim = _incompatible_simulator(seed=44)
    with pytest.raises(merlin.MeasurementError):
        sim.peek_x_probability(0)
    vector = sim.state_vector()
    expected = np.vdot(vector, vector[np.arange(vector.size) ^ 1]).real
    assert sim.peek_observable_expectation("X__") == pytest.approx(expected, abs=2e-5)


def _apply_parity_phase(sim: Simulator, qubits: list[int], power: int) -> None:
    target = qubits[-1]
    for control in qubits[:-1]:
        sim.cnot(control, target)
    for _ in range(power):
        sim.t(target)
    for control in reversed(qubits[:-1]):
        sim.cnot(control, target)


def _apply_ccz(sim: Simulator, a: int, b: int, c: int) -> None:
    for qubit in (a, b, c):
        _apply_parity_phase(sim, [qubit], 1)
    for pair in ([a, b], [a, c], [b, c]):
        _apply_parity_phase(sim, pair, 7)
    _apply_parity_phase(sim, [a, b, c], 1)


def _incompatible_simulator(seed: int) -> Simulator:
    sim = Simulator(seed=seed)
    sim.reset_x(0, 1, 2)
    sim.t(0)
    _apply_ccz(sim, 0, 1, 2)
    return sim


def test_incompatible_x_operations_still_raise_without_mutation() -> None:
    sim = _incompatible_simulator(9)
    control = sim.copy(copy_rng=True)
    with pytest.raises(MeasurementError, match="incompatible"):
        sim.peek_x_probability(0)
    with pytest.raises(MeasurementError, match="incompatible"):
        sim.measure_x(0)
    assert sim.measure_many(1, 2) == control.measure_many(1, 2)


def test_reset_x_handles_an_initially_incompatible_target() -> None:
    sim = _incompatible_simulator(10)
    sim.reset_x(0)
    assert sim.peek_x_probability(0) == 0.0
    assert sim.measure_x(0) is False


def test_copy_can_duplicate_rng_state() -> None:
    original = Simulator(seed=11)
    original.reset_x(*range(8))
    copied = original.copy(copy_rng=True)
    assert original is not copied
    assert original.measure_many(*range(8)) == copied.measure_many(*range(8))


def test_copy_can_reseed_rng() -> None:
    original = Simulator(seed=12)
    original.reset_x(*range(8))
    left = original.copy(seed=99)
    right = original.copy(seed=99)
    assert left.measure_many(*range(8)) == right.measure_many(*range(8))

    fresh_rng = original.copy()
    assert fresh_rng.num_qubits == original.num_qubits
    assert fresh_rng.peek_x_probability(0) == original.peek_x_probability(0)


def test_copy_rejects_conflicting_rng_options() -> None:
    sim = Simulator()
    with pytest.raises(ValueError, match="cannot both"):
        sim.copy(copy_rng=True, seed=1)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_invalid_seed_uses_integer_conversion_error(seed: int) -> None:
    with pytest.raises(OverflowError):
        Simulator(seed=seed)
    with pytest.raises(OverflowError):
        Simulator().copy(seed=seed)


@pytest.mark.parametrize(
    "method",
    [
        "measure",
        "measure_z",
        "measure_x",
        "peek_probability",
        "peek_z_probability",
        "peek_x_probability",
    ],
)
def test_invalid_single_measurement_targets(method: str) -> None:
    sim = Simulator()
    with pytest.raises(IndexError):
        getattr(sim, method)(-1)
    assert sim.num_qubits == 0


def test_old_probability_query_names_are_removed() -> None:
    for name in (
        "measure_probability",
        "measure_z_probability",
        "measure_x_probability",
    ):
        assert not hasattr(Simulator, name)
