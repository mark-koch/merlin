from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

class MeasurementError(Exception): ...

class SampleResult:
    """Results returned by `CircuitSampler.sample`.

    Each array has one row per accepted shot. With unpacked sampling the three
    bit arrays have shape ``(accepted_shots, count)`` and NumPy `bool` dtype.
    With bit-packed sampling they have shape
    ``(accepted_shots, ceil(count / 8))``, `uint8` dtype, and independently
    padded little-endian rows. Expectation values are never packed and have
    shape ``(accepted_shots, num_exp_vals)``.
    """

    @property
    def measurements(self) -> NDArray[np.bool_] | NDArray[np.uint8]:
        """Raw measurement records in circuit order."""
        ...

    @property
    def detectors(self) -> NDArray[np.bool_] | NDArray[np.uint8]:
        """Detector flips relative to the noiseless reference sample."""
        ...

    @property
    def observables(self) -> NDArray[np.bool_] | NDArray[np.uint8]:
        """Logical-observable flips relative to the noiseless reference."""
        ...

    @property
    def exp_vals(self) -> NDArray[np.float64]:
        """Expectation values from `EXP_VAL`, with NumPy `float64` dtype."""
        ...

    @property
    def total_shots(self) -> int:
        """Number of attempted shots requested from the sampler."""
        ...

    @property
    def accepted_shots(self) -> int:
        """Number of rows that survived detector postselection."""
        ...

    @property
    def discards(self) -> int:
        """Number of attempted shots rejected by postselection."""
        ...

class CircuitSampler:
    """Parse-once sampler backed by `Simulator`.

    A private noiseless replay is performed during construction to define the
    detector and observable reference parities. Every attempted shot otherwise
    starts from a fresh all-zero state and executes the circuit in source order.
    """

    def __init__(
        self,
        circuit: str,
        *,
        seed: int | None = None,
        postselect: Sequence[int] | None = None,
    ) -> None:
        """Create a sampler and eagerly compute its noiseless reference.

        The circuit is parsed once. Each attempted shot then starts from a fresh
        all-zero `Simulator` and executes the circuit in source order.

        Args:
            circuit: Circuit source in the supported extended Stim format.
            seed: Optional unsigned 64-bit seed for reproducible sampling.
            postselect: Detector indices whose normalized `True` value rejects
                a shot. Indices must be unique and in range.

        Raises:
            ValueError: If the circuit or postselection mask is invalid.
            MeasurementError: If noiseless reference generation encounters an
                incompatible measurement.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Number of logical qubits referenced by the circuit."""
        ...

    @property
    def num_measurements(self) -> int:
        """Number of measurement-record columns in a complete shot."""
        ...

    @property
    def num_detectors(self) -> int:
        """Number of detector columns."""
        ...

    @property
    def num_observables(self) -> int:
        """Number of logical-observable columns."""
        ...

    @property
    def num_exp_vals(self) -> int:
        """Number of expectation-value columns produced by `EXP_VAL`."""
        ...

    @property
    def num_fault_sites(self) -> int:
        """Number of explicit noise and readout-fault sites.

        Batched single-qubit noise targets count separately, while two- and
        three-qubit noise groups count once. A measurement target is counted
        only when its readout probability was written explicitly, including
        an explicit probability of zero.
        """
        ...

    def sample(
        self,
        shots: int,
        *,
        bit_packed: bool = False,
        num_faults: int | None = None
    ) -> SampleResult:
        """Sample the circuit.

        Args:
            shots: Number of shots to attempt. Rejected postselected attempts
                are not replaced.
            bit_packed: Pack each boolean output family independently into
                little-endian `uint8` rows. `exp_vals` remains `float64`.
            num_faults: Sample uniformly from executions where exactly `num_faults`
                happen. Assumes that all fault sites have uniform proability
                and does not support `PAULI_CHANNEL_1` or `PAULI_CHANNEL_2`
                instructions.

        Returns:
            Measurement records, detector flips, observable flips, expectation
            values, and attempted/accepted shot counts. Postselected attempts
            are stopped at the rejecting detector and omitted from all arrays.
            Successive calls continue the sampler's RNG stream.

        Raises:
            ValueError: If `shots` or `num_faults` is negative.
            MeasurementError: If a shot encounters an incompatible measurement.
                Randomness consumed before the failure remains consumed.

        Examples:
            ```python
            from merlin import CircuitSampler

            sampler = CircuitSampler(
                '''
                RX 0
                T 0
                EXP_VAL X0 Z0
                MX 0
                DETECTOR rec[-1]
                OBSERVABLE_INCLUDE(0) rec[-1]
                ''',
                seed=1234,
                postselect=[0],
            )

            assert sampler.num_qubits == 1
            assert sampler.num_measurements == 1
            assert sampler.num_detectors == 1
            assert sampler.num_observables == 1
            assert sampler.num_exp_vals == 2

            result = sampler.sample(100)
            assert result.total_shots == 100
            assert result.discards == result.total_shots - result.accepted_shots
            assert result.measurements.shape == (result.accepted_shots, 1)
            assert result.detectors.shape == (result.accepted_shots, 1)
            assert result.observables.shape == (result.accepted_shots, 1)
            assert result.exp_vals.shape == (result.accepted_shots, 2)

            packed = sampler.sample(100, bit_packed=True)
            assert packed.measurements.dtype.name == "uint8"
            ```
        """
        ...

class Simulator:
    """Interactive simulator.

    Missing qubits are automatically appended in the all-zero state before a
    mutating operation.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Create an empty interactive simulator.

        Missing qubits are automatically appended in the all-zero state before
        a mutating operation.

        Args:
            seed: Optional unsigned 64-bit seed for measurements and noise.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            ```
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Number of currently tracked qubits.

        Examples:
            Mutating a previously untracked target grows the simulator.

            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(2)
            assert sim.num_qubits == 3
            ```
        """
        ...

    def x(self, *targets: int) -> None:
        """Apply Pauli X sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def y(self, *targets: int) -> None:
        """Apply Pauli Y sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def z(self, *targets: int) -> None:
        """Apply Pauli Z sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def s(self, *targets: int) -> None:
        """Apply the phase gate S sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def s_dag(self, *targets: int) -> None:
        """Apply S-dagger sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def t(self, *targets: int) -> None:
        """Apply a T gate sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def t_dag(self, *targets: int) -> None:
        """Apply a T-dagger gate sequentially to each target.

        Raises:
            IndexError: If a target is negative.
        """
        ...

    def cnot(self, *targets: int) -> None:
        """Apply CNOTs to consecutive control-target pairs.

        All targets are validated before the simulator grows or mutates.

        Raises:
            ValueError: If there is an odd target count or a pair repeats a
                qubit.
            IndexError: If a target is negative.
        """
        ...

    def cx(self, *targets: int) -> None:
        """Exact alias of `cnot`."""
        ...

    def cz(self, *targets: int) -> None:
        """Apply controlled-Z gates to consecutive target pairs.

        Raises:
            ValueError: If there is an odd target count or a pair repeats a
                qubit.
            IndexError: If a target is negative.
        """
        ...

    def swap(self, *targets: int) -> None:
        """Swap consecutive target pairs.

        Raises:
            ValueError: If there is an odd target count or a pair repeats a
                qubit.
            IndexError: If a target is negative.
        """
        ...

    def ccz(self, *targets: int) -> None:
        """Apply CCZ gates to consecutive target triples.

        Raises:
            ValueError: If the target count is not divisible by three or a
                triple repeats a qubit.
            IndexError: If a target is negative.
        """
        ...

    def measure(self, target: int) -> bool:
        """Measure one qubit in the Z basis and collapse the state.

        Returns:
            `False` for the +1 eigenvalue and `True` for the -1 eigenvalue.

        Raises:
            IndexError: If `target` is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.reset_x(0)
            bit = sim.measure(0)
            ```
        """
        ...

    def measure_z(self, target: int) -> bool:
        """Exact alias of `measure`.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            assert sim.measure_z(0) is False
            ```
        """
        ...

    def measure_many(self, *targets: int) -> list[bool]:
        """Measure targets sequentially in the Z basis.

        Raises:
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            bits = sim.measure_many(0, 1, 2)
            assert bits == [False, False, False]
            ```
        """
        ...

    def measure_x(self, target: int) -> bool:
        """Measure one qubit in the X basis and collapse the state.

        Raises:
            IndexError: If `target` is negative.
            MeasurementError: If the projected state is not supported by the
                DCP representation. Failure does not mutate state or RNG.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(0)
            assert sim.measure_x(0) is False
            ```
        """
        ...

    def measure_observable(
        self, observable: str, *, flip_probability: float = 0.0
    ) -> bool:
        """Measure a signed Hermitian Pauli product.

        The compact observable syntax has an optional leading sign followed by
        positional `X`, `Y`, `Z`, `I`, or `_` factors, for example `-X_YZ`.
        Readout flipping occurs after a successful physical projection.

        Args:
            observable: Compact positional Pauli string.
            flip_probability: Probability of flipping the reported bit.

        Raises:
            ValueError: If the observable or probability is invalid.
            MeasurementError: If the projection is incompatible. Failure does
                not mutate state or RNG.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.reset_x(0, 1)
            bit = sim.measure_observable("-XX", flip_probability=0.01)
            ```
        """
        ...

    def peek_observable_expectation(self, observable: str) -> float:
        """Return a Pauli expectation without mutating state, size, or RNG.

        Untracked factors are evaluated against implicit all-zero qubits. The
        result is a general floating-point value in `[-1, 1]`.

        Raises:
            ValueError: If `observable` is malformed.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(0)
            assert sim.peek_observable_expectation("X") == 1.0
            ```
        """
        ...

    def peek_probability(self, target: int) -> float:
        """Return the probability that a Z measurement reports `True`.

        This does not mutate state, grow the simulator, or consume randomness.
        An untracked target has probability zero.

        Raises:
            IndexError: If `target` is negative.

        Examples:
            `peek_z_probability` is an exact alias.

            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(0)
            p_true = sim.peek_probability(0)
            assert sim.peek_z_probability(0) == p_true == 0.5
            ```
        """
        ...

    def peek_z_probability(self, target: int) -> float:
        """Exact alias of `peek_probability`.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            assert sim.peek_z_probability(0) == sim.peek_probability(0)
            ```
        """
        ...

    def peek_x_probability(self, target: int) -> float:
        """Return the probability that an X measurement reports `True`.

        Raises:
            IndexError: If `target` is negative.
            MeasurementError: If the corresponding X measurement is
                incompatible with the DCP representation.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(0)
            assert sim.peek_x_probability(0) == 0.0
            ```
        """
        ...

    def x_error(self, *targets: int, p: float) -> None:
        """Independently apply X to each target with probability `p`.

        Raises:
            ValueError: If `p` is not finite or is outside `[0, 1]`.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.x_error(0, p=0.01)
            ```
        """
        ...

    def y_error(self, *targets: int, p: float) -> None:
        """Independently apply Y to each target with probability `p`.

        Raises:
            ValueError: If `p` is not finite or is outside `[0, 1]`.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.y_error(0, p=0.01)
            ```
        """
        ...

    def z_error(self, *targets: int, p: float) -> None:
        """Independently apply Z to each target with probability `p`.

        Raises:
            ValueError: If `p` is not finite or is outside `[0, 1]`.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.z_error(0, p=0.01)
            ```
        """
        ...

    def depolarize1(self, *targets: int, p: float) -> None:
        """Apply uniform single-qubit depolarizing noise to each target.

        Raises:
            ValueError: If `p` is not finite or is outside `[0, 1]`.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.depolarize1(0, 1, 2, p=0.02)
            ```
        """
        ...

    def depolarize2(self, *targets: int, p: float) -> None:
        """Apply uniform two-qubit depolarizing noise to consecutive pairs.

        The non-identity Pauli product is chosen uniformly from 15 choices.

        Raises:
            ValueError: If `p` is invalid, the target count is odd, or a pair
                uses the same qubit twice.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.depolarize2(0, 1, 2, 3, p=0.03)
            ```
        """
        ...

    def depolarize3(self, *targets: int, p: float) -> None:
        """Apply uniform three-qubit depolarizing noise to target triples.

        The non-identity Pauli product is chosen uniformly from 63 choices.

        Raises:
            ValueError: If `p` is invalid, the target count is not divisible
                by three, or a triple repeats a qubit.
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.depolarize3(0, 1, 2, 3, 4, 5, p=0.03)
            ```
        """
        ...

    def state_vector(
        self, *, endian: Literal["little", "big"] = "little"
    ) -> NDArray[np.complex64]:
        """Return a normalized, canonical-phase state vector.

        Args:
            endian: Whether qubit 0 is the least- or most-significant index bit.

        Returns:
            A one-dimensional NumPy `complex64` array.

        This query does not mutate state or consume randomness.

        Raises:
            ValueError: If `endian` is not `"little"` or `"big"`.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator()
            sim.reset_x(0)
            little_endian = sim.state_vector()
            big_endian = sim.state_vector(endian="big")
            ```
        """
        ...

    def reset(self, *targets: int) -> None:
        """Reset targets sequentially to the +1 Z eigenstate `|0>`.

        Reset measurements can consume randomness, but their results are not
        reported.

        Raises:
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.reset(0)
            ```
        """
        ...

    def reset_z(self, *targets: int) -> None:
        """Exact alias of `reset`.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.reset_z(0)
            ```
        """
        ...

    def reset_x(self, *targets: int) -> None:
        """Reset targets sequentially to the +1 X eigenstate `|+>`.

        Raises:
            IndexError: If a target is negative.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            sim.reset_x(0)
            ```
        """
        ...

    def copy(
        self, *, copy_rng: bool = False, seed: int | None = None
    ) -> Self:
        """Copy the quantum state with configurable RNG behavior.

        By default the copy receives a fresh entropy-seeded RNG. Set
        `copy_rng=True` to reproduce the exact RNG state, or pass `seed` to use
        a chosen new stream.

        Raises:
            ValueError: If both `copy_rng=True` and `seed` are supplied.

        Examples:
            ```python
            from merlin import Simulator

            sim = Simulator(seed=1234)
            quantum_copy = sim.copy()
            exact_copy = sim.copy(copy_rng=True)
            seeded_copy = sim.copy(seed=5678)
            ```
        """
        ...
