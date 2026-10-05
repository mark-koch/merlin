# Circuit instructions

`CircuitSampler` accepts an extended subset of the
[Stim circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md),
including the non-Clifford `T` and `T_DAG` gates.

## Gates

| Instruction | Targets | Effect |
| --- | --- | --- |
| `I` | Qubits | Identity. Its targets still contribute to the circuit's qubit count. |
| `X` | Qubits | Pauli X. |
| `Y` | Qubits | Pauli Y. |
| `Z` | Qubits | Pauli Z. |
| `S` | Qubits | Phase gate, `diag(1, i)`. |
| `S_DAG` | Qubits | Inverse phase gate, `diag(1, -i)`. |
| `T` | Qubits | T gate, `diag(1, exp(i pi/4))`. |
| `T_DAG` | Qubits | Inverse T gate, `diag(1, exp(-i pi/4))`. |
| `CX`, `CNOT` | Control-target pairs | Controlled X. `CX 0 1 2 3` applies `CX 0 1` and then `CX 2 3`. |
| `CZ` | Qubit pairs | Controlled Z. |
| `SWAP` | Qubit pairs | Swap each pair. |
| `CCZ` | Qubit triples | Controlled-controlled Z. `CCZ 0 1 2 3 4 5` applies two gates in order. |


## Resets and measurements

| Instruction | Syntax | Effect |
| --- | --- | --- |
| `R`, `RZ` | `R q...` | Reset each target to `|0⟩`; produces no record. |
| `RX` | `RX q...` | Reset each target to `|+⟩`; produces no record. |
| `M`, `MZ` | `M[(p)] [!]q...` | Measure Z and append one bit per target. |
| `MX` | `MX[(p)] [!]q...` | Measure X and append one bit per target. |
| `MR`, `MRZ` | `MR[(p)] [!]q...` | Measure Z, record the result, then reset to `|0⟩`. |
| `MRX` | `MRX[(p)] [!]q...` | Measure X, record the result, then reset to `|+⟩`. |

The optional probability `p` independently flips each reported bit after the
physical measurement. Prefixing a target with `!` also flips its reported bit.
Neither readout noise nor inversion changes the post-measurement quantum state
or the state prepared by a measurement-reset instruction.

Targets are measured sequentially in written order:

```stim
RX 0 1
M(0.01) !0 1
MRX 0
```

Z measurements are always representable. Some X measurements are incompatible
with the DCP state representation and may raise `MeasurementError` during sampling.

## Pauli products

### `MPP`

`MPP` measures whitespace-separated Pauli products sequentially and appends one
measurement bit per product:

```stim
MPP X0*Y1 !Z2*X3
MPP(0.001) Z0*Z1 X0*X1
```

Factors within a product are joined with `*` and use `Xq`, `Yq`, or `Zq`.
Each qubit may occur at most once within a product. Prefixing any factor with
`!` negates the product; multiple inversions combine by parity. The optional
probability independently flips each reported product result. Products
containing X or Y factors can raise `MeasurementError` when their projections
are incompatible with the DCP representation.

### `EXP_VAL`

`EXP_VAL` evaluates one or more Pauli-product expectation values without
collapsing or otherwise changing the state:

```stim
RX 0
T 0
EXP_VAL X0 Y0 Z0 !X0
```

Products use the same factor and inversion syntax as `MPP`, but `EXP_VAL` does
not accept a probability argument and does not append to the measurement
record. Values are written, in instruction and product order, to
`SampleResult.exp_vals`. Negating a product negates its expectation value.

## Noise

Noise is sampled independently for every listed target or pair and is skipped
during construction of the noiseless detector and observable reference.

| Instruction | Syntax | Distribution |
| --- | --- | --- |
| `X_ERROR` | `X_ERROR(p) q...` | Apply X with probability `p`. |
| `Y_ERROR` | `Y_ERROR(p) q...` | Apply Y with probability `p`. |
| `Z_ERROR` | `Z_ERROR(p) q...` | Apply Z with probability `p`. |
| `DEPOLARIZE1` | `DEPOLARIZE1(p) q...` | With probability `p`, choose X, Y, or Z uniformly. |
| `DEPOLARIZE2` | `DEPOLARIZE2(p) a b...` | With probability `p`, choose uniformly from the 15 non-identity two-qubit Pauli products. |
| `DEPOLARIZE3` | `DEPOLARIZE3(p) a b c...` | With probability `p`, choose uniformly from the 63 non-identity three-qubit Pauli products. |
| `PAULI_CHANNEL_1` | `PAULI_CHANNEL_1(px, py, pz) q...` | Apply X, Y, or Z with the corresponding probability. |
| `PAULI_CHANNEL_2` | `PAULI_CHANNEL_2(p1, ..., p15) a b...` | Apply the corresponding two-qubit Pauli product to each pair. |

For `PAULI_CHANNEL_1`, the argument order is `X`, `Y`, `Z`. For
`PAULI_CHANNEL_2`, it is:

```text
IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
```

In either channel, the remaining probability applies identity. Two- and
three-qubit depolarizing noise use consecutive distinct pairs or triples.

```stim
X_ERROR(0.001) 0 1
DEPOLARIZE2(0.01) 0 1 2 3
DEPOLARIZE3(0.01) 0 1 2
PAULI_CHANNEL_1(0.01, 0.02, 0.03) 4
```

## Measurement feedback

A recorded measurement can classically control a Pauli on a qubit:

```stim
M(0.01) !0
CX rec[-1] 1
CY rec[-1] 2
CZ rec[-1] 3
```

`CX` and `CNOT` conditionally apply X, `CY` conditionally applies Y, and `CZ`
conditionally applies Z. The control is the final recorded bit, including
target inversion and readout noise. The record must already exist, the
`rec[-k]` target must come first in its pair, and the Pauli target must be a
qubit. Feedback itself consumes no randomness and produces no record.

Quantum and record-controlled pairs can be mixed and still execute in written
order:

```stim
M 0
CX rec[-1] 1 2 3
```

This conditionally applies X to qubit 1 and then applies CNOT from qubit 2 to
qubit 3. Ordinary qubit-controlled `CY` is not supported.

## Detectors and logical observables

| Instruction | Effect |
| --- | --- |
| `DETECTOR[(coordinates)] rec[-k]...` | Append the XOR of the referenced measurement bits as one detector. Numeric coordinates are accepted but ignored. |
| `OBSERVABLE_INCLUDE(index) rec[-k]...` | XOR the referenced bits into logical observable `index`. Multiple fragments with the same index accumulate by XOR. |

Record references must point to earlier measurements. Observable indices are
non-negative integers; the reported observable width is one plus the largest
index. Pauli targets in `OBSERVABLE_INCLUDE` are not supported.

`CircuitSampler` eagerly runs a noiseless reference shot. Returned detector and
logical-observable bits are the sampled parities XORed with the corresponding
reference parities. A postselected detector rejects an attempt when this
normalized detector bit is `True`.

```stim
X_ERROR(0.01) 0
M 0
DETECTOR(10, 20) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
```

## Repeats and ignored annotations

`REPEAT` blocks can be nested and are expanded in written order:

```stim
REPEAT 2 {
    T 0
    REPEAT 3 {
        CX 0 1
    }
}
```

The body of `REPEAT 0` is still validated, but it contributes no operations,
qubits, or result columns.

`TICK`, `QUBIT_COORDS`, and `SHIFT_COORDS` are accepted and ignored. They do
not change the state, result shapes, or circuit qubit count.

## Limitations

Instructions not listed on this page are rejected with `ValueError`. Notable
unsupported Stim features include Hadamard gates, Y-basis measurement and
reset, correlated-error instructions, sweep-bit controls, product
observables in `OBSERVABLE_INCLUDE`, and measurement-record editing. General
quantum controlled-Y is also unsupported; `CY` is accepted only with a
`rec[-k]` control.

Even when an instruction parses successfully, an X-basis or mixed-Pauli
projection can be incompatible with the DCP representation. This raises
`MeasurementError` either while constructing the sampler's noiseless reference
or while sampling the affected shot.
