# Merlin

`merlin` is a simulator for non-Clifford quantum error correction circuits. It supports circuits in the CNOT+T gateset and runs in polynomial time.

## Quick Start

Install `merlin` by running

```console
pip install merlin-sim
```

The `Simulator` class can be used to interactively apply gates and measurements:

```python
from merlin import Simulator

sim = Simulator(seed=1234)
sim.reset_x(0, 1)
sim.t(0)
sim.t_dag(1)
sim.cnot(0, 1)

result = sim.measure(0)
p_true = sim.peek_z_probability(1)
parity = sim.measure_observable("-XX", flip_probability=0.001)
```

The `CircuitSampler` class can be used to sample from circuits in an extended stim-like format:

```python
from merlin import CircuitSampler

sampler = CircuitSampler("""
    RX 0
    T 0
    EXP_VAL X0 Z0
    MX 0
    R 0
    X_ERROR(0.01) 0
    M 0
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
""", seed=1234, postselect=[0])

result = sampler.sample(1000)
print(result.measurements)
print(result.exp_vals)
print(result.detectors)
print(result.observables)
```


## Rust Interface

The simulation core is also a standalone Rust library:

```rust
use merlin::{
    Circuit, CircuitSampler, Simulator, StateVectorEndian,
};

let mut sim = Simulator::with_seed(1234);
sim.reset_x(0);
sim.t(0);
assert_eq!(sim.state_vector(StateVectorEndian::Little).len(), 2);

let circuit = Circuit::parse("RX 0\nT 0\nMX 0");
let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 1234)?;
let samples = sampler.sample(100)?;
assert_eq!(samples.measurements.len(), 100 * circuit.num_measurements());

let detector_circuit = Circuit::parse("M 0\nDETECTOR rec[-1]");
let mut detector_sampler = CircuitSampler::from_circuit_with_seed(&detector_circuit, 1234)?;
let detector_samples = detector_sampler.sample(100)?;
assert_eq!(detector_samples.accepted_shots, 100);
# Ok::<(), merlin::MeasurementError>(())
```
