#  Overview

`merlin` is a simulator for non-Clifford quantum error correction that runs in polynomial time.


## Installation

Install merlin by running

```console
pip install merlin-sim
```


## Quick start

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

Note that `merlin` only supports certain kinds of X measurements such as flag or gauge measurements.
For unsupported measurements, the simulator will raise a `MeasurementError`:

```python
# Measuring CCZ|+++> in X is not supported
sim = Simulator()
sim.reset_x(0, 1, 2)
sim.ccz(0, 1, 2)
sim.measure_x(0)
```

Output:

```
MeasurementError: X measurement on qubit 0 is incompatible: non-affine zero set
```

The `CircuitSampler` class can be used to sample from circuits in an extended stim-like format:

```python
from merlin import CircuitSampler

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
