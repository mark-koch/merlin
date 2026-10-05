# Python API

::: merlin.Simulator
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      members:
        - __init__
        - num_qubits
        - x
        - y
        - z
        - s
        - s_dag
        - t
        - t_dag
        - cnot
        - cx
        - cz
        - swap
        - ccz
        - measure
        - measure_z
        - measure_many
        - measure_x
        - measure_observable
        - peek_observable_expectation
        - peek_probability
        - peek_z_probability
        - peek_x_probability
        - x_error
        - y_error
        - z_error
        - depolarize1
        - depolarize2
        - depolarize3
        - state_vector
        - reset
        - reset_z
        - reset_x
        - copy

::: merlin.CircuitSampler
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      members:
        - __init__
        - num_qubits
        - num_measurements
        - num_detectors
        - num_observables
        - num_exp_vals
        - num_fault_sites
        - sample

See [Circuit instructions](circuit-format.md) for the complete accepted circuit
language.

::: merlin.SampleResult
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      members:
        - measurements
        - detectors
        - observables
        - exp_vals
        - total_shots
        - accepted_shots
        - discards
