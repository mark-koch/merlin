use std::panic::catch_unwind;

use merlin::{Circuit, CircuitSampler, Simulator, StateVectorEndian};

#[test]
fn interactive_api_grows_and_clones_rng() {
    let mut left = Simulator::with_seed(5);
    left.reset_x(0);
    left.t(0);
    left.cnot(0, 1);
    assert_eq!(left.num_qubits(), 2);
    assert_eq!(left.state_vector(StateVectorEndian::Little).len(), 4);

    let mut right = left.clone();
    assert_eq!(left.measure_z(0), right.measure_z(0));
    assert_eq!(left.peek_x_probability(1), right.peek_x_probability(1));
}

#[test]
fn reset_noise_and_probability_api() {
    let mut sim = Simulator::with_seed(9);
    sim.reset(2);
    assert_eq!(sim.peek_probability(2), 0.0);
    sim.x_error(2, 1.0);
    assert!(sim.measure(2));
    sim.reset_x(2);
    assert_eq!(sim.peek_x_probability(2).unwrap(), 0.0);
    sim.y_error(0, 0.0);
    sim.z_error(0, 0.0);
    sim.depolarize1(0, 0.0);
    sim.depolarize2(0, 1, 0.0);
    sim.depolarize3(0, 1, 2, 0.0);
}

#[test]
fn complete_interactive_gate_api_is_available() {
    let mut sim = Simulator::with_seed(91);
    sim.x(0);
    sim.y(1);
    sim.z(2);
    sim.s(0);
    sim.s_dag(0);
    sim.cz(0, 1);
    sim.swap(1, 2);
    sim.ccz(0, 1, 2);
    assert_eq!(sim.num_qubits(), 3);

    let mut plus = Simulator::new();
    plus.reset_x(0);
    plus.reset_x(1);
    plus.reset_x(2);
    let before = plus.state_vector(StateVectorEndian::Little);
    plus.ccz(0, 1, 2);
    let after = plus.state_vector(StateVectorEndian::Little);
    assert_eq!(after[7], -before[7]);
    plus.ccz(0, 1, 2);
    assert_eq!(plus.state_vector(StateVectorEndian::Little), before);
}

#[test]
fn malformed_rust_inputs_panic() {
    assert!(catch_unwind(|| Circuit::parse("H 0")).is_err());
    assert!(
        catch_unwind(|| {
            let mut sim = Simulator::new();
            sim.cnot(0, 0);
        })
        .is_err()
    );
    assert!(
        catch_unwind(|| {
            let mut sim = Simulator::new();
            sim.x_error(0, f64::NAN);
        })
        .is_err()
    );
    assert!(catch_unwind(|| Simulator::new().peek_observable_expectation("")).is_err());
    assert!(
        catch_unwind(|| {
            let mut sim = Simulator::new();
            sim.ccz(0, 1, 1);
        })
        .is_err()
    );
    assert!(
        catch_unwind(|| {
            let _ = CircuitSampler::with_options("M 0\nDETECTOR rec[-1]", Some(1), &[1]);
        })
        .is_err()
    );
}

#[test]
fn direct_circuit_sampler_is_available() {
    let circuit = Circuit::parse("X 0\nM 0\nR 0\nM 0");
    let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 17).unwrap();
    assert_eq!(sampler.num_qubits(), 1);
    assert_eq!(sampler.num_measurements(), 2);
    assert_eq!(
        sampler.sample(2).unwrap().measurements,
        vec![true, false, true, false]
    );

    let mut packed = CircuitSampler::with_seed("X 0\nM 0", 17).unwrap();
    assert_eq!(
        packed.sample_bit_packed(2).unwrap().measurements,
        vec![1, 1]
    );
}

#[test]
fn direct_sampler_supports_ccz_and_depolarize3() {
    let circuit = Circuit::parse("RX 0 1 2\nCCZ 0 1 2\nDEPOLARIZE3(0) 0 1 2\nEXP_VAL X0 X1 X2");
    assert_eq!(circuit.num_qubits(), 3);
    let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 92).unwrap();
    let samples = sampler.sample(1).unwrap();
    assert_eq!(samples.exp_vals, vec![0.5, 0.5, 0.5]);
}

#[test]
fn direct_samplers_support_record_controlled_paulis() {
    let source = "X 0\nM 0\nCX rec[-1] 1\nCY rec[-1] 2\nRX 3\nCZ rec[-1] 3\nM 1 2\nMX 3";
    let circuit = Circuit::parse(source);
    assert_eq!(circuit.num_qubits(), 4);

    let mut measurements = CircuitSampler::from_circuit_with_seed(&circuit, 40).unwrap();
    assert_eq!(
        measurements.sample(1).unwrap().measurements,
        vec![true, true, true, true]
    );
}

#[test]
fn circuit_sampler_reports_detectors() {
    let circuit = Circuit::parse(
        "X_ERROR(1) 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]",
    );
    assert_eq!(circuit.num_detectors(), 2);
    assert_eq!(circuit.num_observables(), 3);

    let mut sampler = CircuitSampler::from_circuit_with_options(&circuit, Some(19), &[]).unwrap();
    let samples = sampler.sample(1).unwrap();
    assert_eq!(samples.accepted_shots, 1);
    assert_eq!(samples.detectors, vec![true, false]);
    assert_eq!(samples.observables, vec![false, false, true]);

    let mut postselected =
        CircuitSampler::with_options("X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]", Some(19), &[0])
            .unwrap();
    assert_eq!(postselected.sample(2).unwrap().accepted_shots, 0);
}

#[test]
fn observable_measurement_and_mpp_are_available() {
    let mut simulator = Simulator::with_seed(18);
    simulator.reset_x(0);
    simulator.cnot(0, 1);
    assert!(!simulator.measure_observable("XX").unwrap());
    assert!(simulator.measure_observable("YY").unwrap());
    assert!(simulator.measure_observable("-ZZ").unwrap());
    assert_eq!(simulator.peek_observable_expectation("XX"), 1.0);
    assert_eq!(simulator.peek_observable_expectation("YY"), -1.0);
    assert_eq!(simulator.peek_observable_expectation("ZZ"), 1.0);

    let circuit = Circuit::parse("RX 0\nCX 0 1\nMPP X0*X1 Y0*Y1 Z0*Z1");
    assert_eq!(circuit.num_measurements(), 3);
    let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 18).unwrap();
    assert_eq!(
        sampler.sample(1).unwrap().measurements,
        vec![false, true, false]
    );
}

#[test]
fn circuit_sampler_reports_expectation_values() {
    let circuit = Circuit::parse("RX 0\nT 0\nEXP_VAL X0 Y0 Z0");
    assert_eq!(circuit.num_exp_vals(), 3);
    let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 18).unwrap();
    let samples = sampler.sample(2).unwrap();
    assert_eq!(samples.accepted_shots, 2);
    for row in samples.exp_vals.chunks_exact(3) {
        assert!((row[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((row[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert_eq!(row[2], 0.0);
    }
}

#[test]
fn circuit_sampler_supports_uniform_fixed_fault_sampling() {
    let circuit = Circuit::parse("X_ERROR(0) 0 1 2\nM 0 1 2");
    let mut sampler = CircuitSampler::from_circuit_with_seed(&circuit, 93).unwrap();
    assert_eq!(sampler.num_fault_sites(), 3);
    let samples = sampler.sample_fixed_faults(32, 1).unwrap();
    for row in samples.measurements.chunks_exact(3) {
        assert_eq!(row.iter().filter(|&&value| value).count(), 1);
    }
    assert_eq!(
        sampler
            .sample_fixed_faults_bit_packed(1, 3)
            .unwrap()
            .measurements,
        vec![0b111]
    );
}

#[test]
fn interactive_backend_is_available() {
    let mut left = Simulator::with_seed(81);
    left.reset_x(0);
    left.t(0);
    left.cnot(0, 1);
    assert_eq!(left.num_qubits(), 2);
    assert_eq!(left.state_vector(StateVectorEndian::Little).len(), 4);
    assert!(left.peek_x_probability(0).is_ok());

    let mut right = left.clone();
    assert_eq!(left.measure_z(1), right.measure_z(1));
    left.reset_x(1);
    assert_eq!(left.peek_x_probability(1).unwrap(), 0.0);
}
