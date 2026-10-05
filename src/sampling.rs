use std::collections::HashSet;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::MeasurementError;
use crate::circuit::{Basis, Circuit, GateInstruction, Instruction, NoiseInstruction};
use crate::simulator::Simulator;

#[derive(Debug)]
struct ExecutionRecord {
    measurements: Vec<bool>,
    detectors: Vec<bool>,
    observables: Vec<bool>,
    exp_vals: Vec<f64>,
}

#[derive(Debug)]
enum ShotOutcome {
    Accepted(ExecutionRecord),
    Rejected,
}

#[derive(Debug)]
struct ExecutionError {
    message: String,
    measurement_index: usize,
}

#[derive(Debug)]
struct FixedFaults {
    indices: Vec<usize>,
    complement: bool,
    position: usize,
    site: usize,
}

impl FixedFaults {
    fn sample(rng: &mut ChaCha8Rng, num_sites: usize, num_faults: usize) -> Self {
        debug_assert!(num_faults <= num_sites);
        let complement = num_faults > num_sites - num_faults;
        let count = if complement {
            num_sites - num_faults
        } else {
            num_faults
        };
        let mut chosen = HashSet::with_capacity(count);
        for upper in num_sites - count..num_sites {
            let candidate = rng.random_range(0..=upper);
            if !chosen.insert(candidate) {
                chosen.insert(upper);
            }
        }
        let mut indices = chosen.into_iter().collect::<Vec<_>>();
        indices.sort_unstable();
        Self {
            indices,
            complement,
            position: 0,
            site: 0,
        }
    }

    fn next(&mut self) -> bool {
        let listed = self
            .indices
            .get(self.position)
            .is_some_and(|&index| index == self.site);
        if listed {
            self.position += 1;
        }
        self.site += 1;
        listed ^ self.complement
    }
}

#[derive(Debug)]
enum FaultMode {
    Disabled,
    Probabilistic,
    Fixed(FixedFaults),
}

impl FaultMode {
    fn event(&mut self, simulator: &mut Simulator, probability: f64) -> bool {
        match self {
            Self::Disabled => false,
            Self::Probabilistic => simulator.sample_event(probability),
            Self::Fixed(faults) => faults.next(),
        }
    }

    fn records_exp_vals(&self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

fn execute_circuit(
    circuit: &Circuit,
    rng: ChaCha8Rng,
    mut fault_mode: FaultMode,
    detector_reference: Option<&[bool]>,
    postselect: Option<&[bool]>,
) -> (ChaCha8Rng, Result<ShotOutcome, ExecutionError>) {
    let mut simulator = Simulator::from_rng(rng);
    simulator.ensure_num_qubits(circuit.num_qubits);
    let mut measurements = Vec::new();
    let mut detectors = Vec::new();
    let mut observables = vec![false; circuit.num_observables()];
    let mut exp_vals = Vec::new();

    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate(gate) => apply_gate(&mut simulator, gate),
            Instruction::ClassicallyControlledPauli {
                pauli,
                lookback,
                target,
            } => {
                if measurements[measurements.len() - lookback] {
                    apply_pauli(&mut simulator, *target, *pauli);
                }
            }
            Instruction::Noise(noise) => {
                apply_noise(&mut simulator, noise, &mut fault_mode);
            }
            Instruction::Reset { basis, qubits } => {
                for &qubit in qubits {
                    match basis {
                        Basis::Z => simulator.reset_z_raw(qubit),
                        Basis::X => simulator.reset_x_raw(qubit),
                    }
                }
            }
            Instruction::Measure {
                basis,
                qubits,
                readout_probability,
                reset,
            } => {
                for &(qubit, inverted) in qubits {
                    let measurement_index = measurements.len();
                    let minus = match basis {
                        Basis::Z => simulator.measure_z_raw(qubit).0 == -1,
                        Basis::X => match simulator.measure_x_raw(qubit) {
                            Ok((outcome, _)) => outcome == -1,
                            Err(message) => {
                                let rng = simulator.into_rng();
                                return (
                                    rng,
                                    Err(ExecutionError {
                                        message,
                                        measurement_index,
                                    }),
                                );
                            }
                        },
                    };

                    if *reset && minus {
                        match basis {
                            Basis::Z => simulator.x(qubit),
                            Basis::X => simulator.z(qubit),
                        }
                    }
                    let readout_flip = readout_probability
                        .is_some_and(|probability| fault_mode.event(&mut simulator, probability));
                    measurements.push(minus ^ inverted ^ readout_flip);
                }
            }
            Instruction::MeasurePauliProduct {
                products,
                readout_probability,
            } => {
                for product in products {
                    let measurement_index = measurements.len();
                    let minus = match simulator.measure_pauli_product_raw(product) {
                        Ok((outcome, _)) => outcome == -1,
                        Err(message) => {
                            let rng = simulator.into_rng();
                            return (
                                rng,
                                Err(ExecutionError {
                                    message,
                                    measurement_index,
                                }),
                            );
                        }
                    };
                    let readout_flip = readout_probability
                        .is_some_and(|probability| fault_mode.event(&mut simulator, probability));
                    measurements.push(minus ^ readout_flip);
                }
            }
            Instruction::ExpectationValue { products } => {
                if fault_mode.records_exp_vals() {
                    exp_vals.extend(
                        products
                            .iter()
                            .map(|product| simulator.pauli_product_expectation(product)),
                    );
                }
            }
            Instruction::Detector { lookbacks } => {
                let raw = parity_from_lookbacks(&measurements, lookbacks);
                let detector =
                    detector_reference.map_or(raw, |reference| raw ^ reference[detectors.len()]);
                detectors.push(detector);
                if postselect.is_some_and(|mask| mask[detectors.len() - 1] && detector) {
                    return (simulator.into_rng(), Ok(ShotOutcome::Rejected));
                }
            }
            Instruction::ObservableInclude {
                observable,
                lookbacks,
            } => {
                observables[*observable] ^= parity_from_lookbacks(&measurements, lookbacks);
            }
        }
    }

    let rng = simulator.into_rng();
    (
        rng,
        Ok(ShotOutcome::Accepted(ExecutionRecord {
            measurements,
            detectors,
            observables,
            exp_vals,
        })),
    )
}

fn parity_from_lookbacks(measurements: &[bool], lookbacks: &[usize]) -> bool {
    lookbacks.iter().fold(false, |parity, lookback| {
        parity ^ measurements[measurements.len() - lookback]
    })
}

/// Flat row-major results produced by [`CircuitSampler`].
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitSamples<T> {
    /// Number of attempted shots.
    pub total_shots: usize,
    /// Number of accepted rows represented by the buffers.
    pub accepted_shots: usize,
    /// Measurement data with row width [`CircuitSampler::num_measurements`].
    pub measurements: Vec<T>,
    /// Detector data with row width [`CircuitSampler::num_detectors`].
    pub detectors: Vec<T>,
    /// Observable data with row width [`CircuitSampler::num_observables`].
    pub observables: Vec<T>,
    /// Expectation values with row width [`CircuitSampler::num_exp_vals`].
    pub exp_vals: Vec<f64>,
}

/// A parse-once circuit sampler backed by [`Simulator`].
///
/// Every attempted shot starts in a fresh all-zero state. Measurements,
/// resets, feedback, detector annotations, and expectation probes execute at
/// their written positions. Detector and observable bits are normalized by a
/// noiseless reference replay performed during construction.
///
/// ```
/// use merlin::CircuitSampler;
///
/// let mut sampler = CircuitSampler::with_seed(
///     "RX 0\nT 0\nEXP_VAL X0\nMX 0",
///     7,
/// )?;
/// let samples = sampler.sample(1)?;
/// assert_eq!(samples.measurements.len(), 1);
/// assert_eq!(samples.exp_vals.len(), 1);
/// # Ok::<(), merlin::MeasurementError>(())
/// ```
#[derive(Debug)]
pub struct CircuitSampler {
    circuit: Circuit,
    reference_detectors: Vec<bool>,
    reference_observables: Vec<bool>,
    postselect: Vec<bool>,
    num_fault_sites: usize,
    contains_pauli_channel: bool,
    rng: ChaCha8Rng,
}

impl CircuitSampler {
    /// Parses a circuit and creates a sampler using system entropy.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if the noiseless reference replay contains
    /// an incompatible measurement.
    ///
    /// # Panics
    /// Panics when the circuit source is malformed or unsupported.
    pub fn new(source: &str) -> Result<Self, MeasurementError> {
        Self::from_circuit(&Circuit::parse(source))
    }

    /// Parses a circuit and creates a sampler with a reproducible shot seed.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if the noiseless reference replay contains
    /// an incompatible measurement.
    ///
    /// # Panics
    /// Panics when the circuit source is malformed or unsupported.
    pub fn with_seed(source: &str, seed: u64) -> Result<Self, MeasurementError> {
        Self::from_circuit_with_seed(&Circuit::parse(source), seed)
    }

    /// Parses a circuit and creates a sampler with explicit options.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if the noiseless reference replay contains
    /// an incompatible measurement.
    ///
    /// # Panics
    /// Panics for malformed circuit source, duplicate postselection indices, or
    /// a postselection index outside the circuit's detector range.
    pub fn with_options(
        source: &str,
        seed: Option<u64>,
        postselect: &[usize],
    ) -> Result<Self, MeasurementError> {
        Self::from_circuit_with_options(&Circuit::parse(source), seed, postselect)
    }

    /// Creates a sampler from a parsed circuit using system entropy.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if reference generation is incompatible.
    pub fn from_circuit(circuit: &Circuit) -> Result<Self, MeasurementError> {
        Self::from_circuit_with_options(circuit, None, &[])
    }

    /// Creates a seeded sampler from a parsed circuit.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if reference generation is incompatible.
    pub fn from_circuit_with_seed(circuit: &Circuit, seed: u64) -> Result<Self, MeasurementError> {
        Self::from_circuit_with_options(circuit, Some(seed), &[])
    }

    /// Creates a sampler from a parsed circuit with explicit options.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if reference generation is incompatible.
    ///
    /// # Panics
    /// Panics for duplicate or out-of-range postselection indices.
    pub fn from_circuit_with_options(
        circuit: &Circuit,
        seed: Option<u64>,
        postselect: &[usize],
    ) -> Result<Self, MeasurementError> {
        let mut postselection_mask = vec![false; circuit.num_detectors()];
        for &detector in postselect {
            assert!(
                detector < postselection_mask.len(),
                "postselected detector index {detector} is out of range"
            );
            assert!(
                !postselection_mask[detector],
                "postselected detector index {detector} appears more than once"
            );
            postselection_mask[detector] = true;
        }

        let (_, reference) = execute_circuit(
            circuit,
            ChaCha8Rng::seed_from_u64(0),
            FaultMode::Disabled,
            None,
            None,
        );
        let reference = match reference {
            Ok(ShotOutcome::Accepted(reference)) => reference,
            Ok(ShotOutcome::Rejected) => unreachable!("reference generation does not postselect"),
            Err(error) => {
                return Err(MeasurementError::at_measurement(
                    error.message,
                    error.measurement_index,
                ));
            }
        };

        Ok(Self {
            circuit: circuit.clone(),
            reference_detectors: reference.detectors,
            reference_observables: reference.observables,
            postselect: postselection_mask,
            num_fault_sites: circuit.num_fault_sites(),
            contains_pauli_channel: circuit.contains_pauli_channel(),
            rng: seed.map_or_else(ChaCha8Rng::from_os_rng, ChaCha8Rng::seed_from_u64),
        })
    }

    /// Returns the number of qubits referenced by the circuit.
    pub fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    /// Returns the number of measurement records produced per complete shot.
    pub fn num_measurements(&self) -> usize {
        self.circuit.num_measurements()
    }

    /// Returns the number of detector columns.
    pub fn num_detectors(&self) -> usize {
        self.circuit.num_detectors()
    }

    /// Returns the number of observable columns.
    pub fn num_observables(&self) -> usize {
        self.circuit.num_observables()
    }

    /// Returns the number of expectation-value columns.
    pub fn num_exp_vals(&self) -> usize {
        self.circuit.num_exp_vals()
    }

    /// Returns the number of explicit fault sites in the circuit.
    ///
    /// Each noise target or target group is one site. Each target of a
    /// measurement with an explicitly written readout probability is also one
    /// site, including when that probability is zero.
    pub fn num_fault_sites(&self) -> usize {
        self.num_fault_sites
    }

    /// Samples measurements, detector flips, observables, and expectations.
    ///
    /// `shots` is the number of attempts. Postselected attempts are omitted,
    /// so [`CircuitSamples::accepted_shots`] can be smaller. Boolean buffers
    /// are flat and row-major; expectation values are always `f64`.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if an attempted shot encounters an
    /// incompatible measurement.
    pub fn sample(&mut self, shots: usize) -> Result<CircuitSamples<bool>, MeasurementError> {
        self.sample_with(shots, SamplingMode::Probabilistic)
    }

    /// Samples uniformly from circuit executions having exactly `num_faults`
    /// faults.
    ///
    /// Fault locations are selected uniformly without replacement and their
    /// configured probabilities are ignored. Postselected attempts are
    /// stopped and omitted exactly as in [`Self::sample`]. This method returns
    /// conditional samples only; callers are responsible for weighting fault
    /// strata when estimating an unconditional quantity.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if an attempted shot encounters an
    /// incompatible measurement.
    ///
    /// # Panics
    /// Panics when `num_faults` exceeds [`Self::num_fault_sites`] or when the
    /// circuit contains `PAULI_CHANNEL_1` or `PAULI_CHANNEL_2`.
    pub fn sample_fixed_faults(
        &mut self,
        shots: usize,
        num_faults: usize,
    ) -> Result<CircuitSamples<bool>, MeasurementError> {
        self.validate_fixed_faults(num_faults);
        self.sample_with(shots, SamplingMode::Fixed(num_faults))
    }

    fn validate_fixed_faults(&self, num_faults: usize) {
        assert!(
            !self.contains_pauli_channel,
            "fixed-fault sampling does not support PAULI_CHANNEL_1 or PAULI_CHANNEL_2"
        );
        assert!(
            num_faults <= self.num_fault_sites,
            "num_faults ({num_faults}) exceeds the number of fault sites ({})",
            self.num_fault_sites
        );
    }

    #[cfg(feature = "python")]
    pub(crate) fn fixed_fault_error(&self, num_faults: usize) -> Option<String> {
        if self.contains_pauli_channel {
            Some(
                "fixed-fault sampling does not support PAULI_CHANNEL_1 or PAULI_CHANNEL_2"
                    .to_owned(),
            )
        } else if num_faults > self.num_fault_sites {
            Some(format!(
                "num_faults ({num_faults}) exceeds the number of fault sites ({})",
                self.num_fault_sites
            ))
        } else {
            None
        }
    }

    fn sample_with(
        &mut self,
        shots: usize,
        mode: SamplingMode,
    ) -> Result<CircuitSamples<bool>, MeasurementError> {
        let mut measurements = Vec::new();
        let mut detectors = Vec::new();
        let mut observables = Vec::new();
        let mut exp_vals = Vec::new();
        let mut accepted_shots = 0;
        for shot in 0..shots {
            let mut shot_rng = self.rng.clone();
            let fault_mode = match mode {
                SamplingMode::Probabilistic => FaultMode::Probabilistic,
                SamplingMode::Fixed(num_faults) => FaultMode::Fixed(FixedFaults::sample(
                    &mut shot_rng,
                    self.num_fault_sites,
                    num_faults,
                )),
            };
            let (rng, result) = execute_circuit(
                &self.circuit,
                shot_rng,
                fault_mode,
                Some(&self.reference_detectors),
                Some(&self.postselect),
            );
            self.rng = rng;
            match result {
                Ok(ShotOutcome::Accepted(mut record)) => {
                    for (value, reference) in record
                        .observables
                        .iter_mut()
                        .zip(&self.reference_observables)
                    {
                        *value ^= reference;
                    }
                    accepted_shots += 1;
                    measurements.extend(record.measurements);
                    detectors.extend(record.detectors);
                    observables.extend(record.observables);
                    exp_vals.extend(record.exp_vals);
                }
                Ok(ShotOutcome::Rejected) => {}
                Err(error) => {
                    return Err(MeasurementError::at(
                        error.message,
                        shot,
                        error.measurement_index,
                    ));
                }
            }
        }
        Ok(CircuitSamples {
            total_shots: shots,
            accepted_shots,
            measurements,
            detectors,
            observables,
            exp_vals,
        })
    }

    /// Samples with each boolean output family independently bit-packed.
    ///
    /// Bits are little-endian within each byte. Each row is padded separately.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if an attempted shot encounters an
    /// incompatible measurement.
    pub fn sample_bit_packed(
        &mut self,
        shots: usize,
    ) -> Result<CircuitSamples<u8>, MeasurementError> {
        let samples = self.sample(shots)?;
        Ok(self.pack_samples(samples))
    }

    /// Uniform fixed-fault sampling with independently bit-packed outputs.
    ///
    /// This has the same behavior and validation as [`Self::sample_fixed_faults`].
    /// Bits are little-endian within each byte and each output family is padded
    /// independently.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] if an attempted shot encounters an
    /// incompatible measurement.
    ///
    /// # Panics
    /// Panics when `num_faults` is invalid or the circuit contains a Pauli channel.
    pub fn sample_fixed_faults_bit_packed(
        &mut self,
        shots: usize,
        num_faults: usize,
    ) -> Result<CircuitSamples<u8>, MeasurementError> {
        let samples = self.sample_fixed_faults(shots, num_faults)?;
        Ok(self.pack_samples(samples))
    }

    fn pack_samples(&self, samples: CircuitSamples<bool>) -> CircuitSamples<u8> {
        CircuitSamples {
            total_shots: samples.total_shots,
            accepted_shots: samples.accepted_shots,
            measurements: pack_rows(
                &samples.measurements,
                samples.accepted_shots,
                self.num_measurements(),
            ),
            detectors: pack_rows(
                &samples.detectors,
                samples.accepted_shots,
                self.num_detectors(),
            ),
            observables: pack_rows(
                &samples.observables,
                samples.accepted_shots,
                self.num_observables(),
            ),
            exp_vals: samples.exp_vals,
        }
    }
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Probabilistic,
    Fixed(usize),
}

fn pack_rows(values: &[bool], rows: usize, columns: usize) -> Vec<u8> {
    let width = columns.div_ceil(8);
    let mut packed = vec![0; rows * width];
    for row in 0..rows {
        for column in 0..columns {
            if values[row * columns + column] {
                packed[row * width + (column >> 3)] |= 1 << (column & 7);
            }
        }
    }
    packed
}

fn apply_gate(simulator: &mut Simulator, gate: &GateInstruction) {
    match gate {
        GateInstruction::I(_) => {}
        GateInstruction::X(qubits) => {
            for &qubit in qubits {
                simulator.x(qubit);
            }
        }
        GateInstruction::Y(qubits) => {
            for &qubit in qubits {
                simulator.y(qubit);
            }
        }
        GateInstruction::Z(qubits) => {
            for &qubit in qubits {
                simulator.z(qubit);
            }
        }
        GateInstruction::S(qubits) => {
            for &qubit in qubits {
                simulator.s(qubit);
            }
        }
        GateInstruction::Sdg(qubits) => {
            for &qubit in qubits {
                simulator.s_dag(qubit);
            }
        }
        GateInstruction::T(qubits) => {
            for &qubit in qubits {
                simulator.t(qubit);
            }
        }
        GateInstruction::Tdg(qubits) => {
            for &qubit in qubits {
                simulator.t_dag(qubit);
            }
        }
        GateInstruction::Cx(pairs) => {
            for &(control, target) in pairs {
                simulator.cnot(control, target);
            }
        }
        GateInstruction::Cz(pairs) => {
            for &(left, right) in pairs {
                simulator.cz(left, right);
            }
        }
        GateInstruction::Ccz(triples) => {
            for &(first, second, third) in triples {
                simulator.ccz(first, second, third);
            }
        }
        GateInstruction::Swap(pairs) => {
            for &(left, right) in pairs {
                simulator.swap(left, right);
            }
        }
    }
}

fn apply_pauli(simulator: &mut Simulator, qubit: usize, pauli: crate::circuit::Pauli) {
    match pauli {
        crate::circuit::Pauli::X => simulator.x(qubit),
        crate::circuit::Pauli::Y => simulator.y(qubit),
        crate::circuit::Pauli::Z => simulator.z(qubit),
    }
}

fn apply_noise(simulator: &mut Simulator, noise: &NoiseInstruction, fault_mode: &mut FaultMode) {
    match noise {
        NoiseInstruction::Error {
            pauli,
            qubits,
            probability,
        } => {
            for &q in qubits {
                if fault_mode.event(simulator, *probability) {
                    apply_pauli(simulator, q, *pauli);
                }
            }
        }
        NoiseInstruction::Depolarize1 {
            qubits,
            probability,
        } => {
            for &q in qubits {
                if fault_mode.event(simulator, *probability) {
                    let code = simulator.random_pauli_code(1);
                    simulator.apply_pauli_code(q, code);
                }
            }
        }
        NoiseInstruction::Depolarize2 { pairs, probability } => {
            for &(q1, q2) in pairs {
                if fault_mode.event(simulator, *probability) {
                    let code = simulator.random_pauli_code(2);
                    simulator.apply_pauli_code(q1, code >> 2);
                    simulator.apply_pauli_code(q2, code & 3);
                }
            }
        }
        NoiseInstruction::Depolarize3 {
            triples,
            probability,
        } => {
            for &(q1, q2, q3) in triples {
                if fault_mode.event(simulator, *probability) {
                    let code = simulator.random_pauli_code(3);
                    simulator.apply_pauli_code(q1, code >> 4);
                    simulator.apply_pauli_code(q2, (code >> 2) & 3);
                    simulator.apply_pauli_code(q3, code & 3);
                }
            }
        }
        NoiseInstruction::PauliChannel1 {
            qubits,
            probabilities,
        } => match fault_mode {
            FaultMode::Disabled => {}
            FaultMode::Probabilistic => {
                for &q in qubits {
                    simulator.pauli_channel1(q, probabilities);
                }
            }
            FaultMode::Fixed(_) => {
                unreachable!("Pauli channels are rejected before fixed-fault sampling")
            }
        },
        NoiseInstruction::PauliChannel2 {
            pairs,
            probabilities,
        } => match fault_mode {
            FaultMode::Disabled => {}
            FaultMode::Probabilistic => {
                for &(q1, q2) in pairs {
                    simulator.pauli_channel2(q1, q2, probabilities);
                }
            }
            FaultMode::Fixed(_) => {
                unreachable!("Pauli channels are rejected before fixed-fault sampling")
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorEndian;
    use num_complex::Complex32;
    use rand::RngCore;

    struct MeasurementSampler(CircuitSampler);

    impl MeasurementSampler {
        fn with_seed(source: &str, seed: u64) -> Self {
            Self(CircuitSampler::with_seed(source, seed).unwrap())
        }

        fn num_qubits(&self) -> usize {
            self.0.num_qubits()
        }

        fn num_measurements(&self) -> usize {
            self.0.num_measurements()
        }

        fn sample(&mut self, shots: usize) -> Result<Vec<bool>, MeasurementError> {
            self.0.sample(shots).map(|samples| samples.measurements)
        }

        fn sample_bit_packed(&mut self, shots: usize) -> Result<Vec<u8>, MeasurementError> {
            self.0
                .sample_bit_packed(shots)
                .map(|samples| samples.measurements)
        }
    }

    type DetectorSampler = CircuitSampler;

    fn assert_vector_close(actual: &[Complex32], expected: &[Complex32]) {
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected) {
            assert!((*left - *right).norm() < 1e-6, "{left:?} != {right:?}");
        }
    }

    #[test]
    fn phase_and_pair_gate_lowering_matches_dense_evolution() {
        let scale = std::f32::consts::FRAC_1_SQRT_2;

        let mut s = Simulator::with_seed(10);
        s.ensure_num_qubits(1);
        s.reset_x_raw(0);
        apply_gate(&mut s, &GateInstruction::S(vec![0]));
        assert_vector_close(
            &s.state_vector(StateVectorEndian::Little),
            &[Complex32::new(scale, 0.0), Complex32::new(0.0, scale)],
        );

        let mut cz = Simulator::with_seed(11);
        cz.ensure_num_qubits(2);
        cz.reset_x_raw(0);
        cz.reset_x_raw(1);
        apply_gate(&mut cz, &GateInstruction::Cz(vec![(0, 1)]));
        assert_vector_close(
            &cz.state_vector(StateVectorEndian::Little),
            &[
                Complex32::new(0.5, 0.0),
                Complex32::new(0.5, 0.0),
                Complex32::new(0.5, 0.0),
                Complex32::new(-0.5, 0.0),
            ],
        );

        let mut swap = Simulator::with_seed(12);
        swap.ensure_num_qubits(2);
        swap.x(0);
        apply_gate(&mut swap, &GateInstruction::Swap(vec![(0, 1)]));
        assert_vector_close(
            &swap.state_vector(StateVectorEndian::Little),
            &[
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
            ],
        );
    }

    #[test]
    fn deterministic_gates_and_mid_circuit_measurements() {
        let mut sampler =
            MeasurementSampler::with_seed("X 0\nM 0\nR 0\nX 1\nCZ 0 1\nSWAP 0 1\nM 0 1", 1);
        assert_eq!(sampler.sample(1).unwrap(), vec![true, true, false]);
    }

    #[test]
    fn measurement_reset_corrects_the_collapsed_state() {
        let mut z = MeasurementSampler::with_seed("X 0\nMR 0\nM 0", 2);
        assert_eq!(z.sample(1).unwrap(), vec![true, false]);

        let mut x = MeasurementSampler::with_seed("RX 0\nZ 0\nMRX 0\nMX 0", 2);
        assert_eq!(x.sample(1).unwrap(), vec![true, false]);
    }

    #[test]
    fn packing_is_little_endian() {
        let source = "X 0 2 7 8\nM 0 1 2 3 4 5 6 7 8";
        let mut sampler = MeasurementSampler::with_seed(source, 3);
        assert_eq!(sampler.sample_bit_packed(1).unwrap(), vec![0b1000_0101, 1]);
    }

    #[test]
    fn repeats_readout_inversion_and_wide_targets_execute_in_place() {
        let mut repeated =
            MeasurementSampler::with_seed("REPEAT 2 {\nX 0\nM 0\n}\nM(1) !0\nM 0", 13);
        assert_eq!(repeated.sample(1).unwrap(), vec![true, false, false, false]);

        let mut wide = MeasurementSampler::with_seed("X 129\nM 129", 14);
        assert_eq!(wide.num_qubits(), 130);
        assert_eq!(wide.sample(1).unwrap(), vec![true]);
    }

    #[test]
    fn seeded_sampling_is_reproducible_across_calls() {
        let source = "RX 0\nX_ERROR(.25) 0\nM 0";
        let mut split = MeasurementSampler::with_seed(source, 4);
        let mut together = MeasurementSampler::with_seed(source, 4);
        let mut split_samples = split.sample(7).unwrap();
        split_samples.extend(split.sample(9).unwrap());
        assert_eq!(split_samples, together.sample(16).unwrap());
    }

    #[test]
    fn supports_pauli_channels() {
        let mut sampler = MeasurementSampler::with_seed(
            "PAULI_CHANNEL_1(1,0,0) 0\nPAULI_CHANNEL_2(0,0,0,1,0,0,0,0,0,0,0,0,0,0,0) 1 2\nM 0 1 2",
            5,
        );
        assert_eq!(sampler.sample(1).unwrap(), vec![true, true, false]);
    }

    #[test]
    fn expectation_values_are_reported_at_their_instruction_positions() {
        let mut sampler =
            CircuitSampler::with_seed("RX 0\nT 0\nEXP_VAL X0 Y0 Z0\nT_DAG 0\nEXP_VAL X0", 6)
                .unwrap();
        assert_eq!(sampler.num_exp_vals(), 4);
        let samples = sampler.sample(2).unwrap();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        for row in samples.exp_vals.chunks_exact(4) {
            assert!((row[0] - expected).abs() < 1e-12);
            assert!((row[1] - expected).abs() < 1e-12);
            assert_eq!(row[2], 0.0);
            assert_eq!(row[3], 1.0);
        }
    }

    #[test]
    fn mpp_products_measure_sequentially_with_inversion_and_readout_noise() {
        let source = "RX 0\nCX 0 1\nMPP X0*X1 Y0*Y1 Z0*Z1\nX 0\nMPP Z0*Z1\nMPP(1) !Z0*Z1";
        let mut sampler = MeasurementSampler::with_seed(source, 15);
        assert_eq!(sampler.num_qubits(), 2);
        assert_eq!(sampler.num_measurements(), 5);
        assert_eq!(
            sampler.sample(1).unwrap(),
            vec![false, true, false, true, true]
        );
    }

    #[test]
    fn record_controlled_paulis_use_the_reported_measurement_bit() {
        let source = "X 0\nM 0\nCX rec[-1] 1\nCY rec[-1] 2\nRX 3\nCZ rec[-1] 3\nM 1 2\nMX 3";
        let mut sampler = MeasurementSampler::with_seed(source, 17);
        assert_eq!(sampler.sample(1).unwrap(), vec![true, true, true, true]);

        let mut false_controls = MeasurementSampler::with_seed(
            "M 0\nCX rec[-1] 1\nCY rec[-1] 2\nRX 3\nCZ rec[-1] 3\nM 1 2\nMX 3",
            18,
        );
        assert_eq!(
            false_controls.sample(1).unwrap(),
            vec![false, false, false, false]
        );

        let mut inverted = MeasurementSampler::with_seed("M !0\nCX rec[-1] 1\nM 1", 19);
        assert_eq!(inverted.sample(1).unwrap(), vec![true, true]);

        let mut readout = MeasurementSampler::with_seed("M(1) 0\nCX rec[-1] 1\nM 1", 20);
        assert_eq!(readout.sample(1).unwrap(), vec![true, true]);

        let mut product =
            MeasurementSampler::with_seed("RX 1\nX 0\nMPP Z0\nCZ rec[-1] 1\nMX 1", 21);
        assert_eq!(product.sample(1).unwrap(), vec![true, true]);
    }

    #[test]
    fn mixed_feedback_and_quantum_pairs_execute_in_source_order() {
        let mut sampler = MeasurementSampler::with_seed("X 0\nM 0\nCX rec[-1] 1 1 2\nM 1 2", 22);
        assert_eq!(sampler.sample(1).unwrap(), vec![true, true, true]);
    }

    #[test]
    fn repeated_wide_mpp_products_pack_in_record_order() {
        let source = "X 129\nREPEAT 2 {\nMPP Z129 !Z129\n}";
        let mut sampler = MeasurementSampler::with_seed(source, 16);
        assert_eq!(sampler.num_qubits(), 130);
        assert_eq!(sampler.num_measurements(), 4);
        assert_eq!(sampler.sample_bit_packed(1).unwrap(), vec![0b0101]);
    }

    #[test]
    fn detector_samples_are_normalized_against_the_eager_reference() {
        let mut expected_odd = DetectorSampler::with_seed(
            "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            20,
        )
        .unwrap();
        assert_eq!(expected_odd.num_detectors(), 1);
        assert_eq!(expected_odd.num_observables(), 1);
        let samples = expected_odd.sample(2).unwrap();
        assert_eq!(samples.accepted_shots, 2);
        assert_eq!(samples.detectors, vec![false, false]);
        assert_eq!(samples.observables, vec![false, false]);

        let mut noisy = DetectorSampler::with_seed(
            "X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            21,
        )
        .unwrap();
        let samples = noisy.sample(1).unwrap();
        assert_eq!(samples.detectors, vec![true]);
        assert_eq!(samples.observables, vec![true]);

        let mut readout = DetectorSampler::with_seed("M(1) 0\nDETECTOR rec[-1]", 22).unwrap();
        assert_eq!(readout.sample(1).unwrap().detectors, vec![true]);
    }

    #[test]
    fn feedback_participates_in_detector_reference_and_noisy_shots() {
        let source = "M !0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let mut reference_feedback = DetectorSampler::with_seed(source, 29).unwrap();
        let samples = reference_feedback.sample(1).unwrap();
        assert_eq!(samples.detectors, vec![false]);
        assert_eq!(samples.observables, vec![false]);

        let noisy_source =
            "M(1) 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let mut noisy_feedback = DetectorSampler::with_seed(noisy_source, 30).unwrap();
        let samples = noisy_feedback.sample(1).unwrap();
        assert_eq!(samples.detectors, vec![true]);
        assert_eq!(samples.observables, vec![true]);
    }

    #[test]
    fn postselection_aborts_before_later_feedback_and_measurement() {
        let source = "M(1) 0\nDETECTOR rec[-1]\nCX rec[-1] 1\nMX 2";
        let mut sampler = DetectorSampler::with_options(source, Some(31), &[0]).unwrap();
        let samples = sampler.sample(1).unwrap();
        assert_eq!(samples.accepted_shots, 0);
    }

    #[test]
    fn observable_fragments_sparse_indices_and_gauges_are_supported() {
        let source = "X_ERROR(1) 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n\
                      OBSERVABLE_INCLUDE(2) rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-1]";
        let mut sampler = DetectorSampler::with_seed(source, 23).unwrap();
        let samples = sampler.sample(1).unwrap();
        assert_eq!(samples.detectors, vec![true, false]);
        assert_eq!(samples.observables, vec![false, false, true]);

        let mut gauge =
            DetectorSampler::with_seed("RX 0\nM 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]", 24)
                .unwrap();
        for row in gauge.sample(32).unwrap().detectors.chunks_exact(2) {
            assert_eq!(row[0], row[1]);
        }
    }

    #[test]
    fn postselection_omits_rows_and_aborts_before_later_rng_draws() {
        let source = "X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nX_ERROR(.25) 1\nM 1";
        let mut sampler = DetectorSampler::with_options(source, Some(25), &[0]).unwrap();
        let samples = sampler.sample(5).unwrap();
        assert_eq!(samples.accepted_shots, 0);
        assert!(samples.detectors.is_empty());

        let circuit = Circuit::parse(source);
        let truncated = Circuit::parse("X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]");
        let mask = [true];
        let reference = [false];
        let (mut actual_rng, actual) = execute_circuit(
            &circuit,
            ChaCha8Rng::seed_from_u64(26),
            FaultMode::Probabilistic,
            Some(&reference),
            Some(&mask),
        );
        let (mut expected_rng, expected) = execute_circuit(
            &truncated,
            ChaCha8Rng::seed_from_u64(26),
            FaultMode::Probabilistic,
            Some(&reference),
            Some(&mask),
        );
        assert!(matches!(actual, Ok(ShotOutcome::Rejected)));
        assert!(matches!(expected, Ok(ShotOutcome::Rejected)));
        assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
    }

    #[test]
    fn detector_bit_packing_uses_independent_row_widths() {
        let source = "X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR\nDETECTOR\n\
                      DETECTOR\nDETECTOR\nDETECTOR\nDETECTOR\nDETECTOR\nDETECTOR\n\
                      OBSERVABLE_INCLUDE(8) rec[-1]";
        let mut sampler = DetectorSampler::with_seed(source, 27).unwrap();
        let samples = sampler.sample_bit_packed(1).unwrap();
        assert_eq!(samples.accepted_shots, 1);
        assert_eq!(samples.detectors, vec![1, 0]);
        assert_eq!(samples.observables, vec![0, 1]);
    }

    #[test]
    fn incompatible_reference_fails_during_construction() {
        let mut lines = vec![
            "RX 0 1 2".to_owned(),
            "T 0".to_owned(),
            "T 0".to_owned(),
            "T 1".to_owned(),
            "T 2".to_owned(),
        ];
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            lines.push(format!("CX {left} {right}"));
            lines.push(format!("T_DAG {right}"));
            lines.push(format!("CX {left} {right}"));
        }
        lines.extend([
            "CX 0 2".to_owned(),
            "CX 1 2".to_owned(),
            "T 2".to_owned(),
            "CX 1 2".to_owned(),
            "CX 0 2".to_owned(),
            "MX 0".to_owned(),
        ]);
        let error = DetectorSampler::with_seed(&lines.join("\n"), 28).unwrap_err();
        assert_eq!(error.shot(), None);
        assert_eq!(error.measurement_index(), Some(0));
        assert!(error.to_string().contains("reference measurement 0"));
    }

    #[test]
    fn fixed_fault_subsets_have_exact_cardinality() {
        for num_sites in 0..20 {
            for k in 0..=num_sites {
                let mut rng = ChaCha8Rng::seed_from_u64((num_sites * 31 + k) as u64);
                let mut faults = FixedFaults::sample(&mut rng, num_sites, k);
                assert_eq!((0..num_sites).filter(|_| faults.next()).count(), k);
            }
        }
    }

    #[test]
    fn fixed_fault_sampling_forces_exactly_k_explicit_sites() {
        let source = "X_ERROR(0) 0 1 2 3 4\nM 0 1 2 3 4";
        let mut sampler = CircuitSampler::with_seed(source, 40).unwrap();
        assert_eq!(sampler.num_fault_sites(), 5);
        let samples = sampler.sample_fixed_faults(100, 2).unwrap();
        assert_eq!(samples.total_shots, 100);
        assert_eq!(samples.accepted_shots, 100);
        for row in samples.measurements.chunks_exact(5) {
            assert_eq!(row.iter().filter(|&&value| value).count(), 2);
        }

        let zero = sampler.sample_fixed_faults(1, 0).unwrap();
        assert_eq!(zero.measurements, vec![false; 5]);
        let all = sampler.sample_fixed_faults(1, 5).unwrap();
        assert_eq!(all.measurements, vec![true; 5]);
    }

    #[test]
    fn explicit_zero_readout_arguments_are_fault_sites() {
        let source = "M 0\nM(0) 1\nMPP(0) Z2\nMR(0) 3\nRX 4\nMRX(0) 4";
        let mut sampler = CircuitSampler::with_seed(source, 41).unwrap();
        assert_eq!(sampler.num_fault_sites(), 4);
        let samples = sampler.sample_fixed_faults(1, 4).unwrap();
        assert_eq!(samples.measurements, vec![false, true, true, true, true]);
    }

    #[test]
    fn fixed_fault_sampling_is_seeded_packable_and_postselection_aware() {
        let source = "X_ERROR(0) 0 1 2\nM 0 1 2\nDETECTOR rec[-3]";
        let mut left = CircuitSampler::with_seed(source, 42).unwrap();
        let mut right = CircuitSampler::with_seed(source, 42).unwrap();
        assert_eq!(
            left.sample_fixed_faults(32, 1).unwrap(),
            right.sample_fixed_faults(32, 1).unwrap()
        );

        let mut packed = CircuitSampler::with_seed(source, 43).unwrap();
        let packed = packed.sample_fixed_faults_bit_packed(1, 3).unwrap();
        assert_eq!(packed.measurements, vec![0b111]);

        let mut postselected = CircuitSampler::with_options(source, Some(44), &[0]).unwrap();
        let samples = postselected.sample_fixed_faults(100, 1).unwrap();
        assert_eq!(samples.total_shots, 100);
        assert!(samples.accepted_shots > 0 && samples.accepted_shots < 100);
        assert!(samples.detectors.iter().all(|value| !value));
    }

    #[test]
    fn fixed_fault_validation_precedes_rng_consumption() {
        let source = "X_ERROR(0) 0\nM 0";
        let mut invalid = CircuitSampler::with_seed(source, 45).unwrap();
        let mut control = CircuitSampler::with_seed(source, 45).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = invalid.sample_fixed_faults(0, 2);
            }))
            .is_err()
        );
        assert_eq!(
            invalid.sample_fixed_faults(20, 1).unwrap(),
            control.sample_fixed_faults(20, 1).unwrap()
        );

        let channel = "PAULI_CHANNEL_1(1,0,0) 0\nM 0";
        let mut unsupported = CircuitSampler::with_seed(channel, 46).unwrap();
        assert_eq!(unsupported.num_fault_sites(), 1);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = unsupported.sample_fixed_faults(0, 0);
            }))
            .is_err()
        );
    }
}
