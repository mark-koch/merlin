use crate::{
    MeasurementError, Simulator,
    bits::BitVec,
    circuit::{Pauli, PauliProduct},
};

impl Simulator {
    /// Measures a Hermitian Pauli-string observable.
    ///
    /// The observable uses compact positional notation, for example `"-X_YZ"`.
    /// `_` and `I` are identity factors. The returned bit is `false` for the
    /// observable's +1 eigenvalue and `true` for its -1 eigenvalue.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] when the projection is incompatible with
    /// the DCP representation. In that case the state and RNG are unchanged.
    ///
    /// # Panics
    /// Panics when the observable string is malformed.
    pub fn measure_observable(&mut self, observable: &str) -> Result<bool, MeasurementError> {
        let product =
            PauliProduct::from_compact(observable).unwrap_or_else(|message| panic!("{message}"));
        self.measure_pauli_product(&product)
            .map(|(outcome, _)| outcome == -1)
            .map_err(MeasurementError::new)
    }

    pub(crate) fn measure_pauli_product(
        &mut self,
        product: &PauliProduct,
    ) -> Result<(i8, f64), String> {
        if product.num_qubits <= self.n {
            return self.measure_pauli_product_raw(product);
        }
        let mut grown = self.clone();
        grown.ensure_num_qubits(product.num_qubits);
        let result = grown.measure_pauli_product_raw(product)?;
        *self = grown;
        Ok(result)
    }

    pub(crate) fn measure_pauli_product_raw(
        &mut self,
        product: &PauliProduct,
    ) -> Result<(i8, f64), String> {
        debug_assert!(product.num_qubits <= self.n);
        let mut observable = PackedPauliObservable::from_product(product, self.n);
        if observable.x.is_zero() && observable.z.is_zero() {
            return Ok((if observable.negative { -1 } else { 1 }, 1.0));
        }

        let mut steps = Vec::new();
        let measured = if let Some(pivot) = observable.x.pivot() {
            let other_x = observable
                .x
                .iter_ones()
                .filter(|&qubit| qubit != pivot)
                .collect::<Vec<_>>();
            for qubit in other_x {
                self.cnot(pivot, qubit);
                observable.conjugate_cnot(pivot, qubit);
                steps.push(ObservableBasisStep::Cnot(pivot, qubit));
            }
            let other_z = observable
                .z
                .iter_ones()
                .filter(|&qubit| qubit != pivot)
                .collect::<Vec<_>>();
            for qubit in other_z {
                self.cz(pivot, qubit);
                observable.conjugate_cz(pivot, qubit);
                steps.push(ObservableBasisStep::Cz(pivot, qubit));
            }
            if observable.z.get(pivot) {
                self.apply_t_power(pivot, -2);
                observable.conjugate_s(pivot, true);
                steps.push(ObservableBasisStep::Sdg(pivot));
            }
            debug_assert_eq!(observable.x.iter_ones().collect::<Vec<_>>(), vec![pivot]);
            debug_assert!(observable.z.is_zero());
            self.measure_x_raw(pivot)
        } else {
            let pivot = observable
                .z
                .pivot()
                .expect("nonidentity observable has a Z");
            let other_z = observable
                .z
                .iter_ones()
                .filter(|&qubit| qubit != pivot)
                .collect::<Vec<_>>();
            for qubit in other_z {
                self.cnot(qubit, pivot);
                observable.conjugate_cnot(qubit, pivot);
                steps.push(ObservableBasisStep::Cnot(qubit, pivot));
            }
            debug_assert_eq!(observable.z.iter_ones().collect::<Vec<_>>(), vec![pivot]);
            Ok(self.measure_z_raw(pivot))
        };

        match measured {
            Ok((outcome, probability)) => {
                self.undo_observable_basis(&steps);
                Ok((
                    if observable.negative {
                        -outcome
                    } else {
                        outcome
                    },
                    probability,
                ))
            }
            Err(message) => {
                self.undo_observable_basis(&steps);
                Err(format!(
                    "Pauli observable measurement is incompatible: {message}"
                ))
            }
        }
    }

    fn undo_observable_basis(&mut self, steps: &[ObservableBasisStep]) {
        for step in steps.iter().rev() {
            match *step {
                ObservableBasisStep::Cnot(control, target) => self.cnot(control, target),
                ObservableBasisStep::Cz(left, right) => self.cz(left, right),
                ObservableBasisStep::Sdg(qubit) => self.s(qubit),
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PackedPauliObservable {
    x: BitVec,
    z: BitVec,
    negative: bool,
}

impl PackedPauliObservable {
    fn from_product(product: &PauliProduct, n: usize) -> Self {
        let mut x = BitVec::zero(n);
        let mut z = BitVec::zero(n);
        for &(qubit, pauli) in &product.factors {
            match pauli {
                Pauli::X => x.set(qubit, true),
                Pauli::Y => {
                    x.set(qubit, true);
                    z.set(qubit, true);
                }
                Pauli::Z => z.set(qubit, true),
            }
        }
        Self {
            x,
            z,
            negative: product.negative,
        }
    }

    fn conjugate_cnot(&mut self, control: usize, target: usize) {
        let xc = self.x.get(control);
        let xt = self.x.get(target);
        let zc = self.z.get(control);
        let zt = self.z.get(target);
        self.negative ^= xc && zt && (xt ^ zc ^ true);
        if xc {
            self.x.toggle(target);
        }
        if zt {
            self.z.toggle(control);
        }
    }

    fn conjugate_s(&mut self, qubit: usize, dagger: bool) {
        let x = self.x.get(qubit);
        let z = self.z.get(qubit);
        self.negative ^= if dagger { x && !z } else { x && z };
        if x {
            self.z.toggle(qubit);
        }
    }

    fn conjugate_cz(&mut self, left: usize, right: usize) {
        self.conjugate_s(left, false);
        self.conjugate_s(right, false);
        self.conjugate_cnot(left, right);
        self.conjugate_s(right, true);
        self.conjugate_cnot(left, right);
    }
}

#[derive(Clone, Copy, Debug)]
enum ObservableBasisStep {
    Cnot(usize, usize),
    Cz(usize, usize),
    Sdg(usize),
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        Simulator, StateVectorEndian,
        circuit::PauliProduct,
        simulator::test::{dense_observable_expectation, equivalent, mask, project_observable},
    };

    #[test]
    fn pauli_observable_measurements_match_dense_projection() {
        let mut state = Simulator::from_product(b"+++", 31);
        state.t(0);
        state.t_dag(1);
        state.cnot(0, 2);
        state.cz(1, 2);
        let before = state.state_vector(StateVectorEndian::Little);

        for code in 0..64 {
            let mut remaining = code;
            let body = (0..3)
                .map(|_| {
                    let symbol = ['_', 'X', 'Y', 'Z'][remaining & 3];
                    remaining >>= 2;
                    symbol
                })
                .collect::<String>();
            for prefix in ["", "-"] {
                let source = format!("{prefix}{body}");
                let product = PauliProduct::from_compact(&source).unwrap();
                let mut measured = state.clone();
                match measured.measure_pauli_product_raw(&product) {
                    Ok((outcome, probability)) => {
                        let (expected_probability, expected) =
                            project_observable(&before, &product, outcome == -1)
                                .expect("sampled branch has nonzero probability");
                        assert!((probability - expected_probability).abs() < 1.0e-5);
                        assert!(equivalent(
                            &measured.state_vector(StateVectorEndian::Little),
                            &expected
                        ));
                    }
                    Err(message) => {
                        assert!(message.contains("incompatible"));
                        assert!(equivalent(
                            &measured.state_vector(StateVectorEndian::Little),
                            &before
                        ));
                        let mut control = state.clone();
                        assert_eq!(measured.measure_z_raw(0), control.measure_z_raw(0));
                    }
                }
            }
        }
    }

    #[test]
    fn observable_expectations_match_dense_evolution() {
        let mut state = Simulator::from_product(b"+++", 35);
        state.t(0);
        state.t_dag(1);
        state.cnot(0, 2);
        state.cz(1, 2);
        let dense = state.state_vector(StateVectorEndian::Little);

        for code in 0..64 {
            let mut remaining = code;
            let body = (0..3)
                .map(|_| {
                    let symbol = ['_', 'X', 'Y', 'Z'][remaining & 3];
                    remaining >>= 2;
                    symbol
                })
                .collect::<String>();
            for prefix in ["", "-"] {
                let source = format!("{prefix}{body}");
                let product = PauliProduct::from_compact(&source).unwrap();
                let expected = dense_observable_expectation(&dense, &product);
                let actual = state.peek_observable_expectation(&source);
                assert!(
                    (actual - expected).abs() < 2.0e-5,
                    "{source}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn randomized_observable_expectations_match_dense_vectors() {
        let mut rng = ChaCha8Rng::seed_from_u64(40);
        for _ in 0..80 {
            let n = rng.random_range(1..7);
            let mut state = Simulator::from_product(&vec![b'+'; n], rng.random());
            for _ in 0..30 {
                match rng.random_range(0..3) {
                    0 => state.t(rng.random_range(0..n)),
                    1 => state.t_dag(rng.random_range(0..n)),
                    _ if n > 1 => {
                        let control = rng.random_range(0..n);
                        let mut target = rng.random_range(0..n - 1);
                        if target >= control {
                            target += 1;
                        }
                        state.cnot(control, target);
                    }
                    _ => {}
                }
            }
            let dense = state.state_vector(StateVectorEndian::Little);
            for _ in 0..20 {
                let mut source = (0..n)
                    .map(|_| ['_', 'X', 'Y', 'Z'][rng.random_range(0..4)])
                    .collect::<String>();
                if rng.random_bool(0.5) {
                    source.insert(0, '-');
                }
                let product = PauliProduct::from_compact(&source).unwrap();
                let expected = dense_observable_expectation(&dense, &product);
                let actual = state.peek_observable_expectation(&source);
                assert!((actual - expected).abs() < 3.0e-5, "{source}");
            }
        }
    }

    #[test]
    fn observable_expectations_cover_non_clifford_and_incompatible_cases() {
        let mut t_state = Simulator::from_product(b"+", 36);
        t_state.t(0);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((t_state.peek_observable_expectation("X") - expected).abs() < 1.0e-12);
        assert!((t_state.peek_observable_expectation("Y") - expected).abs() < 1.0e-12);
        assert_eq!(t_state.peek_observable_expectation("Z"), 0.0);

        let mut incompatible = Simulator::from_product(b"+++++", 37);
        incompatible.add_phase_gadget(mask(5, &[0]), 2);
        incompatible.ccz(0, 1, 2);
        incompatible.ccz(0, 3, 4);
        assert!(incompatible.classify_x_measurement(0).is_err());
        let product = PauliProduct::from_compact("X____").unwrap();
        let dense = incompatible.state_vector(StateVectorEndian::Little);
        let expected = dense_observable_expectation(&dense, &product);
        assert!((incompatible.peek_observable_expectation("X____") - expected).abs() < 2.0e-5);
    }

    #[test]
    fn observable_expectations_are_read_only_and_support_wide_indices() {
        let state = Simulator::with_seed(38);
        let mut control = state.clone();
        assert_eq!(state.peek_observable_expectation("-I"), -1.0);
        assert_eq!(state.peek_observable_expectation("Z_I"), 1.0);
        assert_eq!(state.peek_observable_expectation("X_I"), 0.0);
        assert_eq!(state.num_qubits(), 0);
        assert_eq!(state.clone().measure(0), control.measure(0));

        let mut wide = Simulator::with_seed(39);
        for qubit in 0..130 {
            wide.reset_x(qubit);
        }
        wide.t(129);
        let observable = format!("{}X", "_".repeat(129));
        assert!(
            (wide.peek_observable_expectation(&observable) - std::f64::consts::FRAC_1_SQRT_2).abs()
                < 1.0e-12
        );
        assert_eq!(wide.num_qubits(), 130);

        for invalid in ["", "+", "-", "Xi"] {
            assert!(
                std::panic::catch_unwind(|| state.peek_observable_expectation(invalid)).is_err()
            );
        }
        assert_eq!(state.num_qubits(), 0);
    }

    #[test]
    fn compact_observables_grow_and_validate() {
        let mut state = Simulator::with_seed(32);
        assert!(!state.measure_observable("+Z_I").unwrap());
        assert_eq!(state.num_qubits(), 3);
        assert!(state.measure_observable("-I").unwrap());
        for invalid_source in ["", "+", "-", "Xi"] {
            assert!(
                std::panic::catch_unwind(|| {
                    let mut invalid = Simulator::new();
                    let _ = invalid.measure_observable(invalid_source);
                })
                .is_err()
            );
        }
    }

    #[test]
    fn incompatible_observable_rolls_back_basis_and_rng() {
        let mut state = Simulator::from_product(b"+++", 33);
        state.add_phase_gadget(mask(3, &[0]), 1);
        state.ccz(0, 1, 2);
        state.cnot(0, 1);
        let before = state.state_vector(StateVectorEndian::Little);
        let mut rng_control = state.clone();

        let error = state.measure_observable("XX_I").unwrap_err();
        assert!(error.to_string().contains("incompatible"));
        assert_eq!(state.num_qubits(), 3);
        assert!(equivalent(
            &state.state_vector(StateVectorEndian::Little),
            &before
        ));
        assert_eq!(state.measure_z_raw(2), rng_control.measure_z_raw(2));
    }
}
