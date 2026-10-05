use rand::Rng;

use crate::{Simulator, circuit::Pauli};

impl Simulator {
    /// Applies a single-qubit Pauli with the given probability distribution.
    ///
    /// # Panics
    /// Panics unless `probabilities` describes a valid distribution.
    pub fn pauli_channel1(&mut self, qubit: usize, probabilities: &[f64; 3]) {
        if let Some(index) = self.choose_channel(probabilities) {
            self.apply_pauli_code(qubit, index + 1);
        }
    }

    /// Applies a two-qubit Pauli with the given probability distribution.
    ///
    /// # Panics
    /// Panics unless `probabilities` describes a valid distribution.
    pub fn pauli_channel2(&mut self, qubit1: usize, qubit2: usize, probabilities: &[f64; 15]) {
        if let Some(index) = self.choose_channel(probabilities) {
            let code = index + 1;
            self.apply_pauli_code(qubit1, code >> 2);
            self.apply_pauli_code(qubit2, code & 3);
        }
    }

    /// Applies an X error with the given probability.
    ///
    /// # Panics
    /// Panics unless the probability is finite and in `[0, 1]`.
    pub fn x_error(&mut self, qubit: usize, probability: f64) {
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit + 1);
        if self.sample_event(probability) {
            self.x(qubit);
        }
    }

    /// Applies a Y error with the given probability.
    ///
    /// # Panics
    /// Panics unless the probability is finite and in `[0, 1]`.
    pub fn y_error(&mut self, qubit: usize, probability: f64) {
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit + 1);
        if self.sample_event(probability) {
            self.y(qubit);
        }
    }

    /// Applies a Z error with the given probability.
    ///
    /// # Panics
    /// Panics unless the probability is finite and in `[0, 1]`.
    pub fn z_error(&mut self, qubit: usize, probability: f64) {
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit + 1);
        if self.sample_event(probability) {
            self.z(qubit);
        }
    }

    /// Applies a given Pauli error with the given probability.
    ///
    /// # Panics
    /// Panics unless the probability is finite and in `[0, 1]`.
    pub fn pauli_error(&mut self, qubit: usize, pauli: &Pauli, probability: f64) {
        if self.sample_event(probability) {
            self.apply_pauli_code(qubit, pauli.code());
        }
    }

    /// Applies single-qubit depolarizing noise.
    ///
    /// # Panics
    /// Panics unless the probability is finite and in `[0, 1]`.
    pub fn depolarize1(&mut self, qubit: usize, probability: f64) {
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit + 1);
        if self.sample_event(probability) {
            let code = self.random_pauli_code(1);
            self.apply_pauli_code(qubit, code);
        }
    }

    /// Applies two-qubit depolarizing noise.
    ///
    /// # Panics
    /// Panics for equal endpoints or an invalid probability.
    pub fn depolarize2(&mut self, qubit1: usize, qubit2: usize, probability: f64) {
        assert_ne!(qubit1, qubit2, "depolarize2 endpoints must be distinct");
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit1.max(qubit2) + 1);
        if self.sample_event(probability) {
            let code = self.random_pauli_code(2);
            self.apply_pauli_code(qubit1, code >> 2);
            self.apply_pauli_code(qubit2, code & 3);
        }
    }

    /// Applies three-qubit depolarizing noise.
    ///
    /// # Panics
    /// Panics unless the endpoints are distinct and the probability is valid.
    pub fn depolarize3(&mut self, qubit1: usize, qubit2: usize, qubit3: usize, probability: f64) {
        assert!(
            qubit1 != qubit2 && qubit1 != qubit3 && qubit2 != qubit3,
            "depolarize3 endpoints must be distinct"
        );
        Self::checked_probability(probability);
        self.ensure_num_qubits(qubit1.max(qubit2).max(qubit3) + 1);
        if self.sample_event(probability) {
            let code = self.random_pauli_code(3);
            self.apply_pauli_code(qubit1, code >> 4);
            self.apply_pauli_code(qubit2, (code >> 2) & 3);
            self.apply_pauli_code(qubit3, code & 3);
        }
    }

    pub(crate) fn sample_event(&mut self, probability: f64) -> bool {
        if probability == 0.0 {
            false
        } else if probability == 1.0 {
            true
        } else {
            self.rng.random_bool(probability)
        }
    }

    pub(crate) fn random_pauli_code(&mut self, num_qubits: usize) -> u8 {
        debug_assert!((1..=3).contains(&num_qubits));
        self.rng.random_range(1_u8..(1_u8 << (2 * num_qubits)))
    }

    fn checked_probability(probability: f64) {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "probability must be finite and between 0 and 1"
        );
    }

    pub(crate) fn apply_pauli_code(&mut self, qubit: usize, code: u8) {
        match code {
            0 => {}
            1 => self.x(qubit),
            2 => self.y(qubit),
            3 => self.z(qubit),
            _ => unreachable!("a Pauli code has two bits"),
        }
    }

    fn choose_channel<const N: usize>(&mut self, probabilities: &[f64; N]) -> Option<u8> {
        let mut sole = None;
        let mut nonzero = 0;
        for (index, &probability) in probabilities.iter().enumerate() {
            if probability > 0.0 {
                sole = Some(index as u8);
                nonzero += 1;
            }
        }
        if nonzero == 0 {
            return None;
        }
        if nonzero == 1 && sole.is_some_and(|index| probabilities[index as usize] == 1.0) {
            return sole;
        }
        let draw = self.rng.random::<f64>();
        let mut cumulative = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                return Some(index as u8);
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{Simulator, StateVectorEndian, simulator::test::equivalent};

    #[test]
    fn depolarize3_decodes_every_non_identity_pauli_product() {
        let mut base = Simulator::from_product(b"+++", 90);
        base.t(0);
        base.s(1);
        base.cnot(0, 2);
        base.ccz(0, 1, 2);

        let mut actual_rng = ChaCha8Rng::seed_from_u64(91);
        let mut expected_rng = actual_rng.clone();
        let mut seen = [false; 64];
        for _ in 0..2_000 {
            let mut actual = base.clone();
            actual.rng = actual_rng.clone();
            actual.depolarize3(0, 1, 2, 1.0);
            actual_rng = actual.rng.clone();

            let code = expected_rng.random_range(1_u8..64);
            seen[usize::from(code)] = true;
            let mut expected = base.clone();
            expected.apply_pauli_code(0, code >> 4);
            expected.apply_pauli_code(1, (code >> 2) & 3);
            expected.apply_pauli_code(2, code & 3);
            assert!(equivalent(
                &actual.state_vector(StateVectorEndian::Little),
                &expected.state_vector(StateVectorEndian::Little),
            ));
        }
        assert!(seen[1..].iter().all(|value| *value));
    }
}
