use std::f64::consts::FRAC_PI_4;

use rand::Rng;

use crate::{MeasurementError, Simulator, bits::BitVec};

#[derive(Clone, Debug)]
pub(crate) enum XMeasurementPlan {
    Deterministic {
        minus: bool,
    },
    OutsideSpan,
    SupportCut {
        mask: BitVec,
        constant: bool,
    },
    Quadratic {
        linear: BitVec,
        matrix: Vec<BitVec>,
        constant: bool,
    },
    Odd {
        mask: BitVec,
        k: u8,
        p_plus: f64,
    },
}

impl XMeasurementPlan {
    /// The probability of getting the +1 outcome.
    pub(crate) fn plus_probability(&self) -> f64 {
        match self {
            Self::Deterministic { minus } => f64::from(!minus),
            Self::OutsideSpan | Self::SupportCut { .. } | Self::Quadratic { .. } => 0.5,
            Self::Odd { p_plus, .. } => *p_plus,
        }
    }
}

impl Simulator {
    /// Measures Z, returning `true` for the -1 eigenvalue.
    pub fn measure(&mut self, qubit: usize) -> bool {
        self.ensure_num_qubits(qubit + 1);
        self.measure_z_raw(qubit).0 == -1
    }

    /// Alias of [`Self::measure`].
    pub fn measure_z(&mut self, qubit: usize) -> bool {
        self.measure(qubit)
    }

    /// Measures X, returning `true` for the -1 eigenvalue.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] when the X projection is incompatible.
    pub fn measure_x(&mut self, qubit: usize) -> Result<bool, MeasurementError> {
        self.ensure_num_qubits(qubit + 1);
        self.measure_x_raw(qubit)
            .map(|(outcome, _)| outcome == -1)
            .map_err(MeasurementError::new)
    }

    /// Returns the probability that Z measurement reports `true`.
    pub fn peek_probability(&self, qubit: usize) -> f64 {
        if qubit >= self.n {
            0.0
        } else if self.basis_rows[qubit].is_zero() {
            f64::from(self.x0.get(qubit))
        } else {
            0.5
        }
    }

    /// Alias of [`Self::peek_probability`].
    pub fn peek_z_probability(&self, qubit: usize) -> f64 {
        self.peek_probability(qubit)
    }

    /// Returns the probability that X measurement reports `true`.
    ///
    /// # Errors
    /// Returns [`MeasurementError`] when the X projection is incompatible.
    pub fn peek_x_probability(&self, qubit: usize) -> Result<f64, MeasurementError> {
        if qubit >= self.n {
            Ok(0.5)
        } else {
            self.classify_x_measurement(qubit)
                .map(|plan| 1.0 - plan.plus_probability())
                .map_err(MeasurementError::new)
        }
    }

    #[cfg(test)]
    pub(crate) fn measure_z_with_outcome(
        &mut self,
        qubit: usize,
        minus: bool,
    ) -> Result<f64, String> {
        let normal = self.basis_rows[qubit].clone();
        if normal.is_zero() {
            return if minus == self.x0.get(qubit) {
                Ok(1.0)
            } else {
                Err("requested deterministic branch has zero probability".to_owned())
            };
        }
        self.restrict_affine(&normal, minus ^ self.x0.get(qubit));
        Ok(0.5)
    }

    #[cfg(test)]
    pub(crate) fn measure_x_with_outcome(
        &mut self,
        qubit: usize,
        minus: bool,
    ) -> Result<f64, String> {
        let plan = self.classify_x_measurement(qubit)?;
        let p_plus = plan.plus_probability();
        let probability = if minus { 1.0 - p_plus } else { p_plus };
        if probability == 0.0 {
            return Err("requested deterministic branch has zero probability".to_owned());
        }
        self.apply_x_projector(qubit, plan, minus)?;
        Ok(probability)
    }

    pub(crate) fn measure_z_raw(&mut self, qubit: usize) -> (i8, f64) {
        let normal = self.basis_rows[qubit].clone();
        let p_plus = if normal.is_zero() {
            f64::from(!self.x0.get(qubit))
        } else {
            0.5
        };
        let minus = self.sample_outcome(p_plus);
        if !normal.is_zero() {
            self.restrict_affine(&normal, minus ^ self.x0.get(qubit));
        }
        (
            if minus { -1 } else { 1 },
            if minus { 1.0 - p_plus } else { p_plus },
        )
    }

    pub(crate) fn measure_x_raw(&mut self, qubit: usize) -> Result<(i8, f64), String> {
        let plan = self.classify_x_measurement(qubit)?;
        let p_plus = plan.plus_probability();
        let minus = self.sample_outcome(p_plus);
        self.apply_x_projector(qubit, plan, minus)?;
        Ok((
            if minus { -1 } else { 1 },
            if minus { 1.0 - p_plus } else { p_plus },
        ))
    }

    /// Classifies an X measurement or returns `Err` if the measurement is not
    /// compatible.
    pub(crate) fn classify_x_measurement(&self, qubit: usize) -> Result<XMeasurementPlan, String> {
        let Some(direction) = self.unit_basis_coordinates(qubit) else {
            return Ok(XMeasurementPlan::OutsideSpan);
        };
        let (k, linear, matrix) = self.directional_derivative(&direction);
        if k % 2 == 0 {
            if linear.iter().any(|coefficient| coefficient & 1 != 0) {
                return Err(format!(
                    "X measurement on qubit {qubit} is incompatible: mixed phase parity"
                ));
            }
            let mut mask = BitVec::zero(self.rank);
            for (index, coefficient) in linear.into_iter().enumerate() {
                mask.set(index, coefficient == 2);
            }
            if matches!(k, 0 | 4) {
                if matrix.iter().any(|row| !row.is_zero()) {
                    return Err(format!(
                        "X measurement on qubit {qubit} is incompatible: non-affine zero set"
                    ));
                }
                let constant = k == 4;
                return if mask.is_zero() {
                    Ok(XMeasurementPlan::Deterministic { minus: constant })
                } else {
                    Ok(XMeasurementPlan::SupportCut { mask, constant })
                };
            }
            if rank_two_decomposition(&matrix).is_none() {
                return Err(format!(
                    "X measurement on qubit {qubit} is incompatible: quadratic rank exceeds two"
                ));
            }
            return Ok(XMeasurementPlan::Quadratic {
                linear: mask,
                matrix,
                constant: k == 6,
            });
        }

        let target = 4_u8.wrapping_sub(k & 3) & 3;
        let mut mask = BitVec::zero(self.rank);
        for (index, coefficient) in linear.into_iter().enumerate() {
            if coefficient == target {
                mask.set(index, true);
            } else if coefficient != 0 {
                return Err(format!(
                    "X measurement on qubit {qubit} is incompatible: odd phase is not two-valued"
                ));
            }
        }
        for (row, matrix_row) in matrix.iter().enumerate() {
            for column in 0..self.rank {
                if row != column && matrix_row.get(column) != (mask.get(row) && mask.get(column)) {
                    return Err(format!(
                        "X measurement on qubit {qubit} is incompatible: odd-phase cross terms are not rank one"
                    ));
                }
            }
        }
        debug_assert!(mask.dot(&direction));
        let p_plus = (1.0 + (f64::from(k) * FRAC_PI_4).cos()) / 2.0;
        Ok(XMeasurementPlan::Odd { mask, k, p_plus })
    }

    fn apply_x_projector(
        &mut self,
        qubit: usize,
        plan: XMeasurementPlan,
        minus: bool,
    ) -> Result<(), String> {
        match plan {
            XMeasurementPlan::Deterministic { minus: expected } => {
                if minus != expected {
                    return Err(
                        "requested deterministic measurement branch has zero probability"
                            .to_owned(),
                    );
                }
            }
            XMeasurementPlan::OutsideSpan => {
                self.add_affine_coordinate(qubit, minus);
            }
            XMeasurementPlan::SupportCut { mask, constant } => {
                self.restrict_affine(&mask, minus ^ constant);
            }
            XMeasurementPlan::Quadratic {
                linear,
                matrix,
                constant,
            } => {
                let sigma = if minus ^ constant { 1 } else { -1 };
                let (alpha, beta) = rank_two_decomposition(&matrix)
                    .expect("compatible quadratic has rank at most two");
                if alpha.is_zero() && beta.is_zero() {
                    self.add_phase_gadget(linear.clone(), 2 * sigma);
                } else {
                    let mut overlap = alpha.clone();
                    overlap.and_assign(&beta);
                    let mut l = linear.clone();
                    l.xor_assign(&overlap);
                    self.add_phase_gadget(l.clone(), sigma);
                    self.add_phase_gadget(l.xor(&alpha), sigma);
                    self.add_phase_gadget(l.xor(&beta), sigma);
                    let mut all = l.xor(&alpha);
                    all.xor_assign(&beta);
                    self.add_phase_gadget(all, -sigma);
                }
            }
            XMeasurementPlan::Odd { mask, k, .. } => {
                self.add_phase_gadget(mask, 4 * i16::from(minus) - i16::from(k));
            }
        }
        Ok(())
    }

    /// Computes the discrete derivative of the phase polynomial in the given direction.
    ///
    /// Concretely, we compute a scalar k ∈ Z_8, vector l ∈ Z_4^r and matrix M ∈ Z_2^r
    /// such that:
    ///
    ///  ∆_dir q(v) := q(v ⊕ dir) - q(v)  (mod 8)
    ///              = k + sum_i l_i v_i + sum_(i<j) M_ij v_i v_j  (mod 8)
    ///
    pub(crate) fn directional_derivative(&self, direction: &BitVec) -> (u8, Vec<u8>, Vec<BitVec>) {
        let mut k = 0_u8;
        let mut linear = vec![0_u8; self.rank];
        let mut matrix = (0..self.rank)
            .map(|_| BitVec::zero(self.rank))
            .collect::<Vec<_>>();
        for (mask, &coefficient) in &self.gadgets {
            if !mask.dot(direction) {
                continue;
            }
            k = k.wrapping_add(coefficient) & 7;
            for index in mask.iter_ones() {
                linear[index] = linear[index].wrapping_sub(coefficient) & 3;
                if coefficient & 1 != 0 {
                    matrix[index].xor_assign(mask);
                }
            }
        }
        for (index, row) in matrix.iter_mut().enumerate() {
            row.set(index, false);
        }
        (k, linear, matrix)
    }

    fn sample_outcome(&mut self, p_plus: f64) -> bool {
        if p_plus == 1.0 {
            false
        } else if p_plus == 0.0 {
            true
        } else {
            !self.rng.random_bool(p_plus)
        }
    }
}

fn rank_two_decomposition(matrix: &[BitVec]) -> Option<(BitVec, BitVec)> {
    let rank = matrix.len();
    let Some(p) = matrix.iter().position(|row| !row.is_zero()) else {
        return Some((BitVec::zero(rank), BitVec::zero(rank)));
    };
    let q = matrix[p].pivot().expect("nonzero row has a pivot");
    let alpha = matrix[q].clone();
    let beta = matrix[p].clone();
    let matches = (0..rank).all(|row| {
        (0..rank).all(|column| {
            matrix[row].get(column)
                == ((alpha.get(row) && beta.get(column)) ^ (beta.get(row) && alpha.get(column)))
        })
    });
    matches.then_some((alpha, beta))
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        Simulator, StateVectorEndian,
        simulator::{
            measurements::XMeasurementPlan,
            test::{equivalent, mask, project},
        },
    };

    fn assert_x_branches(state: &Simulator, qubit: usize) {
        let dense = state.state_vector(StateVectorEndian::Little);
        for minus in [false, true] {
            let expected = project(&dense, qubit, true, minus);
            let mut measured = state.clone();
            let actual = measured.measure_x_with_outcome(qubit, minus);
            match (expected, actual) {
                (Some((probability, expected)), Ok(actual_probability)) => {
                    assert!((probability - actual_probability).abs() < 1.0e-5);
                    assert!(equivalent(
                        &measured.state_vector(StateVectorEndian::Little),
                        &expected
                    ));
                }
                (None, Err(_)) => {}
                mismatch => panic!("X projection mismatch: {mismatch:?}"),
            }
        }
    }

    #[test]
    fn affine_restriction_transforms_support_and_gadgets() {
        let mut state = Simulator::from_product(b"++", 2);
        state.t(0);
        state.cnot(0, 1);
        let before = state.state_vector(StateVectorEndian::Little);
        let (probability, expected) = project(&before, 1, false, true).unwrap();
        let actual_probability = state.measure_z_with_outcome(1, true).unwrap();
        assert!((actual_probability - probability).abs() < 1.0e-6);
        assert!(equivalent(
            &state.state_vector(StateVectorEndian::Little),
            &expected
        ));
    }

    #[test]
    fn classifies_every_geometric_and_phase_branch() {
        let outside = Simulator::from_product(b"0", 1);
        assert!(matches!(
            outside.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::OutsideSpan
        ));
        assert_x_branches(&outside, 0);

        let plus = Simulator::from_product(b"+", 1);
        assert!(matches!(
            plus.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::Deterministic { minus: false }
        ));
        assert_x_branches(&plus, 0);

        let mut support_cut = Simulator::from_product(b"++", 1);
        support_cut.add_phase_gadget(mask(2, &[0]), 2);
        support_cut.add_phase_gadget(mask(2, &[1]), 2);
        support_cut.add_phase_gadget(mask(2, &[0, 1]), -2);
        assert!(matches!(
            support_cut.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::SupportCut { .. }
        ));
        assert_x_branches(&support_cut, 0);

        let mut rank_zero = Simulator::from_product(b"+", 1);
        rank_zero.add_phase_gadget(mask(1, &[0]), 2);
        assert!(matches!(
            rank_zero.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::Quadratic { .. }
        ));
        assert_x_branches(&rank_zero, 0);

        let mut rank_two = Simulator::from_product(b"+++", 1);
        rank_two.add_phase_gadget(mask(3, &[0]), 2);
        rank_two.ccz(0, 1, 2);
        assert!(matches!(
            rank_two.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::Quadratic { .. }
        ));
        assert_x_branches(&rank_two, 0);

        let mut odd = Simulator::from_product(b"+", 1);
        odd.add_phase_gadget(mask(1, &[0]), 1);
        assert!(matches!(
            odd.classify_x_measurement(0).unwrap(),
            XMeasurementPlan::Odd { .. }
        ));
        assert_x_branches(&odd, 0);

        let mut odd_bad = Simulator::from_product(b"+++", 1);
        odd_bad.add_phase_gadget(mask(3, &[0]), 1);
        odd_bad.ccz(0, 1, 2);
        assert!(odd_bad.classify_x_measurement(0).is_err());

        let mut rank_four = Simulator::from_product(b"+++++", 1);
        rank_four.add_phase_gadget(mask(5, &[0]), 2);
        rank_four.ccz(0, 1, 2);
        rank_four.ccz(0, 3, 4);
        assert!(rank_four.classify_x_measurement(0).is_err());
    }

    #[test]
    fn random_compatible_x_and_z_projections_match_dense_vectors() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        for _ in 0..120 {
            let n = 1 + rng.random_range(0..5);
            let spec = (0..n)
                .map(|_| if rng.random_bool(0.65) { b'+' } else { b'0' })
                .collect::<Vec<_>>();
            let mut state = Simulator::from_product(&spec, 5);
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
            for qubit in 0..n {
                for minus in [false, true] {
                    let mut measured = state.clone();
                    let expected = project(&dense, qubit, false, minus);
                    let actual = measured.measure_z_with_outcome(qubit, minus);
                    match (expected, actual) {
                        (Some((probability, expected)), Ok(actual_probability)) => {
                            assert!((probability - actual_probability).abs() < 1.0e-5);
                            assert!(equivalent(
                                &measured.state_vector(StateVectorEndian::Little),
                                &expected
                            ));
                        }
                        (None, Err(_)) => {}
                        mismatch => panic!("Z projection mismatch: {mismatch:?}"),
                    }

                    let mut measured = state.clone();
                    let expected = project(&dense, qubit, true, minus);
                    let actual = measured.measure_x_with_outcome(qubit, minus);
                    match (expected, actual) {
                        (Some((probability, expected)), Ok(actual_probability)) => {
                            assert!((probability - actual_probability).abs() < 1.0e-5);
                            assert!(equivalent(
                                &measured.state_vector(StateVectorEndian::Little),
                                &expected
                            ));
                        }
                        (_, Err(message)) if message.contains("incompatible") => {}
                        (None, Err(_)) => {}
                        mismatch => panic!("X projection mismatch: {mismatch:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn packed_affine_coordinates_cross_word_boundaries() {
        let mut state = Simulator::from_product(&[b'+'; 130], 6);
        assert_eq!(state.rank, 130);
        state.t(64);
        state.cnot(64, 129);
        assert!(
            (state.peek_x_probability(64).unwrap() - (2.0 - 2.0_f64.sqrt()) / 4.0).abs() < 1.0e-12
        );
        assert!(state.unit_basis_coordinates(129).is_some());
    }
}
