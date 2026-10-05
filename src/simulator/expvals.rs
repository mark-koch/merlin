use num_complex::Complex64;

use crate::{
    Simulator,
    bits::BitVec,
    circuit::{Pauli, PauliProduct},
};

impl Simulator {
    /// Returns the expectation value of a Hermitian Pauli-string observable.
    ///
    /// The observable uses compact positional notation, for example `"-X_YZ"`.
    /// `_` and `I` are identity factors. Unlike stabilizer-state expectations,
    /// the result can be any representable value in `[-1, 1]`.
    ///
    /// This is a non-physical query: it does not mutate the state, grow the
    /// simulator, or consume randomness. Untracked qubits are treated as being
    /// in the |0> state.
    ///
    /// # Panics
    /// Panics when the observable string is malformed.
    pub fn peek_observable_expectation(&self, observable: &str) -> f64 {
        let product =
            PauliProduct::from_compact(observable).unwrap_or_else(|message| panic!("{message}"));
        self.pauli_product_expectation(&product)
    }

    pub(crate) fn pauli_product_expectation(&self, product: &PauliProduct) -> f64 {
        let mut x = BitVec::zero(self.n);
        let mut z = BitVec::zero(self.n);
        let mut y_count = 0_u8;
        for &(qubit, pauli) in &product.factors {
            if qubit >= self.n {
                if matches!(pauli, Pauli::X | Pauli::Y) {
                    return 0.0;
                }
                continue;
            }
            match pauli {
                Pauli::X => x.set(qubit, true),
                Pauli::Y => {
                    x.set(qubit, true);
                    z.set(qubit, true);
                    y_count = y_count.wrapping_add(1) & 3;
                }
                Pauli::Z => z.set(qubit, true),
            }
        }

        let mut z_coordinates = BitVec::zero(self.rank);
        let mut z_constant = false;
        for qubit in z.iter_ones() {
            z_coordinates.xor_assign(&self.basis_rows[qubit]);
            z_constant ^= self.x0.get(qubit);
        }
        let sign_constant = product.negative ^ z_constant;

        if x.is_zero() {
            return if z_coordinates.is_zero() {
                if sign_constant { -1.0 } else { 1.0 }
            } else {
                0.0
            };
        }

        let Some(direction) = self.to_basis_coordinates(&x) else {
            return 0.0;
        };
        let (derivative_constant, derivative_linear, matrix) =
            self.directional_derivative(&direction);
        let linear = derivative_linear
            .into_iter()
            .enumerate()
            .map(|(index, coefficient)| {
                (-i16::from(coefficient) + 2 * i16::from(z_coordinates.get(index))).rem_euclid(4)
                    as u8
            })
            .collect::<Vec<_>>();
        let constant = (-i16::from(derivative_constant)
            + 2 * i16::from(y_count)
            + 4 * i16::from(sign_constant))
        .rem_euclid(8) as u8;
        let value = normalized_quadratic_gauss_sum(constant, linear, matrix);
        debug_assert!(value.im.abs() < 1.0e-9, "Hermitian expectation was {value}");
        let expectation = value.re.clamp(-1.0, 1.0);
        if expectation == 0.0 { 0.0 } else { expectation }
    }
}

/// Evaluates `2^-r sum_u omega^(constant + 2*l.u + 4*sum_{i<j} M_ij u_i u_j)`.
fn normalized_quadratic_gauss_sum(
    constant: u8,
    mut linear: Vec<u8>,
    mut matrix: Vec<BitVec>,
) -> Complex64 {
    let rank = linear.len();
    debug_assert_eq!(matrix.len(), rank);
    let mut active = vec![true; rank];
    let mut constant_term = 0_u8;
    let mut multiplier = omega_power(i16::from(constant));

    loop {
        for variable in 0..rank {
            if active[variable] && linear[variable] == 0 && matrix[variable].is_zero() {
                active[variable] = false;
            }
        }

        let Some(first) =
            (0..rank).find(|&variable| active[variable] && !matrix[variable].is_zero())
        else {
            for variable in 0..rank {
                if active[variable] {
                    multiplier *=
                        (Complex64::new(1.0, 0.0) + i_power(i16::from(linear[variable]))) * 0.5;
                }
            }
            return multiplier * i_power(i16::from(constant_term));
        };

        let second = matrix[first]
            .pivot()
            .expect("a coupled variable has a neighbor");
        let mut remaining_neighbors = matrix[first].clone();
        remaining_neighbors.set(second, false);
        if !remaining_neighbors.is_zero() {
            let second_linear = linear[second];
            linear[second] = 0;
            let second_neighbors = matrix[second]
                .iter_ones()
                .filter(|&neighbor| neighbor != first)
                .collect::<Vec<_>>();
            for neighbor in second_neighbors {
                for other in remaining_neighbors.iter_ones() {
                    toggle_quadratic_edge(&mut matrix, &mut linear, other, neighbor);
                }
            }
            for other in remaining_neighbors.iter_ones() {
                debug_assert!(matrix[first].get(other));
                toggle_quadratic_edge(&mut matrix, &mut linear, first, other);
            }
            if second_linear != 0 {
                let mut substitution = remaining_neighbors.clone();
                substitution.set(second, true);
                add_affine_to_quadratic(
                    &substitution,
                    false,
                    i16::from(second_linear),
                    &mut constant_term,
                    &mut linear,
                    &mut matrix,
                );
            }
        }

        debug_assert_eq!(matrix[first].iter_ones().collect::<Vec<_>>(), vec![second]);
        let first_linear = linear[first];
        let second_linear = linear[second];
        let mut dependent = matrix[second].clone();
        dependent.set(first, false);
        let sums = [false, true].map(|delta| {
            Complex64::new(1.0, 0.0)
                + i_power(i16::from(first_linear))
                + i_power(i16::from(second_linear) + 2 * i16::from(delta))
                    * (Complex64::new(1.0, 0.0) + i_power(i16::from(first_linear) + 2))
        });

        linear[first] = 0;
        linear[second] = 0;
        clear_quadratic_variable(&mut matrix, first);
        clear_quadratic_variable(&mut matrix, second);
        active[first] = false;
        active[second] = false;

        if dependent.is_zero() || sums[0] == sums[1] {
            if sums[0] == Complex64::new(0.0, 0.0) {
                return Complex64::new(0.0, 0.0);
            }
            multiplier *= sums[0] * 0.25;
            continue;
        }

        let nonzero = sums.map(|value| value != Complex64::new(0.0, 0.0));
        if nonzero[0] && nonzero[1] {
            let phase = (0_i16..4)
                .find(|&power| sums[0] * i_power(power) == sums[1])
                .expect("quadratic pair ratio is a power of i");
            multiplier *= sums[0] * 0.25;
            add_affine_to_quadratic(
                &dependent,
                false,
                phase,
                &mut constant_term,
                &mut linear,
                &mut matrix,
            );
            continue;
        }

        let required = !nonzero[0];
        multiplier *= sums[usize::from(required)] * 0.125;
        let pivot = dependent.pivot().expect("a dependent parity has a pivot");
        let mut remainder = dependent.clone();
        remainder.set(pivot, false);
        let pivot_linear = linear[pivot];
        linear[pivot] = 0;
        let pivot_neighbors = matrix[pivot].iter_ones().collect::<Vec<_>>();
        clear_quadratic_variable(&mut matrix, pivot);
        active[pivot] = false;
        if pivot_linear != 0 {
            add_affine_to_quadratic(
                &remainder,
                required,
                i16::from(pivot_linear),
                &mut constant_term,
                &mut linear,
                &mut matrix,
            );
        }
        for neighbor in pivot_neighbors {
            add_mod_four(&mut linear[neighbor], 2 * i16::from(required));
            for other in remainder.iter_ones() {
                toggle_quadratic_edge(&mut matrix, &mut linear, other, neighbor);
            }
        }
    }
}

fn i_power(power: i16) -> Complex64 {
    match power.rem_euclid(4) {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!(),
    }
}

fn omega_power(power: i16) -> Complex64 {
    match power.rem_euclid(8) {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
        2 => Complex64::new(0.0, 1.0),
        3 => Complex64::new(
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
        4 => Complex64::new(-1.0, 0.0),
        5 => Complex64::new(
            -std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
        ),
        6 => Complex64::new(0.0, -1.0),
        7 => Complex64::new(
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
        ),
        _ => unreachable!(),
    }
}

fn add_mod_four(value: &mut u8, amount: i16) {
    *value = (i16::from(*value) + amount).rem_euclid(4) as u8;
}

fn toggle_quadratic_edge(matrix: &mut [BitVec], linear: &mut [u8], left: usize, right: usize) {
    if left == right {
        add_mod_four(&mut linear[left], 2);
    } else {
        matrix[left].toggle(right);
        matrix[right].toggle(left);
    }
}

fn add_affine_to_quadratic(
    mask: &BitVec,
    constant: bool,
    amount: i16,
    constant_term: &mut u8,
    linear: &mut [u8],
    matrix: &mut [BitVec],
) {
    add_mod_four(constant_term, amount * i16::from(constant));
    let signed_amount = amount * if constant { -1 } else { 1 };
    let variables = mask.iter_ones().collect::<Vec<_>>();
    for &variable in &variables {
        add_mod_four(&mut linear[variable], signed_amount);
    }
    if signed_amount & 1 != 0 {
        for (position, &left) in variables.iter().enumerate() {
            for &right in &variables[position + 1..] {
                toggle_quadratic_edge(matrix, linear, left, right);
            }
        }
    }
}

fn clear_quadratic_variable(matrix: &mut [BitVec], variable: usize) {
    let rank = matrix.len();
    let neighbors = matrix[variable].iter_ones().collect::<Vec<_>>();
    for neighbor in neighbors {
        matrix[neighbor].set(variable, false);
    }
    matrix[variable] = BitVec::zero(rank);
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::simulator::expvals::{normalized_quadratic_gauss_sum, omega_power};

    #[test]
    fn quadratic_gauss_elimination_matches_enumeration() {
        let mut rng = ChaCha8Rng::seed_from_u64(34);
        for rank in 0..9 {
            for _ in 0..200 {
                let constant = rng.random_range(0..8);
                let linear = (0..rank)
                    .map(|_| rng.random_range(0..4))
                    .collect::<Vec<_>>();
                let mut matrix = (0..rank)
                    .map(|_| crate::bits::BitVec::zero(rank))
                    .collect::<Vec<_>>();
                for left in 0..rank {
                    for right in left + 1..rank {
                        if rng.random_bool(0.5) {
                            matrix[left].set(right, true);
                            matrix[right].set(left, true);
                        }
                    }
                }

                let actual =
                    normalized_quadratic_gauss_sum(constant, linear.clone(), matrix.clone());
                let mut expected = num_complex::Complex64::new(0.0, 0.0);
                for assignment in 0..(1_usize << rank) {
                    let mut exponent = i16::from(constant);
                    for (variable, &coefficient) in linear.iter().enumerate() {
                        if assignment >> variable & 1 != 0 {
                            exponent += 2 * i16::from(coefficient);
                        }
                    }
                    for (left, row) in matrix.iter().enumerate() {
                        if assignment >> left & 1 != 0 {
                            for right in row.iter_ones().filter(|&right| right > left) {
                                if assignment >> right & 1 != 0 {
                                    exponent += 4;
                                }
                            }
                        }
                    }
                    expected += omega_power(exponent);
                }
                expected /= (1_usize << rank) as f64;
                assert!(
                    (actual - expected).norm() < 1.0e-10,
                    "rank={rank}, constant={constant}, linear={linear:?}, matrix={matrix:?}, actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn quadratic_gauss_elimination_crosses_word_boundaries() {
        let rank = 130;
        let linear = vec![0; rank];
        let mut matrix = (0..rank)
            .map(|_| crate::bits::BitVec::zero(rank))
            .collect::<Vec<_>>();
        for left in (0..rank).step_by(2) {
            matrix[left].set(left + 1, true);
            matrix[left + 1].set(left, true);
        }
        let actual = normalized_quadratic_gauss_sum(0, linear, matrix).re;
        let expected = 0.5_f64.powi(65);
        assert!((actual - expected).abs() <= expected * 1.0e-12);
    }
}
