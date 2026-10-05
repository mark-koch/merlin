mod expvals;
mod gates;
mod measurements;
mod noise;
mod observables;
#[cfg(test)]
mod test;

use std::collections::HashMap;

use num_complex::Complex32;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::StateVectorEndian;
use crate::bits::BitVec;

/// A direct Schrödinger-picture simulator using affine support coordinates and
/// sparse parity-phase gadgets.
///
/// The represented state is `2^(-r/2) sum_u omega^q(u) |x0 xor B u>`.
/// Cloning the simulator copies both the quantum state and the exact RNG state.
///
/// ```
/// use merlin::{Simulator, StateVectorEndian};
///
/// let mut sim = Simulator::with_seed(7);
/// sim.reset_x(0);
/// sim.t(0);
/// assert_eq!(sim.state_vector(StateVectorEndian::Little).len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct Simulator {
    n: usize,
    rank: usize,
    x0: BitVec,
    /// Physical rows of the `n x rank` affine support matrix.
    basis_rows: Vec<BitVec>,
    /// Sparse coefficients of `[mask . u]`, reduced modulo eight.
    gadgets: HashMap<BitVec, u8>,
    rng: ChaCha8Rng,
}

impl Simulator {
    /// Creates an empty simulator using system entropy.
    pub fn new() -> Self {
        Self::new_with_optional_seed(None)
    }

    /// Creates an empty simulator with a reproducible seed.
    pub fn with_seed(seed: u64) -> Self {
        Self::new_with_optional_seed(Some(seed))
    }

    pub(crate) fn new_with_optional_seed(seed: Option<u64>) -> Self {
        Self::from_rng(seed.map_or_else(ChaCha8Rng::from_os_rng, ChaCha8Rng::seed_from_u64))
    }

    pub(crate) fn from_rng(rng: ChaCha8Rng) -> Self {
        Self {
            n: 0,
            rank: 0,
            x0: BitVec::zero(0),
            basis_rows: Vec::new(),
            gadgets: HashMap::new(),
            rng,
        }
    }

    pub(crate) fn into_rng(self) -> ChaCha8Rng {
        self.rng
    }

    #[cfg(test)]
    fn from_product(spec: &[u8], seed: u64) -> Self {
        let n = spec.len();
        let rank = spec.iter().filter(|symbol| **symbol == b'+').count();
        let mut basis_rows = (0..n).map(|_| BitVec::zero(rank)).collect::<Vec<_>>();
        let mut coordinate = 0;
        for (qubit, symbol) in spec.iter().enumerate() {
            match symbol {
                b'0' => {}
                b'+' => {
                    basis_rows[qubit].set(coordinate, true);
                    coordinate += 1;
                }
                _ => panic!("product-state symbols must be '0' or '+'"),
            }
        }
        Self {
            n,
            rank,
            x0: BitVec::zero(n),
            basis_rows,
            gadgets: HashMap::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Returns the number of allocated physical qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub(crate) fn ensure_num_qubits(&mut self, new_n: usize) {
        if new_n <= self.n {
            return;
        }
        self.x0.resize(new_n);
        self.basis_rows
            .extend((self.n..new_n).map(|_| BitVec::zero(self.rank)));
        self.n = new_n;
    }

    /// Adds a parity phase gadget term to the phase polynomial.
    fn add_phase_gadget(&mut self, mask: BitVec, coefficient: i16) {
        debug_assert_eq!(mask.len(), self.rank);
        if mask.is_zero() {
            return;
        }
        let coefficient = coefficient.rem_euclid(8) as u8;
        if coefficient == 0 {
            return;
        }
        let combined = self
            .gadgets
            .get(&mask)
            .copied()
            .unwrap_or(0)
            .wrapping_add(coefficient)
            & 7;
        if combined == 0 {
            self.gadgets.remove(&mask);
        } else {
            self.gadgets.insert(mask, combined);
        }
    }

    /// Applies a parity phase gadget to the given slice of qubits.
    fn apply_phase_gadget(&mut self, qubits: &[usize], power: i16) {
        let mut mask = BitVec::zero(self.rank);
        let mut offset = false;
        for &qubit in qubits {
            mask.xor_assign(&self.basis_rows[qubit]);
            offset ^= self.x0.get(qubit);
        }
        self.add_phase_gadget(mask, if offset { -power } else { power });
    }

    /// Restricts the affine subspace to satisfy `normal . u = value`.
    ///
    /// This reduces the rank by 1.
    fn restrict_affine(&mut self, normal: &BitVec, value: bool) {
        debug_assert_eq!(normal.len(), self.rank);
        let pivot = normal.pivot().expect("normal must be nonzero");

        let old_rows = std::mem::take(&mut self.basis_rows);
        for (qubit, old_row) in old_rows.iter().enumerate() {
            let pivot_value = old_row.get(pivot);
            if value && pivot_value {
                self.x0.toggle(qubit);
            }
            let mut transformed = old_row.clone();
            if pivot_value {
                transformed.xor_assign(normal);
            }
            self.basis_rows.push(transformed.remove_bit(pivot));
        }

        let old_gadgets = std::mem::take(&mut self.gadgets);
        self.rank -= 1;
        for (mask, coefficient) in old_gadgets {
            let pivot_value = mask.get(pivot);
            let mut transformed = mask.clone();
            if pivot_value {
                transformed.xor_assign(normal);
            }
            let transformed = transformed.remove_bit(pivot);
            self.add_phase_gadget(
                transformed,
                if value && pivot_value {
                    -i16::from(coefficient)
                } else {
                    i16::from(coefficient)
                },
            );
        }
    }

    /// Adds a new coordinate to the affine subspace.
    ///
    /// This increases the rank by 1.
    fn add_affine_coordinate(&mut self, qubit: usize, minus: bool) {
        for row in &mut self.basis_rows {
            row.resize(self.rank + 1);
        }
        self.basis_rows[qubit].set(self.rank, true);
        let old = std::mem::take(&mut self.gadgets);
        for (mut mask, coefficient) in old {
            mask.resize(self.rank + 1);
            self.gadgets.insert(mask, coefficient);
        }
        self.rank += 1;
        if minus {
            self.add_phase_gadget(BitVec::unit(self.rank, self.rank - 1), 4);
        }
    }

    /// Translates an n-bit vector into basis coordinates of the affine subspace.
    ///
    /// Returns `None` if vector does not lie in the subspace.
    fn to_basis_coordinates(&self, x: &BitVec) -> Option<BitVec> {
        debug_assert_eq!(x.len(), self.n);
        let mut rows = self.basis_rows.clone();
        let mut rhs = (0..self.n).map(|row| x.get(row)).collect::<Vec<_>>();
        let mut pivot_rows = vec![usize::MAX; self.rank];
        let mut next_row = 0;
        for (column, pivot_slot) in pivot_rows.iter_mut().enumerate() {
            let pivot = (next_row..self.n).find(|row| rows[*row].get(column))?;
            rows.swap(next_row, pivot);
            rhs.swap(next_row, pivot);
            let pivot_vector = rows[next_row].clone();
            let pivot_rhs = rhs[next_row];
            for row in 0..self.n {
                if row != next_row && rows[row].get(column) {
                    rows[row].xor_assign(&pivot_vector);
                    rhs[row] ^= pivot_rhs;
                }
            }
            *pivot_slot = next_row;
            next_row += 1;
        }
        if (next_row..self.n).any(|row| rows[row].is_zero() && rhs[row]) {
            return None;
        }
        let mut solution = BitVec::zero(self.rank);
        for (column, row) in pivot_rows.into_iter().enumerate() {
            solution.set(column, rhs[row]);
        }
        Some(solution)
    }

    /// Returns the affine basis coordinates of the i-th unit vector.
    ///
    /// Returns `None` if the unit vector does not lie in the subspace.
    fn unit_basis_coordinates(&self, i: usize) -> Option<BitVec> {
        self.to_basis_coordinates(&BitVec::unit(self.n, i))
    }

    #[cfg(feature = "python")]
    pub(crate) fn copied(&self, copy_rng: bool, seed: Option<u64>) -> Self {
        let mut out = self.clone();
        out.rng = if copy_rng {
            self.rng.clone()
        } else {
            seed.map_or_else(ChaCha8Rng::from_os_rng, ChaCha8Rng::seed_from_u64)
        };
        out
    }

    /// Reconstructs a normalized dense state vector.
    pub fn state_vector(&self, endian: StateVectorEndian) -> Vec<Complex32> {
        let dimension = 1_usize << self.n;
        let support_size = 1_usize << self.rank;
        let scale = 2.0_f32.powf(-(self.rank as f32) / 2.0);
        let omega = |power: u8| match power & 7 {
            0 => Complex32::new(1.0, 0.0),
            1 => Complex32::new(
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
            2 => Complex32::new(0.0, 1.0),
            3 => Complex32::new(
                -std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
            4 => Complex32::new(-1.0, 0.0),
            5 => Complex32::new(
                -std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
            ),
            6 => Complex32::new(0.0, -1.0),
            7 => Complex32::new(
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
            ),
            _ => unreachable!(),
        };
        let index_of = |bits: &BitVec| match endian {
            StateVectorEndian::Little => bits.to_index_lsb(),
            StateVectorEndian::Big => bits.to_index_msb(),
        };

        let gadgets = self.gadgets.iter().collect::<Vec<_>>();
        let mut gadget_values = vec![false; gadgets.len()];
        let mut phase = 0_u8;
        let mut physical = self.x0.clone();
        let mut amplitudes = vec![Complex32::new(0.0, 0.0); dimension];
        let mut first_nonzero = index_of(&physical);
        let mut canonical_phase = 0;
        amplitudes[first_nonzero] = Complex32::new(scale, 0.0);
        let mut previous_gray = 0_usize;
        for step in 1..support_size {
            let gray = step ^ (step >> 1);
            let changed = (gray ^ previous_gray).trailing_zeros() as usize;
            for (qubit, row) in self.basis_rows.iter().enumerate() {
                if row.get(changed) {
                    physical.toggle(qubit);
                }
            }
            for (index, (mask, coefficient)) in gadgets.iter().enumerate() {
                if mask.get(changed) {
                    gadget_values[index] = !gadget_values[index];
                    phase = if gadget_values[index] {
                        phase.wrapping_add(**coefficient) & 7
                    } else {
                        phase.wrapping_sub(**coefficient) & 7
                    };
                }
            }
            let output = index_of(&physical);
            amplitudes[output] = omega(phase) * scale;
            if output < first_nonzero {
                first_nonzero = output;
                canonical_phase = phase;
            }
            previous_gray = gray;
        }
        let rotation = omega(8_u8.wrapping_sub(canonical_phase));
        for amplitude in &mut amplitudes {
            *amplitude *= rotation;
        }
        amplitudes[first_nonzero] = Complex32::new(scale, 0.0);
        amplitudes
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}
