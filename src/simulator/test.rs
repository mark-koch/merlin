//! Statevector utilities for testing.

use num_complex::Complex32;

use crate::circuit::{Pauli, PauliProduct};

const TOLERANCE: f32 = 2.0e-5;

pub fn equivalent(left: &[Complex32], right: &[Complex32]) -> bool {
    let Some(pivot) = right.iter().position(|value| value.norm() > TOLERANCE) else {
        return left.iter().all(|value| value.norm() <= TOLERANCE);
    };
    let phase = left[pivot] / right[pivot];
    left.iter()
        .zip(right)
        .all(|(a, b)| (*a - phase * *b).norm() <= TOLERANCE)
}

pub fn project(
    state: &[Complex32],
    qubit: usize,
    x_basis: bool,
    minus: bool,
) -> Option<(f64, Vec<Complex32>)> {
    let mut projected = vec![Complex32::new(0.0, 0.0); state.len()];
    let epsilon = if minus { -1.0 } else { 1.0 };
    if x_basis {
        for (index, output) in projected.iter_mut().enumerate() {
            *output = (state[index] + epsilon * state[index ^ (1 << qubit)]) * 0.5;
        }
    } else {
        for (index, output) in projected.iter_mut().enumerate() {
            if ((index >> qubit) & 1 != 0) == minus {
                *output = state[index];
            }
        }
    }
    let probability = projected
        .iter()
        .map(|value| f64::from(value.norm_sqr()))
        .sum::<f64>();
    if probability < 1.0e-10 {
        return None;
    }
    let normalization = probability.sqrt() as f32;
    for value in &mut projected {
        *value /= normalization;
    }
    Some((probability, projected))
}

pub fn project_observable(
    state: &[Complex32],
    product: &PauliProduct,
    minus: bool,
) -> Option<(f64, Vec<Complex32>)> {
    let mut acted = vec![Complex32::new(0.0, 0.0); state.len()];
    for (source, amplitude) in state.iter().copied().enumerate() {
        let mut target = source;
        let mut phase = if product.negative {
            Complex32::new(-1.0, 0.0)
        } else {
            Complex32::new(1.0, 0.0)
        };
        for &(qubit, pauli) in &product.factors {
            let bit = (source >> qubit) & 1 != 0;
            match pauli {
                Pauli::X => target ^= 1 << qubit,
                Pauli::Y => {
                    target ^= 1 << qubit;
                    phase *= if bit {
                        Complex32::new(0.0, -1.0)
                    } else {
                        Complex32::new(0.0, 1.0)
                    };
                }
                Pauli::Z => {
                    if bit {
                        phase = -phase;
                    }
                }
            }
        }
        acted[target] += phase * amplitude;
    }
    let epsilon = if minus { -1.0 } else { 1.0 };
    let mut projected = state
        .iter()
        .zip(acted)
        .map(|(original, transformed)| (*original + epsilon * transformed) * 0.5)
        .collect::<Vec<_>>();
    let probability = projected
        .iter()
        .map(|value| f64::from(value.norm_sqr()))
        .sum::<f64>();
    if probability < 1.0e-10 {
        return None;
    }
    let normalization = probability.sqrt() as f32;
    for value in &mut projected {
        *value /= normalization;
    }
    Some((probability, projected))
}

pub fn dense_observable_expectation(state: &[Complex32], product: &PauliProduct) -> f64 {
    let mut expectation = Complex32::new(0.0, 0.0);
    for (source, amplitude) in state.iter().copied().enumerate() {
        let mut target = source;
        let mut phase = if product.negative {
            Complex32::new(-1.0, 0.0)
        } else {
            Complex32::new(1.0, 0.0)
        };
        for &(qubit, pauli) in &product.factors {
            let bit = (source >> qubit) & 1 != 0;
            match pauli {
                Pauli::X => target ^= 1 << qubit,
                Pauli::Y => {
                    target ^= 1 << qubit;
                    phase *= if bit {
                        Complex32::new(0.0, -1.0)
                    } else {
                        Complex32::new(0.0, 1.0)
                    };
                }
                Pauli::Z => {
                    if bit {
                        phase = -phase;
                    }
                }
            }
        }
        expectation += state[target].conj() * phase * amplitude;
    }
    assert!(expectation.im.abs() < TOLERANCE);
    f64::from(expectation.re)
}

pub fn mask(rank: usize, indices: &[usize]) -> crate::bits::BitVec {
    let mut out = crate::bits::BitVec::zero(rank);
    for &index in indices {
        out.set(index, true);
    }
    out
}
