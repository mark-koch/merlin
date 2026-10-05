//! Polynomial-time simulation tools for CNOT+T DCP-stabilizer states.
//!
//! The crate provides [`Simulator`] for interactive simulation and
//! [`CircuitSampler`] for sampling Stim-format circuits.
//!
//! ```
//! use merlin::{Simulator, StateVectorEndian};
//!
//! let mut sim = Simulator::new();
//! sim.t(0);
//! sim.cnot(0, 1);
//! assert_eq!(sim.num_qubits(), 2);
//! assert_eq!(sim.peek_observable_expectation("ZZ"), 1.0);
//! let state = sim.state_vector(StateVectorEndian::Little);
//! assert_eq!(state.len(), 4);
//! ```
#![warn(missing_docs)]

mod bits;
mod circuit;
mod sampling;
mod simulator;

#[cfg(feature = "python")]
mod bindings;

pub use crate::circuit::Circuit;
pub use crate::sampling::{CircuitSampler, CircuitSamples};
pub use crate::simulator::Simulator;

/// Qubit ordering used when reconstructing a state vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateVectorEndian {
    /// Qubit 0 is the least-significant index bit.
    Little,
    /// Qubit 0 is the most-significant index bit.
    Big,
}

/// A measurement that is incompatible with the DCP representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementError {
    message: String,
    shot: Option<usize>,
    measurement_index: Option<usize>,
}

impl MeasurementError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            shot: None,
            measurement_index: None,
        }
    }

    pub(crate) fn at(message: impl Into<String>, shot: usize, measurement_index: usize) -> Self {
        Self {
            message: message.into(),
            shot: Some(shot),
            measurement_index: Some(measurement_index),
        }
    }

    pub(crate) fn at_measurement(message: impl Into<String>, measurement_index: usize) -> Self {
        Self {
            message: message.into(),
            shot: None,
            measurement_index: Some(measurement_index),
        }
    }

    /// Returns the zero-based shot index, when the error occurred while sampling.
    pub fn shot(&self) -> Option<usize> {
        self.shot
    }

    /// Returns the zero-based measurement-record index, when available.
    pub fn measurement_index(&self) -> Option<usize> {
        self.measurement_index
    }
}

impl std::fmt::Display for MeasurementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)?;
        match (self.shot, self.measurement_index) {
            (Some(shot), Some(measurement)) => {
                write!(formatter, " (shot {shot}, measurement {measurement})")?;
            }
            (None, Some(measurement)) => {
                write!(formatter, " (reference measurement {measurement})")?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl std::error::Error for MeasurementError {}
