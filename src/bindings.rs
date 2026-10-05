use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};
use pyo3::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::StateVectorEndian;
use crate::circuit::{Circuit, PauliProduct};
use crate::sampling::{CircuitSampler as CoreCircuitSampler, CircuitSamples};
use crate::simulator::Simulator as CoreSimulator;

pyo3::create_exception!(_core, PyMeasurementError, PyException);

#[pyclass(module = "merlin._core", name = "Simulator", skip_from_py_object)]
struct PySimulator {
    inner: CoreSimulator,
}

#[pyclass(module = "merlin._core", name = "CircuitSampler", skip_from_py_object)]
struct PyCircuitSampler {
    inner: CoreCircuitSampler,
}

#[pyclass(module = "merlin._core", name = "SampleResult", skip_from_py_object)]
struct PySampleResult {
    measurements: Py<PyAny>,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    exp_vals: Py<PyAny>,
    total_shots: usize,
    accepted_shots: usize,
}

impl PyCircuitSampler {
    fn result_to_python(
        &self,
        py: Python<'_>,
        samples: CircuitSamples<bool>,
        bit_packed: bool,
    ) -> PyResult<PySampleResult> {
        let rows = samples.accepted_shots;
        let measurements = sample_array(
            py,
            rows,
            self.inner.num_measurements(),
            samples.measurements,
            bit_packed,
        )?
        .unbind();
        let detectors = sample_array(
            py,
            rows,
            self.inner.num_detectors(),
            samples.detectors,
            bit_packed,
        )?
        .unbind();
        let observables = sample_array(
            py,
            rows,
            self.inner.num_observables(),
            samples.observables,
            bit_packed,
        )?
        .unbind();
        let exp_vals = Array2::from_shape_vec((rows, self.inner.num_exp_vals()), samples.exp_vals)
            .map_err(|error| PyValueError::new_err(format!("invalid sample shape: {error}")))?;
        Ok(PySampleResult {
            measurements,
            detectors,
            observables,
            exp_vals: PyArray2::from_owned_array(py, exp_vals).into_any().unbind(),
            total_shots: samples.total_shots,
            accepted_shots: rows,
        })
    }
}

#[pymethods]
impl PySampleResult {
    #[getter]
    fn measurements(&self, py: Python<'_>) -> Py<PyAny> {
        self.measurements.clone_ref(py)
    }

    #[getter]
    fn detectors(&self, py: Python<'_>) -> Py<PyAny> {
        self.detectors.clone_ref(py)
    }

    #[getter]
    fn observables(&self, py: Python<'_>) -> Py<PyAny> {
        self.observables.clone_ref(py)
    }

    #[getter]
    fn exp_vals(&self, py: Python<'_>) -> Py<PyAny> {
        self.exp_vals.clone_ref(py)
    }

    #[getter]
    fn total_shots(&self) -> usize {
        self.total_shots
    }

    #[getter]
    fn accepted_shots(&self) -> usize {
        self.accepted_shots
    }

    #[getter]
    fn discards(&self) -> usize {
        self.total_shots - self.accepted_shots
    }

    fn __repr__(&self) -> String {
        format!(
            "SampleResult(total_shots={}, accepted_shots={})",
            self.total_shots, self.accepted_shots
        )
    }
}

#[pymethods]
impl PyCircuitSampler {
    #[new]
    #[pyo3(signature = (circuit, *, seed=None, postselect=None))]
    fn new(
        py: Python<'_>,
        circuit: &str,
        seed: Option<u64>,
        postselect: Option<Vec<isize>>,
    ) -> PyResult<Self> {
        py.detach(|| {
            let parsed = catch_unwind(AssertUnwindSafe(|| Circuit::try_parse(circuit)))
                .map_err(|_| PyValueError::new_err("invalid Stim circuit"))?;
            let circuit = parsed.map_err(PyValueError::new_err)?;
            let mut seen = vec![false; circuit.num_detectors()];
            let mut indices = Vec::new();
            for detector in postselect.unwrap_or_default() {
                let detector = usize::try_from(detector).map_err(|_| {
                    PyValueError::new_err("postselected detector indices must be non-negative")
                })?;
                if detector >= seen.len() {
                    return Err(PyValueError::new_err(format!(
                        "postselected detector index {detector} is out of range"
                    )));
                }
                if seen[detector] {
                    return Err(PyValueError::new_err(format!(
                        "postselected detector index {detector} appears more than once"
                    )));
                }
                seen[detector] = true;
                indices.push(detector);
            }
            let inner = CoreCircuitSampler::from_circuit_with_options(&circuit, seed, &indices)
                .map_err(|error| PyMeasurementError::new_err(error.to_string()))?;
            Ok(Self { inner })
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_measurements(&self) -> usize {
        self.inner.num_measurements()
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    #[getter]
    fn num_exp_vals(&self) -> usize {
        self.inner.num_exp_vals()
    }

    #[getter]
    fn num_fault_sites(&self) -> usize {
        self.inner.num_fault_sites()
    }

    #[pyo3(signature = (shots, *, bit_packed=false, num_faults=None))]
    fn sample(
        &mut self,
        py: Python<'_>,
        shots: isize,
        bit_packed: bool,
        num_faults: Option<isize>,
    ) -> PyResult<PySampleResult> {
        if shots < 0 {
            return Err(PyValueError::new_err("shots must be non-negative"));
        }
        let samples = match num_faults {
            Some(num_faults) => {
                let num_faults = usize::try_from(num_faults)
                    .map_err(|_| PyValueError::new_err("num_faults must be non-negative"))?;
                if let Some(message) = self.inner.fixed_fault_error(num_faults) {
                    return Err(PyValueError::new_err(message));
                }
                py.detach(|| self.inner.sample_fixed_faults(shots as usize, num_faults))
                    .map_err(|error| PyMeasurementError::new_err(error.to_string()))?
            }
            None => py
                .detach(|| self.inner.sample(shots as usize))
                .map_err(|error| PyMeasurementError::new_err(error.to_string()))?,
        };
        self.result_to_python(py, samples, bit_packed)
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitSampler(num_qubits={}, num_measurements={}, num_detectors={}, num_observables={}, num_exp_vals={}, num_fault_sites={})",
            self.inner.num_qubits(),
            self.inner.num_measurements(),
            self.inner.num_detectors(),
            self.inner.num_observables(),
            self.inner.num_exp_vals(),
            self.inner.num_fault_sites()
        )
    }
}

fn sample_array<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<bool>,
    bit_packed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if bit_packed {
        let width = columns.div_ceil(8);
        let mut packed = vec![0_u8; rows * width];
        for row in 0..rows {
            for column in 0..columns {
                if values[row * columns + column] {
                    packed[row * width + (column >> 3)] |= 1 << (column & 7);
                }
            }
        }
        let array = Array2::from_shape_vec((rows, width), packed)
            .map_err(|error| PyValueError::new_err(format!("invalid sample shape: {error}")))?;
        Ok(PyArray2::from_owned_array(py, array).into_any())
    } else {
        let array = Array2::from_shape_vec((rows, columns), values)
            .map_err(|error| PyValueError::new_err(format!("invalid sample shape: {error}")))?;
        Ok(PyArray2::from_owned_array(py, array).into_any())
    }
}

impl PySimulator {
    fn qubit_index(index: isize) -> PyResult<usize> {
        if index < 0 {
            Err(PyIndexError::new_err(format!(
                "qubit index {index} must be non-negative"
            )))
        } else {
            Ok(index as usize)
        }
    }

    fn qubit_indices(indices: Vec<isize>) -> PyResult<Vec<usize>> {
        indices.into_iter().map(Self::qubit_index).collect()
    }

    fn required_qubits(targets: &[usize]) -> usize {
        targets.iter().max().map_or(0, |target| target + 1)
    }

    fn probability(probability: f64) -> PyResult<f64> {
        if probability.is_finite() && (0.0..=1.0).contains(&probability) {
            Ok(probability)
        } else {
            Err(PyValueError::new_err(format!(
                "probability must be between 0 and 1, but got {probability}"
            )))
        }
    }

    fn apply_cnots(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        if targets.len() & 1 != 0 {
            return Err(PyValueError::new_err(
                "CNOT requires an even number of targets",
            ));
        }
        let targets = Self::qubit_indices(targets)?;
        if let Some(pair) = targets.chunks_exact(2).find(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(format!(
                "CNOT control and target must be different qubits (both were {})",
                pair[0]
            )));
        }
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for pair in targets.chunks_exact(2) {
                self.inner.cnot(pair[0], pair[1]);
            }
        });
        Ok(())
    }

    fn paired_targets(name: &str, targets: Vec<isize>) -> PyResult<Vec<usize>> {
        if !targets.len().is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "{name} requires an even number of targets"
            )));
        }
        let targets = Self::qubit_indices(targets)?;
        if let Some(pair) = targets.chunks_exact(2).find(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(format!(
                "{name} pair endpoints must be different qubits (both were {})",
                pair[0]
            )));
        }
        Ok(targets)
    }

    fn tripled_targets(name: &str, targets: Vec<isize>) -> PyResult<Vec<usize>> {
        if !targets.len().is_multiple_of(3) {
            return Err(PyValueError::new_err(format!(
                "{name} requires a multiple of three targets"
            )));
        }
        let targets = Self::qubit_indices(targets)?;
        if let Some(triple) = targets.chunks_exact(3).find(|triple| {
            triple[0] == triple[1] || triple[0] == triple[2] || triple[1] == triple[2]
        }) {
            return Err(PyValueError::new_err(format!(
                "{name} triple endpoints must be different (got {}, {}, {})",
                triple[0], triple[1], triple[2]
            )));
        }
        Ok(targets)
    }
}

#[pymethods]
impl PySimulator {
    #[new]
    #[pyo3(signature = (*, seed=None))]
    fn new(seed: Option<u64>) -> Self {
        Self {
            inner: CoreSimulator::new_with_optional_seed(seed),
        }
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[pyo3(signature = (*targets))]
    fn x(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.x(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn y(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.y(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn z(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.z(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn s(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.s(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn s_dag(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.s_dag(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn t(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.t(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn t_dag(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.t_dag(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn cnot(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        self.apply_cnots(py, targets)
    }

    #[pyo3(signature = (*targets))]
    fn cx(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        self.apply_cnots(py, targets)
    }

    #[pyo3(signature = (*targets))]
    fn cz(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::paired_targets("CZ", targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for pair in targets.chunks_exact(2) {
                self.inner.cz(pair[0], pair[1]);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn swap(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::paired_targets("SWAP", targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for pair in targets.chunks_exact(2) {
                self.inner.swap(pair[0], pair[1]);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn ccz(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::tripled_targets("CCZ", targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for triple in targets.chunks_exact(3) {
                self.inner.ccz(triple[0], triple[1], triple[2]);
            }
        });
        Ok(())
    }

    fn measure(&mut self, py: Python<'_>, target: isize) -> PyResult<bool> {
        let target = Self::qubit_index(target)?;
        Ok(py.detach(|| self.inner.measure(target)))
    }

    fn measure_z(&mut self, py: Python<'_>, target: isize) -> PyResult<bool> {
        self.measure(py, target)
    }

    #[pyo3(signature = (*targets))]
    fn measure_many(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<Vec<bool>> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        Ok(py.detach(|| {
            self.inner.ensure_num_qubits(required);
            targets
                .into_iter()
                .map(|target| self.inner.measure(target))
                .collect()
        }))
    }

    fn measure_x(&mut self, py: Python<'_>, target: isize) -> PyResult<bool> {
        let target = Self::qubit_index(target)?;
        py.detach(|| self.inner.measure_x(target))
            .map_err(|error| PyMeasurementError::new_err(error.to_string()))
    }

    #[pyo3(signature = (observable, *, flip_probability=0.0))]
    fn measure_observable(
        &mut self,
        py: Python<'_>,
        observable: &str,
        flip_probability: f64,
    ) -> PyResult<bool> {
        let flip_probability = Self::probability(flip_probability)?;
        let product = PauliProduct::from_compact(observable).map_err(PyValueError::new_err)?;
        py.detach(|| {
            self.inner
                .measure_pauli_product(&product)
                .map(|(outcome, _)| (outcome == -1) ^ self.inner.sample_event(flip_probability))
        })
        .map_err(PyMeasurementError::new_err)
    }

    fn peek_observable_expectation(&self, py: Python<'_>, observable: &str) -> PyResult<f64> {
        let product = PauliProduct::from_compact(observable).map_err(PyValueError::new_err)?;
        Ok(py.detach(|| self.inner.pauli_product_expectation(&product)))
    }

    fn peek_probability(&self, py: Python<'_>, target: isize) -> PyResult<f64> {
        let target = Self::qubit_index(target)?;
        Ok(py.detach(|| self.inner.peek_probability(target)))
    }

    fn peek_z_probability(&self, py: Python<'_>, target: isize) -> PyResult<f64> {
        self.peek_probability(py, target)
    }

    fn peek_x_probability(&self, py: Python<'_>, target: isize) -> PyResult<f64> {
        let target = Self::qubit_index(target)?;
        py.detach(|| self.inner.peek_x_probability(target))
            .map_err(|error| PyMeasurementError::new_err(error.to_string()))
    }

    #[pyo3(signature = (*targets, p))]
    fn x_error(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for t in targets {
                self.inner.x_error(t, probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets, p))]
    fn y_error(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for t in targets {
                self.inner.y_error(t, probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets, p))]
    fn z_error(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for t in targets {
                self.inner.z_error(t, probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets, p))]
    fn depolarize1(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for t in targets {
                self.inner.depolarize1(t, probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets, p))]
    fn depolarize2(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        if targets.len() & 1 != 0 {
            return Err(PyValueError::new_err(
                "depolarize2 requires an even number of targets",
            ));
        }
        let targets = Self::qubit_indices(targets)?;
        if let Some(pair) = targets.chunks_exact(2).find(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(format!(
                "depolarize2 pair endpoints must be different qubits (both were {})",
                pair[0]
            )));
        }
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for pair in targets.chunks_exact(2) {
                self.inner.depolarize2(pair[0], pair[1], probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets, p))]
    fn depolarize3(&mut self, py: Python<'_>, targets: Vec<isize>, p: f64) -> PyResult<()> {
        let probability = Self::probability(p)?;
        let targets = Self::tripled_targets("depolarize3", targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for triple in targets.chunks_exact(3) {
                self.inner
                    .depolarize3(triple[0], triple[1], triple[2], probability);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*, endian="little"))]
    fn state_vector<'py>(
        &self,
        py: Python<'py>,
        endian: &str,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let endian = match endian {
            "little" => StateVectorEndian::Little,
            "big" => StateVectorEndian::Big,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "endian must be 'little' or 'big', but got {endian:?}"
                )));
            }
        };
        let amplitudes = py.detach(|| self.inner.state_vector(endian));
        Ok(PyArray1::from_vec(py, amplitudes))
    }

    #[pyo3(signature = (*targets))]
    fn reset(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.reset_z_raw(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*targets))]
    fn reset_z(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        self.reset(py, targets)
    }

    #[pyo3(signature = (*targets))]
    fn reset_x(&mut self, py: Python<'_>, targets: Vec<isize>) -> PyResult<()> {
        let targets = Self::qubit_indices(targets)?;
        let required = Self::required_qubits(&targets);
        py.detach(|| {
            self.inner.ensure_num_qubits(required);
            for target in targets {
                self.inner.reset_x_raw(target);
            }
        });
        Ok(())
    }

    #[pyo3(signature = (*, copy_rng=false, seed=None))]
    fn copy(&self, copy_rng: bool, seed: Option<u64>) -> PyResult<Self> {
        if copy_rng && seed.is_some() {
            return Err(PyValueError::new_err(
                "copy_rng and seed cannot both be specified",
            ));
        }
        Ok(Self {
            inner: self.inner.copied(copy_rng, seed),
        })
    }

    fn __repr__(&self) -> String {
        format!("Simulator(num_qubits={})", self.inner.num_qubits())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySimulator>()?;
    module.add_class::<PyCircuitSampler>()?;
    module.add_class::<PySampleResult>()?;
    module.add(
        "MeasurementError",
        module.py().get_type::<PyMeasurementError>(),
    )?;
    Ok(())
}
