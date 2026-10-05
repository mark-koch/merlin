"""Shared immutable data models for benchmark protocols."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from typing import Any


def _circuit_metrics(text: str) -> dict[str, int]:
    """Cheap structural metrics for the emitted Stim-compatible program."""
    gates: Counter[str] = Counter()
    instruction_count = 0
    measurement_count = 0
    measurement_gates = {"M", "MX", "MY", "MR", "MRX", "MRY"}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line in {"{", "}"}:
            continue
        fields = line.split()
        gate = fields[0].split("(", 1)[0].upper()
        if gate == "REPEAT":
            continue
        instruction_count += 1
        gates[gate] += 1
        if gate in measurement_gates:
            measurement_count += len(fields) - 1
    return {
        "instruction_count": instruction_count,
        "t_count": gates["T"],
        "t_dag_count": gates["T_DAG"],
        "non_clifford_rotation_count": gates["T"] + gates["T_DAG"],
        "measurement_count": measurement_count,
    }


@dataclass(frozen=True)
class CircuitCase:
    """One fully constructed detector-sampling workload."""

    case_id: str
    family: str
    circuit: str
    n_phys: int
    n_detectors: int
    n_observables: int
    p_phys: float
    noise_model: str
    target_scoring: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detector_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n_detectors))

    def description(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "family": self.family,
            "n_phys": self.n_phys,
            "n_detectors": self.n_detectors,
            "n_observables": self.n_observables,
            "circuit_sha256": hashlib.sha256(self.circuit.encode()).hexdigest(),
            "circuit_metrics": _circuit_metrics(self.circuit),
            **self.metadata,
        }
