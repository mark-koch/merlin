"""Distance-3 and distance-5 T-state cultivation benchmark circuits."""

from __future__ import annotations

from pathlib import Path
import re

from benchmarks.models import CircuitCase


_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "cultivation"
_NOISE = re.compile(r"^(X_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|M|MX)\(([^)]*)\)(.*)$")


def build_cultivation_case(
    distance: int,
    *,
    p_phys: float = 0.001,
    target_scoring: bool = True,
) -> CircuitCase:
    """Load the supplied T circuit and set every noise site to ``p_phys``."""
    if distance not in (3, 5):
        raise ValueError("cultivation distance must be 3 or 5")
    p_phys = float(p_phys)
    if not 0 <= p_phys <= 1:
        raise ValueError("p_phys must be in [0, 1]")

    source = _CIRCUIT_DIR / f"circuit_d{distance}_t.stim"
    lines = source.read_text(encoding="utf-8").splitlines()
    source_probabilities = set()
    rewritten = []
    for line in lines:
        match = _NOISE.fullmatch(line)
        if match:
            gate, probability, targets = match.groups()
            source_probabilities.add(float(probability))
            line = f"{gate}({p_phys:.17g}){targets}"
        if not target_scoring and line.startswith("OBSERVABLE_INCLUDE"):
            continue
        rewritten.append(line)
    if source_probabilities != {0.001}:
        raise ValueError(f"expected uniform 0.001 noise in {source}")

    qubits = [int(line.split()[-1]) for line in lines if line.startswith("QUBIT_COORDS")]
    if sorted(qubits) != list(range(len(qubits))):
        raise ValueError(f"invalid qubit coordinates in {source}")
    n_detectors = sum(line.startswith("DETECTOR") for line in lines)
    n_observables = sum(line.startswith("OBSERVABLE_INCLUDE") for line in lines)
    if n_observables != 2 or not n_detectors:
        raise ValueError(f"unexpected detector or observable count in {source}")

    return CircuitCase(
        case_id=f"cultivation-d{distance}-t",
        family="cultivation",
        circuit="\n".join(rewritten) + "\n",
        n_phys=len(qubits),
        n_detectors=n_detectors,
        n_observables=1 if target_scoring else 0,
        p_phys=p_phys,
        noise_model="uniform_circuit_pauli_and_readout",
        target_scoring=bool(target_scoring),
        metadata={"protocol": "t-state-cultivation", "distance": distance},
    )
