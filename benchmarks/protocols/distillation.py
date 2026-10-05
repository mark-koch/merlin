"""15-to-1 and Bravyi-Haah distillation circuits."""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.models import CircuitCase

from .css import matrix_15to1, matrix_bravyi_haah_3k8, triorthogonal_css_code
from .gates import append_detectors, append_ops, append_x_observables


def _matrix_for_protocol(protocol: str, k: int):
    key = str(protocol).lower()
    if key == "15to1":
        return key, matrix_15to1(), 1
    if key == "bh":
        return f"bh:{int(k)}", matrix_bravyi_haah_3k8(int(k)), int(k)
    raise ValueError(f"unknown distillation protocol: {protocol!r}")


def _append_transversal_t_dag_correction(lines: list[str], generator, k_logical: int) -> None:
    """Remove the decoded Clifford phase left by transversal T-dagger.

    Triorthogonality eliminates cubic phases, but even pair overlaps and
    check rows whose weights are 4 modulo 8 can still leave CZ and Z phases.
    After decoding, generator row ``i`` is qubit ``i``, so these corrections
    can be applied directly in the decoded basis.
    """
    generator = generator.astype(int, copy=False)
    desired_linear = [1] * int(k_logical) + [0] * (len(generator) - int(k_logical))
    for qubit, (row, desired) in enumerate(zip(generator, desired_linear)):
        correction = (desired + int(row.sum())) % 8
        if correction == 4:
            lines.append(f"Z {qubit}")
        elif correction:
            raise ValueError(
                "transversal T-dagger requires a non-Clifford linear correction "
                f"on decoded qubit {qubit}: phase {correction} mod 8"
            )
    for left in range(len(generator)):
        for right in range(left + 1, len(generator)):
            correction = (-2 * int(generator[left] @ generator[right])) % 8
            if correction == 4:
                lines.append(f"CZ {left} {right}")
            elif correction:
                raise ValueError(
                    "transversal T-dagger requires a non-Clifford quadratic "
                    f"correction on decoded qubits {(left, right)}: "
                    f"phase {correction} mod 8"
                )


def build_distillation_case(
    protocol: str,
    *,
    k: int = 2,
    p_phys: float = 0.0,
    noisy_clifford: bool = False,
    target_scoring: bool = True,
    deterministic_z_faults: Iterable[int] = (),
) -> CircuitCase:
    """Construct one parse-once distillation detector circuit.

    The clean model places independent Z noise after each physical T-dagger.
    The noisy-Clifford model instead places depolarizing noise after every
    encoder/decoder CNOT and physical T-dagger.
    """
    p_phys = float(p_phys)
    if not 0.0 <= p_phys <= 1.0:
        raise ValueError("p_phys must be in [0, 1]")
    case_id, generator, k_logical = _matrix_for_protocol(protocol, k)
    code = triorthogonal_css_code(generator, k_logical=k_logical, name=case_id)
    lines: list[str] = []
    if code.n_encoder_rows:
        lines.append("RX " + " ".join(map(str, code.plus_qubits())))

    append_ops(lines, code.encoder_ops(), p_phys=p_phys, noisy=noisy_clifford)
    forced = set(int(q) for q in deterministic_z_faults)
    unknown = forced.difference(range(code.n))
    if unknown:
        raise ValueError(f"deterministic fault qubits out of range: {sorted(unknown)}")
    for qubit in range(code.n):
        lines.append(f"T_DAG {qubit}")
        if p_phys:
            channel = "DEPOLARIZE1" if noisy_clifford else "Z_ERROR"
            lines.append(f"{channel}({p_phys:.17g}) {qubit}")
        if qubit in forced:
            lines.append(f"Z_ERROR(1) {qubit}")
    append_ops(lines, code.decoder_ops(), p_phys=p_phys, noisy=noisy_clifford)
    _append_transversal_t_dag_correction(lines, generator, code.k)

    checks = code.check_qubits()
    for basis, qubit in checks:
        lines.append(f"{'MX' if basis == 'mx' else 'M'} {qubit}")
    append_detectors(lines, len(checks))

    n_observables = 0
    if target_scoring:
        for qubit in range(code.k):
            lines.append(f"T_DAG {qubit}")
        n_observables = append_x_observables(lines, range(code.k))

    noise_model = "depolarizing_all_gates" if noisy_clifford else "z_after_physical_t"
    return CircuitCase(
        case_id=case_id,
        family="distillation",
        circuit="\n".join(lines) + "\n",
        n_phys=code.n,
        n_detectors=len(checks),
        n_observables=n_observables,
        p_phys=p_phys,
        noise_model=noise_model,
        target_scoring=bool(target_scoring),
        metadata={
            "protocol": str(protocol).lower(),
            "k_logical": code.k,
            "n_rows": int(generator.shape[0]),
            "t_phys": code.n,
            "noisy_clifford": bool(noisy_clifford),
            "deterministic_faults": sorted(forced),
        },
    )
