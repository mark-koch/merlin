"""Stim-text helpers and exact CNOT/T decompositions."""

from __future__ import annotations

from collections.abc import Iterable


def append_cnot(lines: list[str], control: int, target: int) -> None:
    lines.append(f"CX {int(control)} {int(target)}")


def append_cs(lines: list[str], control: int, target: int) -> None:
    """Append CS using the phase polynomial 2ab = a+b-(a xor b)."""
    a, b = int(control), int(target)
    lines.extend((f"T {a}", f"T {b}", f"CX {a} {b}", f"T_DAG {b}", f"CX {a} {b}"))


def append_ccz(lines: list[str], a: int, b: int, c: int) -> None:
    """Append CCZ as seven parity-phase gadgets (7 T and 10 CNOTs)."""
    a, b, c = int(a), int(b), int(c)
    lines.extend((f"T {a}", f"T {b}", f"T {c}"))
    for x, y in ((a, b), (a, c), (b, c)):
        lines.extend((f"CX {x} {y}", f"T_DAG {y}", f"CX {x} {y}"))
    lines.extend(
        (f"CX {a} {c}", f"CX {b} {c}", f"T {c}", f"CX {b} {c}", f"CX {a} {c}")
    )


def append_ops(
    lines: list[str],
    ops: Iterable[tuple[str, int, int]],
    *,
    p_phys: float = 0.0,
    noisy: bool = False,
) -> None:
    """Append CNOT operations, optionally followed by two-qubit noise."""
    for kind, a, b in ops:
        if kind != "cnot":
            raise ValueError(f"unsupported CSS operation: {kind!r}")
        append_cnot(lines, a, b)
        if noisy and p_phys:
            lines.append(f"DEPOLARIZE2({p_phys:.17g}) {int(a)} {int(b)}")


def append_detectors(lines: list[str], count: int) -> None:
    for index in range(int(count)):
        lines.append(f"DETECTOR rec[-{int(count) - index}]")


def append_x_observables(lines: list[str], qubits: Iterable[int]) -> int:
    qubits = tuple(int(q) for q in qubits)
    if not qubits:
        return 0
    lines.append("MX " + " ".join(map(str, qubits)))
    count = len(qubits)
    for index in range(count):
        lines.append(f"OBSERVABLE_INCLUDE({index}) rec[-{count - index}]")
    return count
