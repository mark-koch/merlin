"""BB/BT dimension-jump circuits followed by a physical cup-product CCZ."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from benchmarks.models import CircuitCase

from .css import CSSCode, f2_in_row_span, f2_nullspace, f2_solve
from .gates import append_ccz, append_detectors, append_ops, append_x_observables


_BBT_SPECS = {
    # The binomials and cup-product orientations are the d3/d5 BLP examples
    # from arXiv:2510.07269 and its accompanying code release.  A monomial is
    # represented by its exponent in Z_lx x Z_ly.
    "bt27": {
        "size": (3, 3),
        "distance": (3, 3),
        "checks": {
            # These monomial shifts match the oriented complex used by the
            # released physical-CCZ tensor (rather than merely an equivalent
            # normalized presentation of the same code).
            "a": ((1, 0), (2, 1)),
            "b": ((2, 1), (2, 2)),
            "c": ((0, 0), (1, 2)),
        },
        # Equivalent monomial shifts of the check binomials, with the
        # in/out orientations used to construct the physical cup product.
        "orientation": {
            "a": ((1, 0), (2, 1)),
            "b": ((2, 1), (2, 2)),
            "c": ((0, 0), (1, 2)),
        },
    },
    "bt81": {
        "size": (3, 9),
        "distance": (5, 6),
        "checks": {
            "a": ((2, 4), (2, 6)),
            "b": ((1, 3), (2, 1)),
            "c": ((0, 0), (1, 8)),
        },
        "orientation": {
            "a": ((2, 4), (2, 6)),
            "b": ((1, 3), (2, 1)),
            "c": ((0, 0), (1, 8)),
        },
    },
}


def _monomial_matrix(lx: int, ly: int, exponent: tuple[int, int]) -> np.ndarray:
    """Regular representation of one monomial in F2[Z_lx x Z_ly]."""
    dx, dy = exponent
    cells = lx * ly
    matrix = np.zeros((cells, cells), dtype=np.int64)
    for y in range(ly):
        for x in range(lx):
            source = x + lx * y
            target = (x + dx) % lx + lx * ((y + dy) % ly)
            matrix[target, source] = 1
    return matrix


def _binomial_matrix(lx: int, ly: int, terms: tuple[tuple[int, int], ...]) -> np.ndarray:
    matrix = np.zeros((lx * ly, lx * ly), dtype=np.int64)
    for exponent in terms:
        matrix ^= _monomial_matrix(lx, ly, exponent)
    return matrix


def _bbt_check_matrices(spec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx, ly = spec["size"]
    a, b, c = (_binomial_matrix(lx, ly, spec["checks"][name]) for name in "abc")
    zero = np.zeros_like(a)
    source_hx = np.hstack([c, b, a])
    source_hz_t = np.vstack(
        [
            np.hstack([b, a, zero]),
            np.hstack([c, zero, a]),
            np.hstack([zero, c, b]),
        ]
    )
    target_hx = np.hstack([c, b])
    target_hz = np.hstack([b.T, c.T])
    return source_hx, source_hz_t.T, target_hx, target_hz


@dataclass(frozen=True)
class BBTPair:
    name: str
    lx: int
    ly: int
    source: CSSCode
    target: CSSCode
    source_distance: int
    target_distance: int

    @property
    def cells(self) -> int:
        return self.lx * self.ly


def make_bbt_pair(name: str) -> BBTPair:
    key = str(name).lower()
    if key not in _BBT_SPECS:
        raise ValueError(f"unknown BB/BT pair: {name!r}")
    spec = _BBT_SPECS[key]
    lx, ly = spec["size"]
    source_distance, target_distance = spec["distance"]
    source_hx, source_hz, target_hx, target_hz = _bbt_check_matrices(spec)
    target = CSSCode.from_checks(
        f"{key}_2d_ab",
        target_hx,
        target_hz,
        distance=target_distance,
    )
    source = CSSCode.from_checks(
        f"{key}_3d",
        source_hx,
        source_hz,
        distance=source_distance,
    )
    switch = switch_matrix_3d_to_2d(
        BBTPair(key, lx, ly, source, target, source_distance, target_distance)
    )
    logical_map = source.x_logicals @ switch.T @ target.logical_zs.T % 2
    aligned_rows = [
        f2_solve(logical_map.T, np.eye(target.k, dtype=np.int64)[logical])
        for logical in range(target.k)
    ]
    aligned_rows.extend(f2_nullspace(logical_map.T, n=source.k))
    logical_change = np.vstack(aligned_rows) % 2
    source = CSSCode(
        source.name,
        logical_change @ source.x_logicals % 2,
        source.hx,
        source.hz,
        source.distance,
    )
    return BBTPair(
        key,
        lx,
        ly,
        source,
        target,
        source_distance,
        target_distance,
    )


def switch_matrix_3d_to_2d(pair: BBTPair) -> np.ndarray:
    matrix = np.zeros((pair.target.n, pair.source.n), dtype=np.int64)
    for sector in range(pair.target.k):
        start = sector * pair.cells
        for cell in range(pair.cells):
            matrix[start + cell, start + cell] = 1
    return matrix


def switch_cnot_pairs(pair: BBTPair) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(source), int(target))
        for target, source in zip(*np.where(switch_matrix_3d_to_2d(pair) == 1))
    )


def ccz_triples_for_pair(pair: BBTPair) -> tuple[tuple[int, int, int], ...]:
    spec = _BBT_SPECS.get(pair.name)
    if spec is None or spec["size"] != (pair.lx, pair.ly):
        # Tiny/custom pairs used by runner tests have one logical per physical
        # sector, for which the coordinatewise tensor is exact.
        return tuple(
            (sector * pair.cells + cell,) * 3
            for sector in range(pair.source.k)
            for cell in range(pair.cells)
        )

    lx, ly = pair.lx, pair.ly
    modulus = (lx, ly)

    def add(*terms):
        return tuple(sum(term[axis] for term in terms) % modulus[axis] for axis in range(2))

    def inverse(term):
        return tuple(-term[axis] % modulus[axis] for axis in range(2))

    (a_in, a_out), (b_in, b_out), (c_in, c_out) = (
        spec["orientation"][name] for name in "abc"
    )
    da_in, da_out = inverse(a_in), inverse(a_out)
    db_in, db_out = inverse(b_in), inverse(b_out)
    dc_in, dc_out = inverse(c_in), inverse(c_out)
    seeds = (
        ((dc_in, 0), (db_out, 1), (add(da_out, db_out, b_in), 2)),
        ((db_in, 1), (da_out, 2), (add(dc_out, da_out, a_in), 0)),
        ((da_in, 2), (dc_out, 0), (add(db_out, dc_out, c_in), 1)),
        ((dc_in, 0), (da_out, 2), (add(da_out, a_in, db_out), 1)),
        ((db_in, 1), (dc_out, 0), (add(dc_out, c_in, da_out), 2)),
        ((da_in, 2), (db_out, 1), (add(db_out, b_in, dc_out), 0)),
    )
    triples = []
    for y in range(ly):
        for x in range(lx):
            shift = (x, y)
            for seed in seeds:
                triple = []
                for exponent, sector in seed:
                    shifted = add(shift, exponent)
                    triple.append(sector * pair.cells + shifted[0] + lx * shifted[1])
                triples.append(tuple(triple))
    return tuple(dict.fromkeys(triples))


def ccz_generator_tensor(pair: BBTPair) -> np.ndarray:
    """CCZ action on logical-X rows followed by independent X-check rows."""
    rows = pair.source.generator_matrix
    tensor = np.zeros((rows.shape[0],) * 3, dtype=np.int64)
    for a, b, c in ccz_triples_for_pair(pair):
        tensor ^= (
            rows[:, a, None, None]
            * rows[None, :, b, None]
            * rows[None, None, :, c]
        )
    return tensor


def ccz_logical_tensor(pair: BBTPair) -> np.ndarray:
    tensor = np.zeros((pair.source.k,) * 3, dtype=np.int64)
    for a in range(pair.source.k):
        for b in range(pair.source.k):
            for c in range(pair.source.k):
                tensor[a, b, c] = sum(
                    pair.source.x_logicals[a, i]
                    * pair.source.x_logicals[b, j]
                    * pair.source.x_logicals[c, k]
                    for i, j, k in ccz_triples_for_pair(pair)
                ) % 2
    return tensor


def switch_map_commutes(pair: BBTPair) -> bool:
    matrix = switch_matrix_3d_to_2d(pair)
    x_checks_map = all(
        f2_in_row_span(row @ matrix.T % 2, pair.target.hx)
        for row in pair.source.hx
    )
    z_checks_map = all(
        f2_in_row_span(row @ matrix % 2, pair.source.hz)
        for row in pair.target.hz
    )
    return x_checks_map and z_checks_map


def switch_logical_map(pair: BBTPair) -> np.ndarray:
    """Logical CNOT incidence, with 3D controls and 2D targets."""
    matrix = switch_matrix_3d_to_2d(pair)
    return pair.source.x_logicals @ matrix.T @ pair.target.logical_zs.T % 2


def build_code_switching_case(
    pair: str | BBTPair,
    *,
    p_phys: float = 0.0,
    target_scoring: bool = True,
    deterministic_z_faults: Iterable[int] = (),
) -> CircuitCase:
    """Teleport three 2D blocks into 3D, then apply the physical CCZ layer."""
    pair = make_bbt_pair(pair) if isinstance(pair, str) else pair
    p_phys = float(p_phys)
    if not 0.0 <= p_phys <= 1.0:
        raise ValueError("p_phys must be in [0, 1]")
    source_offsets = tuple(block * pair.source.n for block in range(3))
    target_start = 3 * pair.source.n
    target_offsets = tuple(target_start + block * pair.target.n for block in range(3))
    n_phys = target_start + 3 * pair.target.n
    lines: list[str] = []

    plus_qubits = []
    for offset in source_offsets:
        plus_qubits.extend(pair.source.plus_qubits(offset))
    for offset in target_offsets:
        plus_qubits.extend(pair.target.plus_qubits(offset))
    if plus_qubits:
        lines.append("RX " + " ".join(map(str, plus_qubits)))
    for offset in source_offsets:
        append_ops(lines, pair.source.encoder_ops(offset))
    for offset in target_offsets:
        append_ops(lines, pair.target.encoder_ops(offset))

    switch_pairs = switch_cnot_pairs(pair)
    for block in range(3):
        for source_qubit, target_qubit in switch_pairs:
            source = source_offsets[block] + source_qubit
            target = target_offsets[block] + target_qubit
            lines.append(f"CX {source} {target}")
        target_offset = target_offsets[block]
        lines.append(
            "M " + " ".join(str(target_offset + qubit) for qubit in range(pair.target.n))
        )
        for logical in range(pair.target.k):
            source_support = np.flatnonzero(pair.source.x_logicals[logical])
            for measured_qubit in np.flatnonzero(pair.target.logical_zs[logical]):
                lookback = pair.target.n - int(measured_qubit)
                for source_qubit in source_support:
                    lines.append(
                        f"CX rec[-{lookback}] {source_offsets[block] + int(source_qubit)}"
                    )

    source_qubits = tuple(range(3 * pair.source.n))
    if p_phys:
        lines.append(f"DEPOLARIZE1({p_phys:.17g}) " + " ".join(map(str, source_qubits)))
    forced = set(int(q) for q in deterministic_z_faults)
    unknown = forced.difference(source_qubits)
    if unknown:
        raise ValueError(f"deterministic fault qubits out of range: {sorted(unknown)}")
    for qubit in sorted(forced):
        lines.append(f"Z_ERROR(1) {qubit}")

    for a, b, c in ccz_triples_for_pair(pair):
        append_ccz(lines, source_offsets[0] + a, source_offsets[1] + b, source_offsets[2] + c)

    checks = []
    for offset in source_offsets:
        append_ops(lines, pair.source.decoder_ops(offset))
        checks.extend(pair.source.check_qubits(offset))
    for basis, qubit in checks:
        lines.append(f"{'MX' if basis == 'mx' else 'M'} {qubit}")
    append_detectors(lines, len(checks))

    output_qubits = tuple(offset + logical for offset in source_offsets for logical in range(pair.source.k))
    n_observables = 0
    if target_scoring:
        for a, b, c in np.argwhere(ccz_logical_tensor(pair)):
            append_ccz(
                lines,
                source_offsets[0] + int(a),
                source_offsets[1] + int(b),
                source_offsets[2] + int(c),
            )
        n_observables = append_x_observables(lines, output_qubits)

    return CircuitCase(
        case_id=pair.name,
        family="code-switching",
        circuit="\n".join(lines) + "\n",
        n_phys=n_phys,
        n_detectors=len(checks),
        n_observables=n_observables,
        p_phys=p_phys,
        noise_model="depolarizing_3d_data_pre_ccz",
        target_scoring=bool(target_scoring),
        metadata={
            "pair": pair.name,
            "n_phys_3d": pair.source.n,
            "n_phys_2d": pair.target.n,
            "k_3d": pair.source.k,
            "k_2d": pair.target.k,
            "switched_logicals": 3 * pair.target.k,
            "output_logicals": len(output_qubits),
            "source_distance": pair.source_distance,
            "target_distance": pair.target_distance,
            "ccz_triples": len(ccz_triples_for_pair(pair)),
            "switch_cnot_count": 3 * len(switch_pairs),
            "logical_ccz_count": int(np.count_nonzero(ccz_logical_tensor(pair))),
            "deterministic_faults": sorted(forced),
        },
    )
