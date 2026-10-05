"""Binary CSS-code helpers shared by benchmark protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_binary_matrix(matrix, *, n_cols=None, name="matrix"):
    result = np.asarray(matrix, dtype=np.int64)
    if result.ndim == 1:
        result = result.reshape(1, result.shape[0] if n_cols is None else n_cols)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary matrix")
    if n_cols is not None and result.shape[1] != int(n_cols):
        raise ValueError(f"{name} has {result.shape[1]} columns, expected {int(n_cols)}")
    if result.size and not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain only 0/1 entries")
    return result % 2


def f2_rref(rows, n=None):
    array = np.asarray(rows, dtype=np.int64) % 2
    if array.ndim == 1:
        array = array.reshape(1, array.shape[0] if n is None else n)
    if array.size == 0:
        width = int(n if n is not None else (array.shape[1] if array.ndim == 2 else 0))
        return np.zeros((0, width), dtype=np.int64), []
    array = array.copy()
    row = 0
    pivots = []
    for column in range(array.shape[1]):
        pivot = next((i for i in range(row, array.shape[0]) if array[i, column]), None)
        if pivot is None:
            continue
        array[[row, pivot]] = array[[pivot, row]]
        for i in range(array.shape[0]):
            if i != row and array[i, column]:
                array[i] ^= array[row]
        pivots.append(column)
        row += 1
    return array[:row], pivots


def f2_rank(rows, n=None):
    return len(f2_rref(rows, n=n)[1])


def f2_basis(rows, n=None):
    array = np.asarray(rows, dtype=np.int64) % 2
    if array.ndim == 1:
        array = array.reshape(1, array.shape[0] if n is None else n)
    width = int(n if n is not None else (array.shape[1] if array.ndim == 2 else 0))
    basis = []
    rank = 0
    for row in array:
        new_rank = f2_rank(basis + [row], n=width)
        if new_rank > rank:
            basis.append(row.copy())
            rank = new_rank
    return np.vstack(basis) % 2 if basis else np.zeros((0, width), dtype=np.int64)


def f2_in_row_span(vector, rows):
    vector = np.asarray(vector, dtype=np.int64) % 2
    rows = np.asarray(rows, dtype=np.int64) % 2
    return (not vector.any()) if not rows.size else f2_rank(rows) == f2_rank(np.vstack([rows, vector]))


def f2_nullspace(rows, n=None):
    array = np.asarray(rows, dtype=np.int64) % 2
    if array.ndim == 1:
        array = array.reshape(1, array.shape[0] if n is None else n)
    width = int(n if n is not None else (array.shape[1] if array.ndim == 2 else 0))
    rref, pivots = f2_rref(array, n=width)
    free = [column for column in range(width) if column not in set(pivots)]
    result = []
    for column in free:
        vector = np.zeros(width, dtype=np.int64)
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = rref[row, column]
        result.append(vector)
    return np.vstack(result) % 2 if result else np.zeros((0, width), dtype=np.int64)


def f2_solve(matrix, rhs):
    matrix = np.asarray(matrix, dtype=np.int64) % 2
    rhs = np.asarray(rhs, dtype=np.int64).reshape(-1) % 2
    if matrix.ndim != 2 or matrix.shape[0] != rhs.shape[0]:
        raise ValueError("matrix and rhs shape mismatch")
    augmented = np.hstack([matrix.copy(), rhs[:, None]])
    row = 0
    pivots = []
    for column in range(matrix.shape[1]):
        pivot = next((i for i in range(row, matrix.shape[0]) if augmented[i, column]), None)
        if pivot is None:
            continue
        augmented[[row, pivot]] = augmented[[pivot, row]]
        for i in range(matrix.shape[0]):
            if i != row and augmented[i, column]:
                augmented[i] ^= augmented[row]
        pivots.append(column)
        row += 1
    if any(not augmented[i, :-1].any() and augmented[i, -1] for i in range(row, matrix.shape[0])):
        raise ValueError("linear system has no F2 solution")
    solution = np.zeros(matrix.shape[1], dtype=np.int64)
    for i, pivot in enumerate(pivots):
        solution[pivot] = augmented[i, -1]
    return solution


def complete_basis(generator):
    generator = as_binary_matrix(generator, name="generator")
    rows = [row.copy() for row in generator]
    rank = f2_rank(rows, n=generator.shape[1])
    for qubit in range(generator.shape[1]):
        if rank == generator.shape[1]:
            break
        vector = np.zeros(generator.shape[1], dtype=np.int64)
        vector[qubit] = 1
        new_rank = f2_rank(rows + [vector], n=generator.shape[1])
        if new_rank > rank:
            rows.append(vector)
            rank = new_rank
    if rank != generator.shape[1]:
        raise ValueError("could not complete generator to a full basis")
    return np.vstack(rows) % 2


def _swap_ops(a, b):
    return [] if a == b else [("cnot", a, b), ("cnot", b, a), ("cnot", a, b)]


def encoder_ops_from_matrix(generator):
    basis = complete_basis(generator)
    matrix = basis.T.copy()
    reduce_ops = []
    row = 0
    for column in range(matrix.shape[1]):
        pivot = next((i for i in range(row, matrix.shape[0]) if matrix[i, column]), None)
        if pivot is None:
            raise ValueError("completed matrix unexpectedly singular")
        if pivot != row:
            matrix[[row, pivot]] = matrix[[pivot, row]]
            reduce_ops.extend(_swap_ops(row, pivot))
        for i in range(matrix.shape[0]):
            if i != row and matrix[i, column]:
                matrix[i] ^= matrix[row]
                reduce_ops.append(("cnot", row, i))
        row += 1
    return list(reversed(reduce_ops))


@dataclass
class CSSCode:
    name: str
    x_logicals: np.ndarray
    x_stabilizers: np.ndarray
    z_stabilizers: np.ndarray | None = None
    distance: int | None = None

    def __post_init__(self):
        self.x_logicals = as_binary_matrix(self.x_logicals, name="x_logicals")
        n = self.x_logicals.shape[1]
        self.x_stabilizers = as_binary_matrix(self.x_stabilizers, n_cols=n, name="x_stabilizers")
        self.x_stabilizer_basis = f2_basis(self.x_stabilizers, n=n)
        generator = np.vstack([self.x_logicals, self.x_stabilizer_basis]) % 2
        if f2_rank(generator, n=n) != generator.shape[0]:
            raise ValueError("logical X rows must be independent modulo X checks")
        self.z_stabilizers = (
            f2_nullspace(generator, n=n)
            if self.z_stabilizers is None
            else as_binary_matrix(self.z_stabilizers, n_cols=n, name="z_stabilizers")
        )
        if self.x_stabilizers.size and self.z_stabilizers.size and np.any(self.x_stabilizers @ self.z_stabilizers.T % 2):
            raise ValueError("CSS X and Z checks do not commute")
        k = n - f2_rank(self.x_stabilizers, n=n) - f2_rank(self.z_stabilizers, n=n)
        if k != self.x_logicals.shape[0]:
            raise ValueError("CSS dimension mismatch")
        self.n = int(n)
        self.k = int(self.x_logicals.shape[0])
        self.generator_matrix = generator
        self.n_encoder_rows = int(generator.shape[0])
        rows = np.vstack([self.x_stabilizer_basis, self.x_logicals])
        logical_zs = []
        for index in range(self.k):
            rhs = np.zeros(rows.shape[0], dtype=np.int64)
            rhs[self.x_stabilizer_basis.shape[0] + index] = 1
            logical_zs.append(f2_solve(rows, rhs))
        self.logical_zs = np.vstack(logical_zs) % 2

    @classmethod
    def from_checks(
        cls,
        name,
        x_stabilizers,
        z_stabilizers,
        *,
        preferred_x_logicals=(),
        distance=None,
    ):
        """Construct a CSS code and choose X logicals modulo its X checks.

        ``preferred_x_logicals`` is tried first.  This is useful for a code
        switch, where the target logical representatives should be retained
        under the physical chain map.  Remaining representatives are selected
        deterministically from the nullspace of the Z checks.
        """
        hx = as_binary_matrix(x_stabilizers, name="x_stabilizers")
        hz = as_binary_matrix(z_stabilizers, n_cols=hx.shape[1], name="z_stabilizers")
        n = hx.shape[1]
        if hx.size and hz.size and np.any(hx @ hz.T % 2):
            raise ValueError("CSS X and Z checks do not commute")

        x_check_basis = f2_basis(hx, n=n)
        span = [row.copy() for row in x_check_basis]
        rank = len(span)
        logicals = []
        preferred = np.asarray(preferred_x_logicals, dtype=np.int64)
        if preferred.size:
            preferred = as_binary_matrix(preferred, n_cols=n, name="preferred_x_logicals")
        else:
            preferred = np.zeros((0, n), dtype=np.int64)
        candidates = list(preferred) + list(f2_nullspace(hz, n=n))
        for candidate in candidates:
            if hz.size and np.any(hz @ candidate % 2):
                raise ValueError("preferred X logical does not commute with Z checks")
            new_rank = f2_rank(span + [candidate], n=n)
            if new_rank > rank:
                logicals.append(candidate.copy())
                span.append(candidate.copy())
                rank = new_rank

        k = n - f2_rank(hx, n=n) - f2_rank(hz, n=n)
        if len(logicals) != k:
            raise ValueError(f"could not find {k} independent X logicals")
        return cls(
            name,
            np.vstack(logicals) if logicals else np.zeros((0, n), dtype=np.int64),
            hx,
            hz,
            distance,
        )

    @property
    def hx(self):
        return self.x_stabilizers

    @property
    def hz(self):
        return self.z_stabilizers

    def encoder_ops(self, offset=0):
        return [(kind, a + int(offset), b + int(offset)) for kind, a, b in encoder_ops_from_matrix(self.generator_matrix)]

    def decoder_ops(self, offset=0):
        return list(reversed(self.encoder_ops(offset)))

    def plus_qubits(self, offset=0):
        return tuple(int(offset) + q for q in range(self.n_encoder_rows))

    def check_qubits(self, offset=0):
        off = int(offset)
        return (
            tuple(("mx", off + q) for q in range(self.k, self.n_encoder_rows))
            + tuple(("mz", off + q) for q in range(self.n_encoder_rows, self.n))
        )


def validate_triorthogonal_generator(generator, k_logical=None):
    generator = as_binary_matrix(generator, name="generator")
    if not generator.size or generator.shape[0] > generator.shape[1] or f2_rank(generator) != generator.shape[0]:
        raise ValueError("generator must be nonempty, independent, and have at most n rows")
    weights = generator.sum(axis=1) % 2
    if k_logical is None:
        k_logical = next((i for i, value in enumerate(weights) if value == 0), len(weights))
    k_logical = int(k_logical)
    if not (0 < k_logical <= generator.shape[0]) or not np.all(weights[:k_logical] == 1) or not np.all(weights[k_logical:] == 0):
        raise ValueError("logical rows must be the leading odd-weight rows")
    for i in range(generator.shape[0]):
        for j in range(i + 1, generator.shape[0]):
            if int(generator[i] @ generator[j]) % 2:
                raise ValueError("all row-pair overlaps must be even")
            for k in range(j + 1, generator.shape[0]):
                if int((generator[i] * generator[j]) @ generator[k]) % 2:
                    raise ValueError("all row-triple overlaps must be even")
    return generator, k_logical


def matrix_15to1():
    points = range(1, 16)
    return np.asarray([[1] * 15] + [[(value >> bit) & 1 for value in points] for bit in range(4)], dtype=np.int64)


def matrix_bravyi_haah_3k8(k):
    if k <= 0 or k % 2:
        raise ValueError("Bravyi-Haah 3k+8 requires positive even k")
    left = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int64)
    middle = np.asarray([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=np.int64)
    s1 = np.asarray([[0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)
    s2 = np.asarray([[1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=np.int64)
    blocks = k // 2
    odd = []
    for block in range(blocks):
        for row in range(2):
            odd.append(np.concatenate([np.zeros(4, dtype=np.int64), left[row]] + [middle[row] if j == block else np.zeros(6, dtype=np.int64) for j in range(blocks)]))
    even = [np.concatenate([s1[row], s1[row]] + [s2[row] for _ in range(blocks)]) for row in range(3)]
    return np.vstack(odd + even) % 2


def triorthogonal_css_code(generator, *, k_logical=None, name="triorthogonal"):
    generator, k_logical = validate_triorthogonal_generator(generator, k_logical)
    return CSSCode(name, generator[:k_logical], generator[k_logical:], f2_nullspace(generator))


def _cell_index(x, y, lx, ly):
    return int(x) % int(lx) + int(lx) * (int(y) % int(ly))


def _toric_difference_rows(lx, ly, sectors):
    cell_count = int(lx) * int(ly)
    rows = []
    for sector in range(int(sectors)):
        offset = sector * cell_count
        for y in range(int(ly)):
            for x in range(int(lx)):
                q = offset + _cell_index(x, y, lx, ly)
                for dx, dy in ((1, 0), (0, 1)):
                    row = np.zeros(int(sectors) * cell_count, dtype=np.int64)
                    row[q] = row[offset + _cell_index(x + dx, y + dy, lx, ly)] = 1
                    rows.append(row)
    return np.vstack(rows) % 2


def bbt_component_code(name, *, lx, ly, sectors, distance):
    cell_count = int(lx) * int(ly)
    logicals = []
    for sector in range(int(sectors)):
        row = np.zeros(int(sectors) * cell_count, dtype=np.int64)
        row[sector * cell_count:(sector + 1) * cell_count] = 1
        logicals.append(row)
    return CSSCode(name, np.vstack(logicals), _toric_difference_rows(lx, ly, sectors), np.zeros((0, int(sectors) * cell_count), dtype=np.int64), int(distance))
