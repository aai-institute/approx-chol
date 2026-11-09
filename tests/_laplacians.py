"""Shared sparse-matrix fixtures for the Python test suite."""

import scipy.sparse as sp


def grid_laplacian(rows: int, cols: int) -> sp.csr_matrix:
    """Build a ``rows x cols`` 2D grid-graph Laplacian as a scipy CSR matrix."""
    n = rows * cols
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    def vid(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            v = vid(r, c)
            degree = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    u = vid(rr, cc)
                    row_idx.append(v)
                    col_idx.append(u)
                    data.append(-1.0)
                    degree += 1
            row_idx.append(v)
            col_idx.append(v)
            data.append(float(degree))

    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))


def frustrated_grid(rows: int, cols: int) -> sp.csr_matrix:
    """``grid_laplacian`` plus the identity, one edge's off-diagonal positive: the approximate arm solves its two-sheet cover through scratch."""
    matrix = (grid_laplacian(rows, cols) + sp.identity(rows * cols)).tolil()
    matrix[0, 1] = matrix[1, 0] = 1.0
    return matrix.tocsr()


def light_row_grid(k: int, light: float) -> sp.csr_matrix:
    """A frustrated ``k x k`` grid, every seventh edge positive, row 0's edges ``light`` and their ends grounded by ``0.01 * light`` each, as the Rust tests' ``light_row``: its surplus is ``~1e-3 * light`` of the trace."""
    right = [(v, v + 1) for v in range(k * k) if v % k + 1 < k]
    down = [(v, v + k) for v in range(k * (k - 1))]
    diagonal = [0.0] * (k * k)
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    for e, (a, b) in enumerate(right + down):
        weight = light if a < k else 1.0
        sign = 1.0 if e % 7 == 0 else -1.0
        row_idx += [a, b]
        col_idx += [b, a]
        data += [sign * weight, sign * weight]
        for v in (a, b):
            diagonal[v] += weight + (0.01 * weight if v < k else 0.0)
    row_idx += range(k * k)
    col_idx += range(k * k)
    data += diagonal
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(k * k, k * k))
