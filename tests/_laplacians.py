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
