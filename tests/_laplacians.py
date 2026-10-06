"""Shared sparse-matrix fixtures for the Python test suite."""

import numpy as np
import scipy.sparse as sp


def _path_laplacian(k: int) -> sp.dia_matrix:
    degree = np.full(k, 2.0)
    degree[0] -= 1
    degree[-1] -= 1
    return sp.diags([-np.ones(k - 1), degree, -np.ones(k - 1)], [-1, 0, 1])


def grid_laplacian(rows: int, cols: int) -> sp.csr_matrix:
    """Build a ``rows x cols`` 2D grid-graph Laplacian as a scipy CSR matrix."""
    return sp.kronsum(_path_laplacian(cols), _path_laplacian(rows), format="csr")
