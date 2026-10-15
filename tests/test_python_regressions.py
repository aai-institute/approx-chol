import numpy as np

import approx_chol
from tests._laplacians import grid_laplacian


def test_factorize_raw_copies_non_contiguous_arrays():
    row_ptrs = np.array([0, 99, 2, 99, 4], dtype=np.uint32)[::2]
    col_indices = np.array([0, 99, 1, 99, 0, 99, 1], dtype=np.uint32)[::2]
    values = np.array([2.0, 0.0, -1.0, 0.0, -1.0, 0.0, 2.0], dtype=np.float64)[::2]

    assert not row_ptrs.flags["C_CONTIGUOUS"]
    assert not col_indices.flags["C_CONTIGUOUS"]
    assert not values.flags["C_CONTIGUOUS"]

    b = np.array([1.0, -1.0])
    contiguous = (row_ptrs.copy(), col_indices.copy(), values.copy())
    np.testing.assert_array_equal(
        approx_chol.factorize_raw(row_ptrs, col_indices, values, 2).solve(b),
        approx_chol.factorize_raw(*contiguous, 2).solve(b),
    )


def test_factorize_raw_takes_scipy_arrays_as_factorize_does():
    lap = grid_laplacian(8, 8)
    assert lap.indices.dtype == np.int32
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    raw = approx_chol.factorize_raw(lap.indptr, lap.indices, lap.data, lap.shape[0])
    np.testing.assert_array_equal(raw.solve(b), approx_chol.factorize(lap).solve(b))
