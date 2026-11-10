import numpy as np

import approx_chol
from tests._laplacians import MatrixLike, grid_laplacian


def test_a_duck_csr_of_non_contiguous_arrays_factors_as_their_copies():
    indptr = np.array([0, 99, 2, 99, 4], dtype=np.uint32)[::2]
    indices = np.array([0, 99, 1, 99, 0, 99, 1], dtype=np.uint32)[::2]
    data = np.array([2.0, 0.0, -1.0, 0.0, -1.0, 0.0, 2.0], dtype=np.float64)[::2]

    assert not indptr.flags["C_CONTIGUOUS"]
    assert not indices.flags["C_CONTIGUOUS"]
    assert not data.flags["C_CONTIGUOUS"]

    b = np.array([1.0, -1.0])
    contiguous = (indptr.copy(), indices.copy(), data.copy())
    np.testing.assert_array_equal(
        approx_chol.factorize(MatrixLike(indptr, indices, data, (2, 2))).apply(b),
        approx_chol.factorize(MatrixLike(*contiguous, (2, 2))).apply(b),
    )


def test_a_duck_csr_of_scipys_arrays_factors_as_the_scipy_matrix():
    lap = grid_laplacian(8, 8)
    assert lap.indices.dtype == np.int32
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    raw = approx_chol.factorize(
        MatrixLike(lap.indptr, lap.indices, lap.data, lap.shape)
    )
    np.testing.assert_array_equal(raw.apply(b), approx_chol.factorize(lap).apply(b))
