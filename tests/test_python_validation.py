import numpy as np
import pytest

import approx_chol


def _base_csr():
    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)
    return row_ptrs, col_indices, values


class MatrixLike:
    def __init__(self, indptr, indices, data, shape):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape


def test_split_below_two_is_accepted_as_standard_ac():
    row_ptrs, col_indices, values = _base_csr()

    for split in (None, 0, 1):
        factor = approx_chol.factorize_raw(
            row_ptrs, col_indices, values, 2, approx_chol.Config(split=split)
        )
        assert factor.shape == (2, 2)


def test_duck_typed_factorize_validates_indices_and_dimension():
    valid = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    factor = approx_chol.factorize(valid)
    assert factor.shape == (2, 2)

    too_large_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 2**32 + 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(ValueError, match="indices exceeds u32::MAX"):
        approx_chol.factorize(too_large_idx)

    negative_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, -1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(ValueError, match="indices must be non-negative"):
        approx_chol.factorize(negative_idx)

    oversized_dim = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2**32 + 1, 2**32 + 1),
    )
    with pytest.raises(ValueError, match="matrix dimension exceeds u32::MAX"):
        approx_chol.factorize(oversized_dim)


def test_each_column_rejects_the_dtype_kinds_it_cannot_carry():
    # An index column casts to uint32 and a value column to float64, so a float
    # index would truncate silently and a complex value would drop its imaginary
    # part. Each names only the kinds it accepts, and both share the rank check.
    row_ptrs, col_indices, values = _base_csr()

    float_index = MatrixLike(row_ptrs.astype(np.float64), col_indices, values, (2, 2))
    with pytest.raises(ValueError, match="indptr must have an integer dtype"):
        approx_chol.factorize(float_index)

    complex_value = MatrixLike(
        row_ptrs, col_indices, values.astype(np.complex128), (2, 2)
    )
    with pytest.raises(
        ValueError, match="data must have an integer or floating-point dtype"
    ):
        approx_chol.factorize(complex_value)

    two_dimensional = MatrixLike(row_ptrs, col_indices.reshape(2, 2), values, (2, 2))
    with pytest.raises(ValueError, match="indices must be a 1-D array"):
        approx_chol.factorize(two_dimensional)


def test_solve_and_solve_into_raise_value_error_for_shape_and_overlap():
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]

    for wrong_length in (original_n - 1, original_n + 1):
        with pytest.raises(ValueError, match="differs from"):
            factor.solve(np.zeros(wrong_length, dtype=np.float64))

    rhs = np.zeros(original_n, dtype=np.float64)
    rhs[0] = 1.0
    rhs[1] = -1.0

    out_too_short = np.zeros(original_n - 1, dtype=np.float64)
    with pytest.raises(ValueError, match="out length"):
        factor.solve_into(rhs, out_too_short)

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, rhs)


def test_solve_into_rejects_partially_overlapping_views():
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]

    base = np.zeros(original_n + 1, dtype=np.float64)
    rhs = base[:-1]
    out = base[1:]

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, out)
