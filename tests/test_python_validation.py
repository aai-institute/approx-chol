import numpy as np
import pytest

from tests._ext_loader import load_extension_module


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


def test_config_is_strictly_validated():
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()

    with pytest.raises(ValueError, match="config.split must be >= 1"):
        ext.factorize_raw(row_ptrs, col_indices, values, 2, ext.Config(split=0))


def test_duck_typed_factorize_validates_indices_and_dimension():
    ext = load_extension_module()

    valid = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    factor = ext.factorize(valid)
    assert factor.shape[0] >= 2

    too_large_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 2**32 + 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(ValueError, match="indices exceeds u32::MAX"):
        ext.factorize(too_large_idx)

    negative_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, -1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(ValueError, match="indices must be non-negative"):
        ext.factorize(negative_idx)

    oversized_dim = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2**32 + 1, 2**32 + 1),
    )
    with pytest.raises(ValueError, match="matrix dimension exceeds u32::MAX"):
        ext.factorize(oversized_dim)


def test_each_column_rejects_the_dtype_kinds_it_cannot_carry():
    # An index column casts to uint32 and a value column to float64, so a float
    # index would truncate silently and a complex value would drop its imaginary
    # part. Each names only the kinds it accepts, and both share the rank check.
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()

    float_index = MatrixLike(row_ptrs.astype(np.float64), col_indices, values, (2, 2))
    with pytest.raises(ValueError, match="indptr must have an integer dtype"):
        ext.factorize(float_index)

    complex_value = MatrixLike(
        row_ptrs, col_indices, values.astype(np.complex128), (2, 2)
    )
    with pytest.raises(
        ValueError, match="data must have an integer or floating-point dtype"
    ):
        ext.factorize(complex_value)

    two_dimensional = MatrixLike(row_ptrs, col_indices.reshape(2, 2), values, (2, 2))
    with pytest.raises(ValueError, match="indices must be a 1-D array"):
        ext.factorize(two_dimensional)


def test_solve_and_solve_into_raise_value_error_for_shape_and_overlap():
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()
    factor = ext.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]

    rhs_too_long = np.zeros(original_n + 10, dtype=np.float64)
    with pytest.raises(ValueError, match="rhs length"):
        factor.solve(rhs_too_long)

    rhs = np.zeros(original_n, dtype=np.float64)
    if rhs.size >= 2:
        rhs[0] = 1.0
        rhs[1] = -1.0

    out_too_short = np.zeros(max(0, original_n - 1), dtype=np.float64)
    with pytest.raises(ValueError, match="out length"):
        factor.solve_into(rhs, out_too_short)

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, rhs)


def test_solve_rejects_augmented_length_rhs():
    # _base_csr is SDDM (positive row sums), so it augments: original_n=2, n=3.
    # A RHS of length original_n + 1 (the augmented dimension) must be rejected,
    # not silently accepted with its trailing aux entry discarded.
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()
    factor = ext.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]
    assert factor.n == original_n + 1, "SDDM should augment by one vertex"

    rhs_aug_len = np.zeros(original_n + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="rhs length"):
        factor.solve(rhs_aug_len)
    out = np.zeros(original_n, dtype=np.float64)
    with pytest.raises(ValueError, match="rhs length"):
        factor.solve_into(rhs_aug_len, out)


def test_solve_into_rejects_partially_overlapping_views():
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()
    factor = ext.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]

    base = np.zeros(original_n + 1, dtype=np.float64)
    rhs = base[:-1]
    out = base[1:]

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, out)
