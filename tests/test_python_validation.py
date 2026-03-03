import numpy as np
import pytest

from tests._ext_loader import load_extension_module


def _base_csr():
    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)
    return row_ptrs, col_indices, values


def test_config_is_strictly_validated():
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()

    with pytest.raises(ValueError, match="config.split must be >= 1"):
        ext.factorize_raw(row_ptrs, col_indices, values, 2, ext.Config(split=0))


def test_duck_typed_factorize_validates_indices_and_dimension():
    ext = load_extension_module()

    class MatrixLike:
        def __init__(self, indptr, indices, data, shape):
            self.indptr = indptr
            self.indices = indices
            self.data = data
            self.shape = shape

    valid = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    factor = ext.factorize(valid)
    assert factor.n >= 2

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


def test_solve_and_solve_into_raise_value_error_for_shape_and_overlap():
    ext = load_extension_module()
    row_ptrs, col_indices, values = _base_csr()
    factor = ext.factorize_raw(row_ptrs, col_indices, values, 2)

    rhs_too_long = np.zeros(factor.n + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="rhs length"):
        factor.solve(rhs_too_long)

    rhs = np.zeros(factor.n, dtype=np.float64)
    if rhs.size >= 2:
        rhs[0] = 1.0
        rhs[1] = -1.0

    out_too_short = np.zeros(max(0, factor.n - 1), dtype=np.float64)
    with pytest.raises(ValueError, match="out length"):
        factor.solve_into(rhs, out_too_short)

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, rhs)
