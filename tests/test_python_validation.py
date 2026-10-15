import numpy as np
import pytest
import scipy.sparse as sp

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


@pytest.mark.parametrize("split", [0, 1])
def test_split_below_two_is_rejected(split):
    with pytest.raises(ValueError, match=f"split must be at least 2, got {split}"):
        approx_chol.Config(split=split)


def _u32(*xs):
    return np.array(xs, dtype=np.uint32)


def _f64(*xs):
    return np.array(xs, dtype=np.float64)


@pytest.mark.parametrize(
    ("row_ptrs", "col_indices", "values", "n", "message"),
    [
        pytest.param(
            _u32(0, 2),
            _u32(0, 1),
            _f64(2, -1),
            2,
            "CSR array has length 2, expected 3",
            id="row_ptrs_length",
        ),
        pytest.param(
            _u32(0, 2, 4),
            _u32(0, 1, 0, 1),
            _f64(2, -1, -1),
            2,
            "CSR array has length 3, expected 4",
            id="values_length",
        ),
        pytest.param(
            _u32(0, 3, 2, 4),
            _u32(0, 1, 0, 1),
            _f64(2, -1, -1, 2),
            3,
            "row pointers do not ascend from 0 at row 1",
            id="falling_row_ptrs",
        ),
        pytest.param(
            _u32(0, 2, 4),
            _u32(0, 2, 0, 1),
            _f64(2, -1, -1, 2),
            2,
            "row 0 has a column index outside 0..n",
            id="column_out_of_range",
        ),
        pytest.param(
            _u32(0, 2, 4),
            _u32(0, 1, 0, 1),
            _f64(2, -1, -2, 2),
            2,
            "row 0 has an entry with no equal mirror",
            id="not_symmetric",
        ),
        pytest.param(
            _u32(0, 1),
            _u32(0),
            _f64(np.nan),
            1,
            "row 0 has an entry outside the representable range",
            id="nan_entry",
        ),
        pytest.param(
            _u32(0, 1),
            _u32(0),
            _f64(-1),
            1,
            "vertex 0 has a nonpositive diagonal",
            id="negative_diagonal",
        ),
        pytest.param(
            _u32(0, 1),
            _u32(0),
            _f64(np.finfo(np.float64).max),
            1,
            "an entry at vertex 0 is outside the representable range",
            id="diagonal_past_a_quarter_of_the_maximum",
        ),
    ],
)
def test_factorize_raw_names_the_defect(row_ptrs, col_indices, values, n, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        approx_chol.factorize_raw(row_ptrs, col_indices, values, n)


def test_factorize_names_the_defect_of_a_scipy_matrix():
    asymmetric = sp.csr_matrix(np.array([[2.0, -1.0], [0.0, 2.0]]))
    with pytest.raises(ValueError, match="^row 0 has an entry with no equal mirror$"):
        approx_chol.factorize(asymmetric)

    with pytest.raises(ValueError, match="^matrix must be square$"):
        approx_chol.factorize(sp.csr_matrix(np.ones((2, 3))))


def test_duck_typed_factorize_validates_indices_and_dimension():
    valid = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    factor = approx_chol.factorize(valid)
    assert factor.shape[0] >= 2

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

    rhs_too_long = np.zeros(original_n + 10, dtype=np.float64)
    with pytest.raises(ValueError, match="right-hand side has length 12, expected 2"):
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
    # SDDM input gains an internal ground slot, which a RHS must not reach.
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]
    assert factor.n == original_n, "the ground slot stays internal"

    rhs_aug_len = np.zeros(original_n + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="right-hand side has length 3, expected 2"):
        factor.solve(rhs_aug_len)
    out = np.zeros(original_n, dtype=np.float64)
    with pytest.raises(ValueError, match="right-hand side has length 3, expected 2"):
        factor.solve_into(rhs_aug_len, out)


def test_solve_into_rejects_partially_overlapping_views():
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    original_n = factor.shape[0]

    base = np.zeros(original_n + 1, dtype=np.float64)
    rhs = base[:-1]
    out = base[1:]

    with pytest.raises(ValueError, match="must not overlap"):
        factor.solve_into(rhs, out)
