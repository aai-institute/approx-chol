import re

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import approx_chol
from tests._laplacians import grid_laplacian, grid_sddm


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
def test_split_below_two_selects_standard_ac(split):
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0

    def solve(split):
        config = approx_chol.Config(split=split, exact_max_dim=0)
        return approx_chol.factorize(lap, config).solve(b)

    np.testing.assert_array_equal(solve(split), solve(None))
    assert not np.array_equal(solve(2), solve(None))


@pytest.mark.parametrize("split", [-1, 2**32])
def test_split_outside_u32_raises(split):
    with pytest.raises(OverflowError):
        approx_chol.Config(split=split)


def _u32(*xs):
    return np.array(xs, dtype=np.uint32)


def _f64(*xs):
    return np.array(xs, dtype=np.float64)


@pytest.mark.parametrize(
    ("row_ptrs", "col_indices", "values", "n", "message"),
    [
        pytest.param(
            _u32(0, 2, 4),
            _u32(0, 1, 0, 1),
            _f64(2, -1, -1),
            2,
            "col_indices and values have different lengths (4 != 3)",
            id="csr_new",
        ),
        pytest.param(
            _u32(0, 2, 4),
            _u32(0, 1, 0, 1),
            _f64(2, -1, -2, 2),
            2,
            "row 0 has an entry with no equal mirror",
            id="from_csr_csr",
        ),
        pytest.param(
            _u32(0, 1),
            _u32(0),
            _f64(-1),
            1,
            "vertex 0 has a nonpositive diagonal",
            id="from_csr_reduce",
        ),
    ],
)
def test_factorize_raw_names_the_defect(row_ptrs, col_indices, values, n, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        approx_chol.factorize_raw(row_ptrs, col_indices, values, n)


def test_factorize_names_the_defect_of_a_scipy_matrix():
    asymmetric = sp.csr_matrix(np.array([[2.0, -1.0], [0.0, 2.0]]))
    with pytest.raises(ValueError, match="^row 0 has an entry with no equal mirror$"):
        approx_chol.factorize(asymmetric)

    with pytest.raises(ValueError, match="^matrix must be square$"):
        approx_chol.factorize(sp.csr_matrix(np.ones((2, 3))))


@pytest.mark.parametrize(
    ("indptr_dtype", "indices_dtype"),
    [
        (np.int32, np.int32),
        (np.int64, np.int64),
        (np.uint32, np.uint32),
        (np.uint64, np.uint64),
        (np.int16, np.int16),
        (np.int32, np.int64),
        (np.int64, np.uint64),
    ],
)
def test_every_integer_index_dtype_factors_alike(indptr_dtype, indices_dtype):
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    expected = approx_chol.factorize(lap).solve(b)

    typed = MatrixLike(
        lap.indptr.astype(indptr_dtype),
        lap.indices.astype(indices_dtype),
        lap.data,
        lap.shape,
    )
    np.testing.assert_array_equal(approx_chol.factorize(typed).solve(b), expected)


def test_duck_typed_factorize_validates_indices_and_dimension():
    too_large_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 2**32 + 1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(
        ValueError, match="^column index out of bounds at position 1: 4294967297 >= 2$"
    ):
        approx_chol.factorize(too_large_idx)

    negative_idx = MatrixLike(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, -1, 0, 1], dtype=np.int64),
        np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64),
        (2, 2),
    )
    with pytest.raises(
        ValueError,
        match="^col_index value at position 1 cannot be represented as usize$",
    ):
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
    # A float index would truncate silently and a complex value drop its imaginary part.
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


def test_solve_and_solve_into_raise_value_error_for_shape():
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)

    with pytest.raises(ValueError, match="right-hand side has length 12, expected 2"):
        factor.solve(np.zeros(12))

    rhs = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="out length"):
        factor.solve_into(rhs, np.zeros(1))


def test_solve_into_solves_in_place_and_into_an_overlapping_view():
    row_ptrs, col_indices, values = _base_csr()
    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    rhs = np.array([1.0, -1.0])
    expected = factor.solve(rhs)

    in_place = rhs.copy()
    factor.solve_into(in_place, in_place)
    np.testing.assert_array_equal(in_place, expected)

    base = np.array([1.0, -1.0, 0.0])
    factor.solve_into(base[:-1], base[1:])
    np.testing.assert_array_equal(base[1:], expected)


def test_solve_takes_any_1d_array_like_as_the_stub_declares():
    lap = grid_laplacian(4, 4)
    factor = approx_chol.factorize(lap)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    expected = factor.solve(b)
    np.testing.assert_array_equal(factor.solve(b.tolist()), expected)
    np.testing.assert_array_equal(factor.matvec(b.astype(np.float32)), expected)
    block = np.column_stack([b, 2 * b])
    np.testing.assert_array_equal(
        sla.aslinearoperator(factor) @ block, np.column_stack([expected, 2 * expected])
    )


def test_matvec_takes_the_column_scipy_before_1_18_passes():
    lap = grid_laplacian(4, 4)
    factor = approx_chol.factorize(lap)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    np.testing.assert_array_equal(factor.matvec(b.reshape(-1, 1)), factor.solve(b))


def test_an_all_zero_row_factors_and_solves_to_zero():
    dead = sp.csr_matrix(np.array([[0.0, 0, 0], [0, 2, -1], [0, -1, 2]]))
    x = approx_chol.factorize(dead).solve(np.ones(3))
    np.testing.assert_array_equal(x, [0.0, 1.0, 1.0])


def test_a_workspace_from_a_smaller_factor_raises():
    approximate = approx_chol.Config(exact_max_dim=0)
    small = approx_chol.factorize(grid_sddm(2, 2), approximate)
    large = approx_chol.factorize(grid_sddm(16, 16), approximate)
    b = np.zeros(large.n)
    with pytest.raises(ValueError, match="workspace is from a smaller factor"):
        large.solve_into(b, np.empty_like(b), small.workspace())


def test_a_workspace_in_use_raises_instead_of_sharing():
    factor = approx_chol.factorize(grid_laplacian(4, 4))
    workspace = factor.workspace()
    out = np.empty(factor.n)

    class ReentrantRhs:
        def __array__(self, dtype=None, copy=None):
            factor.solve_into(np.zeros(factor.n), np.empty(factor.n), workspace)
            return np.zeros(factor.n)

    with pytest.raises(RuntimeError, match="borrowed"):
        factor.solve_into(ReentrantRhs(), out, workspace)


@pytest.mark.parametrize("cls", [approx_chol.Factor, approx_chol.Workspace])
def test_factor_and_workspace_cannot_be_constructed_directly(cls):
    with pytest.raises(TypeError, match="cannot create"):
        cls()
