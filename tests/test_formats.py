import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import MatrixLike, bits, grid_laplacian

FORMATS = ["csr", "csc", "coo", "dok", "lil", "bsr", "dia"]


def _within_ulps_of_symmetric():
    """A grounded grid whose upper triangle is moved by up to 7 ulps: Rust accepts it, and its transpose is another matrix."""
    lap = (grid_laplacian(12, 12) + 0.1 * sp.identity(144)).tocsr()
    upper = sp.triu(lap, 1).tocoo()
    steps = np.random.default_rng(0).integers(1, 8, upper.nnz)
    moved = upper.data.copy()
    for step in range(7):
        moved = np.where(steps > step, np.nextafter(moved, 0.0), moved)
    moved_upper = sp.coo_array((moved, (upper.row, upper.col)), shape=lap.shape)
    return sp.csr_array(sp.diags(lap.diagonal()) + moved_upper + upper.T)


MATRIX = _within_ulps_of_symmetric()
B = np.sin(np.arange(MATRIX.shape[0], dtype=np.float64))


def _solve(matrix):
    return approx_chol.factorize(
        matrix, config=approx_chol.Config(exact_max_dim=0)
    ).apply(B)


def test_a_csc_factors_its_own_matrix_where_its_arrays_as_a_csr_are_its_transpose():
    """A CSC has ``indptr``, ``indices`` and ``data`` too: scipy's ``tocsr()`` reads it before them."""
    assert (MATRIX != MATRIX.T).nnz > 0
    csc = sp.csc_array(MATRIX)
    duck = MatrixLike(csc.indptr, csc.indices, csc.data, csc.shape)
    transpose = bits(_solve(sp.csr_array(MATRIX.T)))
    np.testing.assert_array_equal(bits(_solve(csc)), bits(_solve(MATRIX)))
    np.testing.assert_array_equal(bits(_solve(duck)), transpose)
    assert not np.array_equal(bits(_solve(MATRIX)), transpose)


@pytest.mark.parametrize("kind", [sp.csr_array, sp.csr_matrix])
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_scipy_format_factors_as_its_csr(fmt, kind):
    matrix = kind(MATRIX).asformat(fmt)
    np.testing.assert_array_equal(bits(_solve(matrix)), bits(_solve(MATRIX)))


@pytest.mark.parametrize("kind", [sp.bsr_array, sp.bsr_matrix])
def test_a_bsr_of_padded_blocks_factors_as_its_csr(kind):
    """3 x 3 blocks over a grid store zeros beside its entries, which ``tocsr()`` keeps."""
    bsr = kind(MATRIX, blocksize=(3, 3))
    assert bsr.tocsr().nnz > MATRIX.nnz
    np.testing.assert_array_equal(bits(_solve(bsr)), bits(_solve(MATRIX)))


class _NoData:
    indptr = np.array([0, 1])
    indices = np.array([0])
    shape = (1, 1)


@pytest.mark.parametrize(
    "matrix",
    [np.eye(2), [[1.0]], None, _NoData()],
    ids=["dense", "list", "none", "no_data"],
)
def test_anything_neither_scipy_sparse_nor_a_csr_raises_type_error(matrix):
    with pytest.raises(
        TypeError,
        match="^matrix must be a scipy sparse matrix or have indptr, indices, data and shape, not ",
    ):
        approx_chol.factorize(matrix)


def test_a_converted_matrix_reports_positions_in_the_csr_read():
    coo = sp.coo_array(([np.nan, 2.0, 2.0], ([1, 0, 1], [1, 0, 1])), shape=(2, 2))
    with pytest.raises(approx_chol.InputError) as caught:
        approx_chol.factorize(coo)
    csr = coo.tocsr()
    assert np.isnan(csr.data[caught.value.position])
    assert (caught.value.kind, caught.value.parameter) == ("out_of_range", "data")
