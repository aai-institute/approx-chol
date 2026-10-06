"""Tests for the scipy LinearOperator duck-type interface on Factor."""

import numpy as np
import scipy.sparse as sp

import approx_chol

from tests._laplacians import grid_laplacian


def _sddm_matrix() -> sp.csr_matrix:
    """2x2 SDDM matrix with surplus on both rows."""
    return sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float64))


class TestFactorShapeAndDtype:
    def test_shape_is_n_by_n(self):
        a = _sddm_matrix()
        factor = approx_chol.factorize(a)
        assert factor.n == 2
        assert factor.shape == (2, 2)

    def test_dtype_is_float64(self):
        a = grid_laplacian(3, 3)
        factor = approx_chol.factorize(a)
        assert factor.dtype == np.float64


class TestMatvec:
    def test_matvec_equals_solve(self):
        a = grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        b = np.zeros(25)
        b[0] = 1.0
        b[-1] = -1.0
        np.testing.assert_array_equal(factor.matvec(b), factor.solve(b))

    def test_rmatvec_equals_matvec(self):
        a = grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        b = np.zeros(25)
        b[0] = 1.0
        b[-1] = -1.0
        np.testing.assert_array_equal(factor.rmatvec(b), factor.matvec(b))
