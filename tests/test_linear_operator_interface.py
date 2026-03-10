"""Tests for the scipy LinearOperator duck-type interface on Factor."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import approx_chol


def _grid_laplacian(rows: int, cols: int) -> sp.csr_matrix:
    n = rows * cols
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    def vid(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            v = vid(r, c)
            degree = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    u = vid(rr, cc)
                    row_idx.append(v)
                    col_idx.append(u)
                    data.append(-1.0)
                    degree += 1
            row_idx.append(v)
            col_idx.append(v)
            data.append(float(degree))

    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))


def _sddm_matrix() -> sp.csr_matrix:
    """2x2 SDDM matrix that triggers Gremban augmentation."""
    return sp.csr_matrix(
        np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float64)
    )


class TestFactorShapeAndDtype:
    def test_shape_matches_original_dimension_laplacian(self):
        a = _grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        assert factor.shape == (25, 25)

    def test_shape_matches_original_dimension_sddm(self):
        a = _sddm_matrix()
        factor = approx_chol.factorize(a)
        # Gremban augmentation adds a vertex: factor.n > 2
        assert factor.n > 2
        # But shape reflects the original 2x2 matrix
        assert factor.shape == (2, 2)

    def test_dtype_is_float64(self):
        a = _grid_laplacian(3, 3)
        factor = approx_chol.factorize(a)
        assert factor.dtype == np.float64


class TestMatvec:
    def test_matvec_returns_original_dimension(self):
        a = _sddm_matrix()
        factor = approx_chol.factorize(a)
        x = np.array([1.0, -1.0])
        result = factor.matvec(x)
        assert result.shape == (2,)

    def test_matvec_equals_solve(self):
        a = _grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        b = np.zeros(25)
        b[0] = 1.0
        b[-1] = -1.0
        np.testing.assert_array_equal(factor.matvec(b), factor.solve(b))

    def test_rmatvec_equals_matvec(self):
        a = _grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        b = np.zeros(25)
        b[0] = 1.0
        b[-1] = -1.0
        np.testing.assert_array_equal(factor.rmatvec(b), factor.matvec(b))


class TestSolveReturnsDimension:
    def test_solve_returns_original_n_for_laplacian(self):
        a = _grid_laplacian(4, 4)
        factor = approx_chol.factorize(a)
        b = np.zeros(16)
        b[0] = 1.0
        b[-1] = -1.0
        x = factor.solve(b)
        assert x.shape == (16,)

    def test_solve_returns_original_n_for_sddm(self):
        a = _sddm_matrix()
        factor = approx_chol.factorize(a)
        b = np.array([1.0, -1.0])
        x = factor.solve(b)
        # Must be original dimension (2), not augmented (3)
        assert x.shape == (2,)


class TestAsLinearOperator:
    def test_aslinearoperator_succeeds(self):
        a = _grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        M = spla.aslinearoperator(factor)
        assert M.shape == (25, 25)
        assert M.dtype == np.float64

    def test_aslinearoperator_matvec(self):
        a = _grid_laplacian(5, 5)
        factor = approx_chol.factorize(a)
        M = spla.aslinearoperator(factor)
        b = np.zeros(25)
        b[0] = 1.0
        b[-1] = -1.0
        result = M.matvec(b)
        np.testing.assert_array_equal(result, factor.solve(b))

    def test_direct_use_in_cg(self):
        a = _grid_laplacian(10, 10)
        n = a.shape[0]
        b = np.zeros(n, dtype=np.float64)
        b[0] = 1.0
        b[-1] = -1.0

        factor = approx_chol.factorize(a, approx_chol.Config(seed=42))
        x, info = spla.cg(a, b, M=factor, rtol=1e-8, atol=0.0, maxiter=500)
        assert info == 0, f"CG did not converge, info={info}"
        residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
        assert residual < 1e-7

    def test_direct_use_in_cg_with_sddm(self):
        """SDDM matrix with Gremban augmentation works directly as M=factor."""
        a = _sddm_matrix()
        n = a.shape[0]
        b = np.array([1.0, -1.0])

        factor = approx_chol.factorize(a, approx_chol.Config(seed=0))
        assert factor.shape == (n, n)

        x, info = spla.cg(a, b, M=factor, rtol=1e-8, atol=0.0, maxiter=100)
        assert info == 0, f"CG did not converge, info={info}"
        residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
        assert residual < 1e-7
