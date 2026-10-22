import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import approx_chol

from tests._laplacians import grid_laplacian


def _relative_residual(a: sp.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ x - b) / np.linalg.norm(b))


def test_preconditioner_reduces_cg_iterations_to_tolerance():
    a = grid_laplacian(20, 20)
    n = a.shape[0]
    b = np.zeros(n, dtype=np.float64)
    b[0] = 1.0
    b[-1] = -1.0

    unpre_iters: list[int] = []
    x_unpre, info_unpre = spla.cg(
        a,
        b,
        rtol=1e-8,
        atol=0.0,
        maxiter=5000,
        callback=lambda _: unpre_iters.append(1),
    )
    assert info_unpre == 0, f"unpreconditioned CG did not converge, info={info_unpre}"

    factor = approx_chol.factorize(a, approx_chol.Config(seed=0))

    pre_iters: list[int] = []
    x_pre, info_pre = spla.cg(
        a,
        b,
        M=factor,
        rtol=1e-8,
        atol=0.0,
        maxiter=5000,
        callback=lambda _: pre_iters.append(1),
    )
    assert info_pre == 0, f"preconditioned CG did not converge, info={info_pre}"

    assert len(pre_iters) < len(unpre_iters), (
        f"expected preconditioned CG to need fewer iterations, "
        f"got pre={len(pre_iters)} vs unpre={len(unpre_iters)}"
    )
    assert _relative_residual(a, x_pre, b) <= _relative_residual(a, x_unpre, b) * 1.5


def test_sddm_solve_matches_dense_inverse():
    # #35: a non-zero-sum rhs on an SDDM small enough that AC is exact, which the old zero-mean recovery corrupted.
    m = np.array([[5.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 5.0]])
    a = sp.csr_matrix(m)
    b = np.array([1.0, 2.0, 3.0])  # sum = 6 != 0

    factor = approx_chol.factorize(
        a, approx_chol.Config(seed=0, backend=approx_chol.Backend.Approximate())
    )
    x = factor.solve(b)

    assert np.linalg.norm(x - np.linalg.solve(m, b)) < 1e-9


def test_sddm_preconditioner_makes_cg_converge():
    # #35: on this non-zero-sum rhs the old zero-mean recovery left CG stalled at maxiter (rel. residual ~1.9).
    lap = grid_laplacian(20, 20)
    n = lap.shape[0]
    a = (lap + 0.5 * sp.eye(n, format="csr")).tocsr()
    b = np.ones(n, dtype=np.float64)  # sum = n != 0

    factor = approx_chol.factorize(a, approx_chol.Config(seed=0))
    x, info = spla.cg(a, b, M=factor, rtol=1e-8, atol=0.0, maxiter=5000)

    assert info == 0, f"preconditioned CG on SDDM did not converge, info={info}"
    assert _relative_residual(a, x, b) < 1e-6
