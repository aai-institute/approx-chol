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
    # Regression for issue #35. A small tridiagonal SDDM augments to a graph
    # small enough that AC is exact, so solve(b) must equal the dense M^-1 b.
    # The RHS has a non-zero sum -- exactly the case the old global zero-mean
    # recovery corrupted.
    m = np.array([[5.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 5.0]])
    a = sp.csr_matrix(m)
    b = np.array([1.0, 2.0, 3.0])  # sum = 6 != 0

    factor = approx_chol.factorize(a, approx_chol.Config(seed=0))
    x = factor.solve(b)

    assert np.linalg.norm(x - np.linalg.solve(m, b)) < 1e-9


def test_sddm_preconditioner_makes_cg_converge():
    # Regression for issue #35. SDDM = grid Laplacian + positive diagonal shift
    # (positive row sums -> Gremban augmentation). The RHS has a non-zero sum, so
    # grounding drives -sum(b) onto the auxiliary vertex. Under the old global
    # zero-mean recovery the preconditioner is inconsistent and CG stalls at
    # maxiter (verified: rel. residual ~1.9); with correct grounding it converges
    # in a handful of iterations.
    lap = grid_laplacian(20, 20)
    n = lap.shape[0]
    a = (lap + 0.5 * sp.eye(n, format="csr")).tocsr()
    b = np.ones(n, dtype=np.float64)  # sum = n != 0

    factor = approx_chol.factorize(a, approx_chol.Config(seed=0))
    x, info = spla.cg(a, b, M=factor, rtol=1e-8, atol=0.0, maxiter=5000)

    assert info == 0, f"preconditioned CG on SDDM did not converge, info={info}"
    assert _relative_residual(a, x, b) < 1e-6


def test_preconditioner_reduces_fixed_budget_residual():
    # Approximate Cholesky is an approximate inverse, so we do not assert exact
    # solves; instead we assert significantly faster residual decay under a
    # fixed CG iteration budget (conditioning proxy).
    a = grid_laplacian(20, 20)
    n = a.shape[0]
    b = np.zeros(n, dtype=np.float64)
    b[0] = 1.0
    b[-1] = -1.0
    budget = 15

    x_unpre, _ = spla.cg(a, b, rtol=0.0, atol=0.0, maxiter=budget)
    residual_unpre = _relative_residual(a, x_unpre, b)

    factor = approx_chol.factorize(a, approx_chol.Config(seed=0))
    x_pre, _ = spla.cg(a, b, M=factor, rtol=0.0, atol=0.0, maxiter=budget)
    residual_pre = _relative_residual(a, x_pre, b)

    assert residual_pre < residual_unpre * 0.1, (
        "expected preconditioned residual to be at least 10x smaller after the "
        f"same iteration budget, got pre={residual_pre:.3e}, "
        f"unpre={residual_unpre:.3e}"
    )
