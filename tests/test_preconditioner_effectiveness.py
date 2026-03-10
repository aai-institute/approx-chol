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


def _relative_residual(a: sp.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ x - b) / np.linalg.norm(b))


def test_preconditioner_reduces_cg_iterations_to_tolerance():
    a = _grid_laplacian(20, 20)
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


def test_preconditioner_reduces_fixed_budget_residual():
    # Approximate Cholesky is an approximate inverse, so we do not assert exact
    # solves; instead we assert significantly faster residual decay under a
    # fixed CG iteration budget (conditioning proxy).
    a = _grid_laplacian(20, 20)
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
