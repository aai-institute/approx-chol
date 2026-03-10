"""Using approx-chol as a preconditioner for scipy's conjugate gradient solver.

The Factor returned by approx_chol.factorize() implements the scipy
LinearOperator duck-type interface (shape, matvec, dtype), so it can be
passed directly as ``M=factor`` to scipy.sparse.linalg.cg and other
iterative solvers — no manual wrapping needed.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import approx_chol


def grid_laplacian(rows: int, cols: int) -> sp.csr_matrix:
    """Build the graph Laplacian of a 2D grid."""
    n = rows * cols
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            degree = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    row_idx.append(v)
                    col_idx.append(rr * cols + cc)
                    data.append(-1.0)
                    degree += 1
            row_idx.append(v)
            col_idx.append(v)
            data.append(float(degree))

    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))


def main() -> None:
    A = grid_laplacian(50, 50)
    n = A.shape[0]

    # RHS in the range of the Laplacian (must sum to zero).
    b = np.zeros(n, dtype=np.float64)
    b[0] = 1.0
    b[-1] = -1.0

    # --- Unpreconditioned CG ---
    unpre_iters = 0

    def count_unpre(_: object) -> None:
        nonlocal unpre_iters
        unpre_iters += 1

    x_unpre, info = spla.cg(A, b, rtol=1e-10, atol=0.0, maxiter=10_000, callback=count_unpre)
    assert info == 0, f"unpreconditioned CG did not converge (info={info})"
    res_unpre = np.linalg.norm(A @ x_unpre - b) / np.linalg.norm(b)

    # --- Preconditioned CG (M=factor, no wrapping needed) ---
    factor = approx_chol.factorize(A, approx_chol.Config(seed=42))

    pre_iters = 0

    def count_pre(_: object) -> None:
        nonlocal pre_iters
        pre_iters += 1

    x_pre, info = spla.cg(A, b, M=factor, rtol=1e-10, atol=0.0, maxiter=10_000, callback=count_pre)
    assert info == 0, f"preconditioned CG did not converge (info={info})"
    res_pre = np.linalg.norm(A @ x_pre - b) / np.linalg.norm(b)

    print(f"Grid size:              {50}x{50} ({n} vertices)")
    print(f"Unpreconditioned CG:    {unpre_iters} iterations, residual {res_unpre:.2e}")
    print(f"Preconditioned CG:      {pre_iters} iterations, residual {res_pre:.2e}")
    print(f"Speedup:                {unpre_iters / pre_iters:.1f}x fewer iterations")


if __name__ == "__main__":
    main()
