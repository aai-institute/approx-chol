"""Type stubs for approx_chol."""

from typing import Any, final

import numpy as np
import numpy.typing as npt

__all__ = ["Config", "Factor", "factorize", "factorize_raw"]

@final
class Config:
    """Configuration for approximate Cholesky factorization.

    ``seed`` keys every sampler stream. ``split`` is the AC2 multi-edge
    multiplicity. Splitting an edge fewer than twice is standard AC, so
    ``None``, ``0`` and ``1`` all select AC. ``dense_up_to`` is the largest
    block dimension factored by exact dense Cholesky; ``0`` never does.
    An argument left out or ``None`` keeps the default, which the getters report.
    """

    def __new__(
        cls,
        seed: int | None = ...,
        split: int | None = ...,
        dense_up_to: int | None = ...,
    ) -> Config: ...
    @property
    def seed(self) -> int: ...
    @property
    def split(self) -> int | None: ...
    @property
    def dense_up_to(self) -> int: ...

@final
class Factor:
    """Approximate Cholesky factor (LDL^T decomposition).

    Obtained from :func:`factorize` or :func:`factorize_raw`.

    Implements the scipy ``LinearOperator`` duck-type interface (``shape``,
    ``matvec``, ``rmatvec``, ``dtype``), so it can be passed directly as
    ``M=factor`` to iterative solvers like ``scipy.sparse.linalg.cg``.
    """

    @property
    def n(self) -> int:
        """Matrix dimension; a Gremban ground vertex stays internal.

        Use this to size output buffers for :meth:`solve_into`.
        """
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Preconditioner shape ``(n, n)`` reflecting the original matrix dimension.

        Part of the scipy ``LinearOperator`` duck-type interface.
        """
        ...

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Numpy dtype of output arrays (``numpy.float64``).

        Part of the scipy ``LinearOperator`` duck-type interface.
        """
        ...

    def matvec(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the preconditioner (alias for :meth:`solve`).

        Part of the scipy ``LinearOperator`` duck-type interface.
        """
        ...

    def rmatvec(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the preconditioner transpose (alias for :meth:`solve`; symmetric).

        Part of the scipy ``LinearOperator`` duck-type interface.
        """
        ...

    def solve(self, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Solve LDL^T x = b, returning a new array of the original matrix dimension.

        Raises:
            ValueError: If ``b`` is not contiguous or ``len(b) != n``.
        """
        ...

    def solve_into(
        self,
        b: npt.ArrayLike,
        out: npt.NDArray[np.float64],
    ) -> None:
        """Solve LDL^T x = b, writing the result into *out*.

        ``b`` must have length ``shape[0]``, and *out* at least that length.

        Raises:
            ValueError: If ``b``/``out`` are not contiguous, sizes are invalid,
                ``out`` is not writeable, or ``b`` and ``out`` overlap in memory.
            BufferError: If ``out`` is already borrowed.
        """
        ...

def factorize(
    matrix: Any,
    config: Config | None = None,
) -> Factor:
    """Factorize an SDDM matrix from a scipy.sparse CSR matrix.

    Args:
        matrix: A ``scipy.sparse.csr_matrix`` or ``csr_array``.
        config: Optional factorization configuration.

    Raises:
        ValueError: If the matrix is not square or symmetric, has invalid
            dtypes/ranges, exceeds index limits, has invalid CSR structure, or
            has a nonpositive diagonal.

    Warns:
        RuntimeWarning: If a block falls short of diagonal dominance after
            scaling, so its deficit was clamped.
    """
    ...

def factorize_raw(
    row_ptrs: npt.NDArray[np.uint32],
    col_indices: npt.NDArray[np.uint32],
    values: npt.NDArray[np.float64],
    n: int,
    config: Config | None = None,
) -> Factor:
    """Factorize an SDDM matrix from raw CSR arrays.

    Args:
        row_ptrs: CSR row pointer array (uint32).
        col_indices: CSR column index array (uint32).
        values: CSR value array (float64).
        n: Matrix dimension.
        config: Optional factorization configuration.

    Raises:
        ValueError: If the CSR structure is invalid, or the matrix is not
            symmetric or has a nonpositive diagonal.

    Warns:
        RuntimeWarning: As :func:`factorize`.
    """
    ...
