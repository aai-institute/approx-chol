"""Type stubs for approx_chol."""

from typing import Any, final

import numpy as np
import numpy.typing as npt
from typing_extensions import disjoint_base

__all__ = ["Backend", "Config", "Factor", "factorize", "factorize_raw"]

@disjoint_base
class Backend:
    """Which factorization each block gets."""

    @final
    class Approximate(Backend):
        """Approximate elimination for every block."""

        __match_args__ = ()
        def __new__(cls) -> Backend.Approximate: ...

    @final
    class ExactBelow(Backend):
        """Exact dense Cholesky at or below ``max_dim``, approximate above.

        A frustrated block, whose signs no ±1 scaling folds, counts twice: its
        cover has two sheets. A ``max_dim`` of ``0`` selects ``Approximate``.
        """

        __match_args__ = ("max_dim",)
        def __new__(cls, max_dim: int) -> Backend.ExactBelow: ...
        @property
        def max_dim(self) -> int: ...

@final
class Config:
    """Configuration for approximate Cholesky factorization.

    ``seed`` keys every sampler stream. ``split`` is the AC2 multi-edge
    multiplicity. Splitting an edge fewer than twice is standard AC, so
    ``None``, ``0`` and ``1`` all select AC. ``backend`` picks which blocks
    are factored exactly.
    An argument left out or ``None`` keeps the default, which the getters report.
    """

    def __new__(
        cls,
        seed: int | None = ...,
        split: int | None = ...,
        backend: Backend | None = ...,
    ) -> Config: ...
    @property
    def seed(self) -> int: ...
    @property
    def split(self) -> int | None: ...
    @property
    def backend(self) -> Backend: ...

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
    def n_steps(self) -> int:
        """Number of elimination steps."""
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

        Part of the scipy ``LinearOperator`` duck-type interface: an ``(n, 1)``
        column returns an ``(n, 1)`` column.
        """
        ...

    def rmatvec(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the preconditioner transpose (alias for :meth:`solve`; symmetric).

        Part of the scipy ``LinearOperator`` duck-type interface: an ``(n, 1)``
        column returns an ``(n, 1)`` column.
        """
        ...

    def solve(self, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Solve LDL^T x = b, returning a new array of the original matrix dimension.

        Raises:
            ValueError: If ``b`` is not 1-D integer or float, or ``len(b) != n``.
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
            ValueError: If ``b`` is not 1-D integer or float, ``out`` is not
                contiguous, sizes are invalid,
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
            has a nonpositive diagonal on a nonzero row; an all-zero row solves to 0.

    Warns:
        RuntimeWarning: If a block falls short of diagonal dominance after
            scaling, so its deficit was clamped, or its scaling did not settle,
            so a singular block may have been grounded.
    """
    ...

def factorize_raw(
    row_ptrs: npt.ArrayLike,
    col_indices: npt.ArrayLike,
    values: npt.ArrayLike,
    n: int,
    config: Config | None = None,
) -> Factor:
    """Factorize an SDDM matrix from raw CSR arrays.

    Args:
        row_ptrs: CSR row pointer array, any integer dtype.
        col_indices: CSR column index array, any integer dtype.
        values: CSR value array, integer or floating-point.
        n: Matrix dimension.
        config: Optional factorization configuration.

    Raises:
        ValueError: If an array is not 1-D or has a dtype it cannot carry, the
            CSR structure is invalid, or the matrix is not symmetric or has a
            nonpositive diagonal on a nonzero row; an all-zero row solves to 0.

    Warns:
        RuntimeWarning: As :func:`factorize`.
    """
    ...
