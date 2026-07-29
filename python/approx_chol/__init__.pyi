"""Type stubs for approx_chol."""

from typing import Any

import numpy as np
import numpy.typing as npt

class ExactFailure:
    """What to do about a block whose exact Cholesky reaches an unusable pivot."""

    FallBackToApproximate: ExactFailure
    Error: ExactFailure

class Backend:
    """Which factorization each connected block gets."""

    class Approximate(Backend): ...

    class ExactBelow(Backend):
        """Exact dense Cholesky at or below ``max_dim`` solved variables.

        A ``max_dim`` of ``0`` claims no block, so it selects ``Approximate``.
        """

        max_dim: int
        on_failure: ExactFailure
        def __init__(self, max_dim: int, on_failure: ExactFailure) -> None: ...

class DenseFailure:
    """Why an exact dense Cholesky pivot was unusable."""

    NonPositivePivot: DenseFailure
    NonFinitePivot: DenseFailure
    Unknown: DenseFailure

class Fallback:
    """Why a block routed to exact Cholesky was factored approximately instead."""

    class InvalidPivot(Fallback):
        vertex: int
        failure: DenseFailure

    class WillNotFit(Fallback):
        dim: int

    class Other(Fallback):
        reason: str

class Config:
    """Configuration for approximate Cholesky factorization.

    ``split`` is the AC2 multi-edge multiplicity. Splitting an edge fewer than
    twice is standard AC, so ``None``, ``0`` and ``1`` all select AC.
    """

    seed: int
    split: int | None
    backend: Backend

    def __init__(
        self,
        seed: int = 0,
        split: int | None = None,
        backend: Backend | None = None,
    ) -> None: ...

class Factor:
    """Approximate Cholesky factor (LDL^T decomposition).

    Obtained from :func:`factorize` or :func:`factorize_raw`.

    Implements the scipy ``LinearOperator`` duck-type interface (``shape``,
    ``matvec``, ``rmatvec``, ``dtype``), so it can be passed directly as
    ``M=factor`` to iterative solvers like ``scipy.sparse.linalg.cg``.
    """

    @property
    def n(self) -> int:
        """Internal factor dimension (may include Gremban augmentation vertex).

        Use this to size work buffers for :meth:`solve_into`.
        """
        ...

    @property
    def n_steps(self) -> int:
        """Number of elimination steps."""
        ...

    @property
    def fallbacks(self) -> list[Fallback]:
        """Blocks routed to exact Cholesky and factored approximately anyway.

        Non-empty means the factor is less accurate than the configured backend
        asked for; :func:`factorize` also emits a ``RuntimeWarning``.
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
            ValueError: If ``b`` is not contiguous or ``len(b) > n``.
        """
        ...

    def solve_into(
        self,
        b: npt.ArrayLike,
        out: npt.NDArray[np.float64],
    ) -> None:
        """Solve LDL^T x = b, writing the result into *out*.

        The *out* array must have length >= ``shape[0]`` (the original matrix
        dimension).

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
        ValueError: If the matrix is not square, has invalid dtypes/ranges,
            exceeds index limits, or has invalid CSR structure.
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
        ValueError: If the CSR structure or config is invalid.
    """
    ...
