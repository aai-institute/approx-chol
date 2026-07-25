"""Type stubs for approx_chol."""

from typing import Any

import numpy as np
import numpy.typing as npt

class ExactFailure:
    """What to do when a block's exact Cholesky hits an invalid pivot."""

    FallBackToApproximate: ExactFailure
    Error: ExactFailure

class Backend:
    """Which factorization to run on each connected block.

    Selecting a backend does not change which inputs are accepted: a block whose
    exact factorization hits an invalid pivot falls back to approximate
    elimination, and a ``RuntimeWarning`` is emitted.
    """

    class Approximate:
        """Approximate elimination for every block."""

        def __init__(self) -> None: ...

    class ExactBelow:
        """Exact Cholesky for blocks of at most ``max_dim`` vertices.

        Exact factorization of a block costs ``O(max_dim**2)`` memory and
        ``O(max_dim**3)`` time; ``max_dim`` is neither capped nor validated.
        """

        max_dim: int
        on_failure: ExactFailure

        def __init__(self, max_dim: int, on_failure: ExactFailure) -> None: ...

class Config:
    """Configuration for approximate Cholesky factorization.

    Args:
        seed: Random seed for the edge-weight sampler.
        split: AC2 multi-edge multiplicity ``k``.
            ``None`` or ``1`` selects standard AC; ``>=2`` enables AC2.
        backend: Per-block factorization backend.
            ``None`` selects ``Backend.ExactBelow(max_dim=24)``.
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
        """Number of approximate elimination steps or exact dense pivots."""
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

    Warns:
        RuntimeWarning: If a block selected for exact Cholesky failed and fell
            back to approximate elimination, meaning the matrix is not positive
            definite within the tolerance it was accepted under.
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

    Warns:
        RuntimeWarning: If a block selected for exact Cholesky failed and fell
            back to approximate elimination, meaning the matrix is not positive
            definite within the tolerance it was accepted under.
    """
    ...
