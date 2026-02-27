"""Type stubs for approx_chol."""

from typing import Any

import numpy as np
import numpy.typing as npt

class Config:
    """Configuration for approximate Cholesky factorization.

    Args:
        seed: Random seed for the edge-weight sampler.
        split: Number of edge copies for AC2 (None = standard AC, >=2 enables AC2).
        merge: Maximum multi-edges per neighbor pair for AC2.
            Requires ``split`` and defaults to ``split`` if not provided.
            When ``split == 1``, ``merge`` must be omitted or set to 1.
    """

    seed: int
    split: int | None
    merge: int | None

    def __init__(
        self,
        seed: int = 0,
        split: int | None = None,
        merge: int | None = None,
    ) -> None: ...

class Factor:
    """Approximate Cholesky factor (LDL^T decomposition).

    Obtained from :func:`factorize` or :func:`factorize_raw`.
    """

    @property
    def n(self) -> int:
        """Matrix dimension (may include Gremban augmentation vertex)."""
        ...

    @property
    def n_steps(self) -> int:
        """Number of elimination steps."""
        ...

    def solve(self, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Solve LDL^T x = b, returning a new array.

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

        Raises:
            ValueError: If ``b``/``out`` are not contiguous, sizes are invalid,
                or ``b`` and ``out`` overlap in memory.
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
