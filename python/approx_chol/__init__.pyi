"""Type stubs for approx_chol."""

from typing import Any, Literal, TypeAlias, final

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator
from typing_extensions import Never

__all__ = [
    "ApproxInverse",
    "Clamp",
    "Component",
    "Config",
    "ScalingPolicy",
    "factorize",
    "factorize_raw",
]

_Frustrated: TypeAlias = Literal[
    "cover", "cover_ground_vertex", "cover_surplus_edges", "signed"
]

@final
class Config:
    """Configuration for approximate Cholesky factorization.

    ``seed`` keys every sampler stream. ``split`` is the number of clique-tree
    copies each edge is sampled as, at least ``1``, costing ``O(split)`` sampler
    draws per edge: ``1`` is standard AC, more is AC2. ``exact_max_dim`` is the
    largest block dimension factored by exact dense Cholesky, a bipartite
    block's by its kept side; ``0`` factors every block approximately.
    ``frustrated`` is how a frustrated block, one no diagonal ``±1`` scaling
    makes an SDDM, is factored approximately: through its Gremban cover on two
    sheets, whose surplus goes to a ground vertex (``"cover_ground_vertex"``),
    onto an edge between each vertex's two copies (``"cover_surplus_edges"``),
    or onto those edges only inside the twist band of
    ``CoverSurplus::Auto`` (``"cover"``); or as it is on one sheet
    (``"signed"``), its fill signed by its ends and what opposite entries
    cancel grounded, so it never floats; a bipartite block's complement is
    formed signed and eliminated so. Any frustration grounds, however small:
    a cancellation adds ``2 min(|a|, |b|)`` to both ends without
    subtracting, so every pivot stays positive; below ``ε_mach·κ`` a solve is
    limited by conditioning, as under every route.
    An argument left out or ``None`` keeps the default, which the getters report.

    Raises:
        ValueError: If ``split`` is outside ``[1, 2**32 - 1]`` or ``frustrated``
            names no route.
        TypeError: If ``split`` is not an integer, or is a bool.
    """

    def __new__(
        cls,
        seed: int | None = ...,
        split: int | None = ...,
        exact_max_dim: int | None = ...,
        *,
        frustrated: _Frustrated | None = ...,
    ) -> Config: ...
    @property
    def seed(self) -> int: ...
    @property
    def split(self) -> int: ...
    @property
    def frustrated(self) -> _Frustrated: ...
    @property
    def exact_max_dim(self) -> int: ...

@final
class ScalingPolicy:
    """How a block short of diagonal dominance is scaled before it is factored.

    ``tolerance`` is the relative dominance violation, in ``[0, 1)``, a row may
    show and still count as dominant; rounding is always tolerated. ``max_iterations`` is the scaling budget a
    block may spend before it is reported unsettled. An argument left out or
    ``None`` keeps the default, which the getters report.

    Raises:
        ValueError: If ``tolerance`` is outside ``[0, 1)``.
    """

    def __new__(
        cls,
        tolerance: float | None = ...,
        max_iterations: int | None = ...,
    ) -> ScalingPolicy: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def max_iterations(self) -> int: ...

@final
class Clamp:
    """How the reduction clamped a component's rows past rounding to make it dominant.

    ``violation`` is the largest relative dominance violation clamped away,
    finite and positive; ``scaling`` is what the scaling proved about it:
    dominant within the policy's tolerance (``"certified"``), not dominant
    past it (``"refuted"``), or neither, the iteration stopped first
    (``"unsettled"``). Read-only, from :attr:`Component.clamp`.
    """

    def __new__(cls, never: Never, /) -> Never: ...
    @property
    def violation(self) -> float: ...
    @property
    def scaling(self) -> Literal["certified", "refuted", "unsettled"]: ...

@final
class Component:
    """One irreducible block of an :class:`ApproxInverse`, from :attr:`ApproxInverse.components`.

    ``vertices`` are its vertices in the factored matrix's numbering, in the
    order its factor numbers them, a read-only array; ``clamp`` is how the
    reduction clamped its rows, or ``None``.
    """

    def __new__(cls, never: Never, /) -> Never: ...
    @property
    def vertices(self) -> npt.NDArray[np.intp]: ...
    @property
    def clamp(self) -> Clamp | None: ...

@final
class ApproxInverse(LinearOperator):
    """An approximate inverse ``B`` of the factored matrix, from :func:`factorize` or :func:`factorize_raw`.

    A ``scipy.sparse.linalg.LinearOperator`` of shape ``(n, n)`` and dtype
    ``float64``, its own adjoint and transpose: ``matvec`` and ``rmatvec`` take
    a vector or a stack of them, ``(..., n)``, ``matmat`` and ``rmatmat``
    ``(..., n, k)``, so scipy's ``cg`` and ``minres`` take it as ``M``. ``lsmr``
    takes no preconditioner; right-precondition there: ``y`` from
    ``lsmr(aslinearoperator(A) @ B, b)``, then ``x = B @ y``.

    It holds a thread budget, :attr:`threads`, at which every apply runs.
    """

    # No Python constructor: the uninhabited argument makes every call a type error.
    def __init__(self, factor: Never, /) -> None: ...
    def __reduce__(self) -> tuple[object, ...]:
        """Pickles as the factor's saved form alone, its header first, encoded with postcard.

        A loaded or copied operator's budget is every core, ``threads=None``.

        Raises:
            ValueError: On unpickling, if the header's format version or scalar
                type is not this build's, naming each that differs, found and
                expected, or if the saved form is otherwise invalid.
        """

    @property
    def n(self) -> int:
        """Matrix dimension."""

    @property
    def n_steps(self) -> int:
        """Number of elimination steps."""

    @property
    def components(self) -> tuple[Component, ...]:
        """One :class:`Component` per irreducible block, by its lowest vertex, then the isolated vertices."""

    @property
    def threads(self) -> int | None:
        """The budget every apply runs at: the build's unless :meth:`with_threads` named another, ``None`` every core."""

    def with_threads(self, threads: int | None) -> ApproxInverse:
        """This operator at another budget, sharing the factor, copying nothing.

        Every budget applies to the same bits.

        Raises:
            ValueError: If ``threads`` is outside ``[1, 2**64 - 1]``.
            TypeError: If ``threads`` is not an integer, or is a bool.
        """

    def apply(
        self, b: npt.ArrayLike, *, out: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """Solve ``A x = b`` approximately, ``x = B b``, at :attr:`threads`.

        ``b`` is a real array ``np.asarray`` reads, a vector of length ``n`` or
        an ``n`` × ``k`` array, converted to ``float64`` if it is not; each
        column solves bit for bit as it would alone. The solution has ``b``'s
        shape, written to ``out`` when given and returned, else to a new
        array. ``out`` is a writable, aligned ``float64`` array of that shape
        in any order whose entries neither overlap nor interleave, each stride
        a whole number of entries, ``b`` itself or one overlapping it allowed:
        the solve runs on a copy. Releases the GIL as :func:`factorize` does.

        Raises:
            TypeError: If ``b`` has a dtype other than integer or floating-point,
                or ``out`` is not a ``float64`` array.
            ValueError: If ``b`` or ``out`` has a wrong shape, ``out``'s
                entries overlap or interleave, ``out`` is unaligned or a
                stride splits an entry, or ``out`` is not writeable.
            BufferError: If ``out`` is already borrowed.
        """

def factorize(
    matrix: Any,
    *,
    config: Config | None = None,
    scaling: ScalingPolicy | None = None,
    threads: int | None = None,
) -> ApproxInverse:
    """Factorize an SDDM matrix from a scipy.sparse CSR matrix.

    Copies the arrays, then releases the GIL to check and factor the copy above 32768 stored entries.
    Both stages run at ``threads``, which the operator then holds; every budget builds to the same bits.

    Args:
        matrix: A ``scipy.sparse.csr_matrix`` or ``csr_array``.
        config: Optional factorization configuration.
        scaling: Optional scaling policy for blocks short of diagonal dominance.
        threads: The thread budget, ``None`` every core.

    Raises:
        ValueError: If the matrix is not square or symmetric, has invalid
            dtypes/ranges, exceeds index limits, has invalid CSR structure, or
            has a nonpositive diagonal on a nonzero row, an all-zero row
            solving to 0; or if ``threads`` is outside ``[1, 2**64 - 1]``.
        TypeError: If ``threads`` is not an integer, or is a bool.

    Warns:
        RuntimeWarning: If a block falls short of diagonal dominance after
            scaling, so its deficit was clamped, or its scaling did not settle,
            so a singular block may have been grounded.
    """

def factorize_raw(
    row_ptrs: npt.ArrayLike,
    col_indices: npt.ArrayLike,
    values: npt.ArrayLike,
    n: int,
    config: Config | None = None,
    scaling: ScalingPolicy | None = None,
) -> ApproxInverse:
    """Factorize an SDDM matrix from raw CSR arrays, on every core.

    Releases the GIL as :func:`factorize`.

    Args:
        row_ptrs: CSR row pointer array, any integer dtype.
        col_indices: CSR column index array, any integer dtype.
        values: CSR value array, integer or floating-point.
        n: Matrix dimension.
        config: Optional factorization configuration.
        scaling: Optional scaling policy, as :func:`factorize`.

    Raises:
        ValueError: If an array is not 1-D or has a dtype it cannot carry, the
            CSR structure is invalid, or the matrix is not symmetric or has a
            nonpositive diagonal on a nonzero row; an all-zero row solves to 0.

    Warns:
        RuntimeWarning: As :func:`factorize`.
    """
