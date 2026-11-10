# Solve interface

The contract is `../contracts/solve.md`; `Threads` and `ApproxInverse` are those of
`interface/build.md`. `StackReq` and `MemStack` are dyn-stack's.

## Decision

```rust
impl<T: Real> ApproxInverse<T> {
    pub fn scratch(&self, columns: usize, threads: Threads) -> StackReq;
    pub fn apply_with<M: MatrixMut<T> + ?Sized>(&self, block: &mut M, threads: Threads,
        stack: &mut MemStack) -> Result<(), ApplyError>;
    pub fn apply<M: MatrixMut<T> + ?Sized>(&self, block: &mut M, threads: Threads)
        -> Result<(), ApplyError>;
}

#[diagnostic::on_unimplemented(message = "…")]
pub trait MatrixMut<T: Real> {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn read_column(&self, column: usize, into: &mut [T]);
    fn write_column(&mut self, column: usize, from: &[T]);
}

impl<T: Real> MatrixMut<T> for [T] {}
impl<T: Real> MatrixMut<T> for Vec<T> {}
impl<T: Real, const K: usize> MatrixMut<T> for [[T; K]] {}
impl<T: Real, const K: usize> MatrixMut<T> for Vec<[T; K]> {}
impl<T: Real> MatrixMut<T> for ColumnMajor<'_, T> {}
impl<T: Real, C: AsRef<[T]> + AsMut<[T]>> MatrixMut<T> for Columns<'_, C> {}

pub struct ColumnMajor<'a, T> { /* private */ }
impl<'a, T> ColumnMajor<'a, T> {
    pub fn new(data: &'a mut [T], rows: usize) -> Result<Self, LayoutError>;
}
pub struct Columns<'s, C> { /* private */ }
impl<'s, C> Columns<'s, C> {
    pub fn new<T>(columns: &'s mut [C]) -> Result<Self, LayoutError> where C: AsRef<[T]>;
}

#[cfg(feature = "faer")]
impl<T: Real> MatrixMut<T> for faer::MatMut<'_, T> {}      // also Mat, ColMut, Col
#[cfg(feature = "nalgebra")]
impl<T: Real, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> MatrixMut<T>
    for nalgebra::Matrix<T, R, C, S> {}
#[cfg(feature = "ndarray")]
impl<T: Real> MatrixMut<T> for ndarray::ArrayRef<T, Ix2> {}   // also Ix1
#[cfg(feature = "faer")]
impl<T: Real + faer::traits::ComplexField> faer::matrix_free::Precond<T> for ApproxInverse<T> {}
    // also LinOp, BiLinOp, BiPrecond

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Length { rows: usize, found: usize },
    Column { column: usize, expected: usize, found: usize },
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    Rows { expected: usize, found: usize },
    Scratch,
}
```

- `scratch(columns, threads)` is what applying to a block of `columns` columns at `threads`
  needs, whichever its layout, the most of the paths below, to solve as many components at once
  as the budget has threads, at most the current pool's: a region per thread, of the most one of
  those components takes. It grows with `columns`; one thread's serves every budget, and a stack
  of fewer regions solves fewer components at once, to the same bits. `apply` allocates its
  scratch per call; `apply_with` allocates nothing at `Count(1)`; a budget above one runs on
  rayon's pool, which builds itself on first use.
- `block` holds the right-hand sides on entry and the solutions on return; a block with columns has
  $n$ rows, else `Rows`, and one without is left as it is, whatever its rows. Then `Scratch` is
  returned when `stack` cannot hold what the block's path takes at one thread, at most
  `scratch(block.columns(), Threads::Count(1))`, alignment included. Either error leaves `block`
  untouched.
- A `MatrixMut` keeps its shape during an apply; `read_column` copies a column into `into`,
  `write_column` copies `from` into one column and touches nothing else, and neither allocates. Each
  panics unless `column < columns()` and the slice has `rows()` entries; the crate's impls keep this
  contract, and the apply relies on no more. A one-dimensional type is one column and `[[T; K]]`
  holds $K$ interleaved columns; both solve where they are. `ColumnMajor` reads `data`
  column-major, `rows` to a column, `Length` unless that makes whole columns, none without data;
  `Columns` is one `&mut [T]`, `Vec<T>` or other `AsMut<[T]>` per column, `Column` at the first not
  of the first's rows. These column-stored layouts solve in interleaved packs of 8, 4 or 1 columns
  filled from their own columns, a column alone where it is; any other `MatrixMut` is staged one
  column at a time through `read_column` and `write_column` into the same packs. Foreign types
  implement it with any stride values: faer's dynamic-shape types, every nalgebra matrix, and
  ndarray's `ArrayRef`, which owned arrays and views dereference to; a faer or ndarray block whose
  columns are contiguous, row stride 1, solves as `ColumnMajor` does, any other stride staged.
  ndarray's `ArrayRef` is unsized, so an `Array1`, `Array2` or view is passed as `&mut *a`, and
  no `&mut dyn MatrixMut` holds one. A typed faer view converts with
  `as_dyn_mut().as_dyn_stride_mut()`.
- The unimplemented-trait message names the feature each foreign type needs and the supported
  versions, faer 0.24, nalgebra 0.35 and ndarray 0.17, and suggests `[..]` for an array and
  `&mut *a` for an ndarray array.
- With `faer`, `ApproxInverse<T>` implements faer's `LinOp`, `BiLinOp`, `Precond` and `BiPrecond`,
  its own adjoint and transpose: faer's `par` is the budget, `par.into()`, and its scratch
  queries are `scratch(rhs_ncols, par.into())`. The in-place methods apply in place, without
  faer's default temporary; `apply` copies `rhs` into `out` first. faer's traits return nothing,
  so a shape faer forbids or a stack too small panics, as faer's own operators do. faer's impls
  for references let a caller pass `&b` where faer takes the operator by value.

Python:

```python
class ApproxInverse(LinearOperator):
    @property
    def threads(self) -> int | None: ...
    def with_threads(self, threads: int | None) -> ApproxInverse: ...
    def apply(self, b: ArrayLike, *,
              out: NDArray[np.float64] | None = None) -> NDArray[np.float64]: ...
```

An `ApproxInverse` holds a budget, the build's `threads` unless `with_threads` names another, `None`
every core; `with_threads` shares $B$, copying nothing; every apply runs at the held budget. It is a
Python class over the Rust one, and a `LinearOperator` of shape $(n, n)$ and dtype `float64`, its
own adjoint: `matvec` and `rmatvec` take a vector or a stack of them, $(\ldots, n)$, `matmat` and
`rmatmat` $(\ldots, n, k)$, as scipy 1.18 passes them. `apply`'s `b` is a real array `np.asarray`
reads, a vector of length $n$ or an $n \times k$ array, converted to `f64` if it is not; any other
dtype raises `TypeError`. The solution has `b`'s shape, written to `out` when given, a writable
`f64` array of that shape in any order whose entries do not overlap, `b` itself allowed, else to a
new array. A wrong shape or an overlapping `out` raises `ValueError`. Every apply releases the GIL
and allocates its own scratch. scipy's `cg` and `minres` take it as `M`; `lsmr` takes no
preconditioner, so it right-preconditions there composed with $A$: `y` from
`lsmr(aslinearoperator(A) @ B, b)`, then `x = B @ y`.

## Why

- **`apply`:** $B$ is the approximate inverse, and applying it solves $Ax = b$ approximately, as
  faer's `LinOp::apply`, PETSc's `PCApply` and scipy's `matvec` apply an operator.
- **Monotone scratch:** one stack sized for the largest serves all, as faer's LSMR assumes of its
  preconditioner.
- **`MatrixMut` a trait:** an apply only copies columns in and out; each type copies its own way,
  with its strides, without `unsafe`, and the adapters implement it without the approximate inverse
  depending on their crates.
- **Packs of 8, 4 or 1:** the fused solve measured 2.5 times faster per column at 8 columns than
  one at a time, 2.1 times at 4, so filling a pack repays its copy; a column left alone gains
  nothing from one.
- **`ColumnMajor` and `Columns`:** the two forms whose input can be malformed, checked once;
  each states its shape once, by its data. Rows matter only to a column there is to solve, so
  a block without columns needs none.
- **ndarray through `ArrayRef`:** a shared or copy-on-write array copies itself on its first
  write; through `ArrayRef` that happens at the caller's `&mut *a`, in sight.
- **faer's operator traits:** faer's CG and LSMR take $B$ as is; their `par` and scratch queries
  map one to one, `par` beside the stack as the budget beside the scratch.
- **Python's held budget:** a `LinearOperator`'s `matvec` takes no budget, so the operator
  holds one; `with_threads` serves a solver that wants another than the build's. `out` is
  numpy's convention for reusing the result.
