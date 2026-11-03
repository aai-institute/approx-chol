# Solve interface

The contract is `../contracts/solve.md`; `Threads` and `ApproxCholesky` are those of
`interface/build.md`. `StackReq` and `MemStack` are dyn-stack's.

## Decision

```rust
impl<T: Real> ApproxCholesky<T> {
    pub fn solver(&self, threads: Threads) -> Solver<'_, T>;
}

#[derive(Clone, Copy, Debug)]
pub struct Solver<'b, T> { /* private */ }
impl<'b, T: Real> Solver<'b, T> {
    pub fn scratch(&self) -> StackReq;
    pub fn solve_with<M: MatrixMut<T> + ?Sized>(&self, block: &mut M, stack: &mut MemStack)
        -> Result<(), SolveError>;
    pub fn solve<M: MatrixMut<T> + ?Sized>(&self, block: &mut M) -> Result<(), SolveError>;
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
impl<T: Real> MatrixMut<T> for Columns<'_, '_, T> {}

pub struct ColumnMajor<'a, T> { /* private */ }
impl<'a, T> ColumnMajor<'a, T> {
    pub fn new(data: &'a mut [T], rows: usize, columns: usize) -> Result<Self, LayoutError>;
}
pub struct Columns<'s, 'a, T> { /* private */ }
impl<'s, 'a, T> Columns<'s, 'a, T> {
    pub fn new(rows: usize, columns: &'s mut [&'a mut [T]]) -> Result<Self, LayoutError>;
}

#[cfg(feature = "faer")]
impl<T: Real> MatrixMut<T> for faer::MatMut<'_, T> {}      // also Mat, ColMut, Col
#[cfg(feature = "nalgebra")]
impl<T: Real, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> MatrixMut<T>
    for nalgebra::Matrix<T, R, C, S> {}
#[cfg(feature = "ndarray")]
impl<T: Real> MatrixMut<T> for ndarray::ArrayRef<T, Ix2> {}   // also Ix1
#[cfg(feature = "faer")]
impl<T: Real + faer::traits::ComplexField> faer::matrix_free::Precond<T> for ApproxCholesky<T> {}
    // also LinOp, BiLinOp, BiPrecond

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Length { expected: usize, found: usize },
    Overflow,
    Column { column: usize, expected: usize, found: usize },
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    Rows { expected: usize, found: usize },
    Scratch,
}
```

- `solver` fixes the budget, `Threads::All` resolved to the pool's thread count at the call; a
  `Solver` keeps that count as a cap in any pool. `scratch` serves every block at that budget.
  `solve` allocates `scratch()` and calls `solve_with`, which allocates nothing.
- `block` holds the right-hand sides on entry and the solutions on return; it has $n$ rows, else
  `Rows`, also without columns. `Scratch` is returned when `stack` cannot hold `scratch()`,
  alignment included. Either error leaves `block` untouched.
- A `MatrixMut` keeps its shape during a solve; `read_column` copies a column into `into`,
  `write_column` copies `from` into one column and touches nothing else, and neither allocates.
  Each panics unless `column < columns()` and the slice has `rows()` entries; the crate's impls
  keep this contract, and the solve relies on no more. A one-dimensional type is one column;
  `[[T; K]]` holds $K$ interleaved columns; `ColumnMajor` reads `data` column-major, `Length`
  unless it holds rows $\cdot$ columns entries, `Overflow` if that product does not fit; `Columns`
  is one slice per column, `Column` at the first not of `rows` entries. Foreign types implement
  it with any stride values: faer's dynamic-shape types, every nalgebra matrix, and ndarray's
  `ArrayRef`, which owned arrays and views dereference to. A typed faer view converts with
  `as_dyn_mut().as_dyn_stride_mut()`.
- The unimplemented-trait message names the feature each foreign type needs and the supported
  versions, faer 0.24, nalgebra 0.35 and ndarray 0.17, and suggests `[..]` for an array.
- With `faer`, `ApproxCholesky<T>` implements faer's `LinOp`, `BiLinOp`, `Precond` and
  `BiPrecond`, its own adjoint and transpose, through `solver(par.into())`. A shape faer's
  contract forbids panics, as faer's own operators do.

Python:

```python
class ApproxCholesky(LinearOperator):
    def solve(self, b: ArrayLike, *, threads: int | None = None,
              out: NDArray[np.float64] | None = None) -> NDArray[np.float64]: ...
```

`b` is a real array `np.asarray` reads, a vector of length $n$ or an $n \times k$ array, converted
to `f64` if it is not; any other dtype raises `TypeError`. The solution has `b`'s shape, written to
`out` when given, a writable `f64` array of that shape in any order whose entries do not
overlap, `b` itself allowed, else to a new array. A wrong shape or an overlapping `out` raises
`ValueError`. The solve releases the GIL and allocates its own scratch.

## Why

- **`Solver`:** $B$ with a budget is a value, the scratch its per-use state; a held `Solver` sizes
  and solves at one budget, and `Scratch` catches a stack from elsewhere.
- **`solve` allocating, `solve_with` not:** the convenience takes the short name, the explicit
  form the suffix, as in rustfft's `process`/`process_with_scratch` and regex's
  `search`/`search_with`.
- **`MatrixMut` a trait:** a solve only copies columns in and out; each type copies its own way,
  with its strides, without `unsafe`, and the adapters implement it without the approximate
  inverse depending on their crates. A layout costs that copy, 4–8% of a block solve.
- **`ColumnMajor` and `Columns`:** the two forms whose input can be malformed, checked once;
  both name their rows, so a block without columns has a shape.
- **ndarray through `ArrayRef`:** a shared or copy-on-write array copies itself on its first
  write; through `ArrayRef` that happens at the caller's `&mut *a`, in sight.
- **faer's operator traits:** faer's CG and LSMR take the approximate inverse as is.
- **Python `out`, scratch per call:** numpy's convention for reusing the result.
