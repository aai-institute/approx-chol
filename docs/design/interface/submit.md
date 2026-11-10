# Submit interface

The contract is `../contracts/submit.md`.

## Decision

```rust
pub mod input {
    pub struct CsrRef<'a, T, I = u32> { /* private */ }
    pub struct TriangleCsrRef<'a, T, I = u32> { /* private */ }
    pub struct Laplacian<'a, T, I = u32> { /* private */ }
    pub struct Sddm<'a, T, I = u32> { /* private */ }
    pub struct FactorWidth2<'a, T, I = u32> { /* private */ }
    pub struct FactorWidth1<'a, T, I = u32> { /* private */ }
    pub struct BipartiteFactorWidth2<'a, T, I = u32> { /* private */ }

    impl<'a, T: Real, I: PrimInt> CsrRef<'a, T, I> {
        pub fn new(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T])
            -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> TriangleCsrRef<'a, T, I> {
        pub fn upper(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T])
            -> Result<Self, InputError>;
        pub fn lower(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T])
            -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
        pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, InputError>;
        pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T])
            -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> Sddm<'a, T, I> {
        pub fn new(laplacian: Laplacian<'a, T, I>, surplus: &'a [T]) -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
        pub fn new(n: usize, (a, x): (&'a [I], &'a [T]), (b, y): (&'a [I], &'a [T]))
            -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> FactorWidth1<'a, T, I> {
        pub fn new(n: usize, rows: &'a [I]) -> Result<Self, InputError>;
        pub fn valued(n: usize, rows: &'a [I], values: &'a [T]) -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> BipartiteFactorWidth2<'a, T, I> {
        pub fn new(left: FactorWidth1<'a, T, I>, right: FactorWidth1<'a, T, I>)
            -> Result<Self, InputError>;
        pub fn scaled(left: FactorWidth1<'a, T, I>, right: FactorWidth1<'a, T, I>, s: &'a [T])
            -> Result<Self, InputError>;
    }

    pub struct Input<'a, T, I = u32> { /* private */ }
    impl<'a, T, I> From<CsrRef<'a, T, I>> for Input<'a, T, I> { /* … */ }

    #[cfg(feature = "sprs")]
    impl<'a, T: Real, I: PrimInt + SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>>
        for CsrRef<'a, T, I>
    {
        type Error = InputError;
    }
    #[cfg(feature = "faer")]
    impl<'a, T: Real, I: PrimInt + faer::Index> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
        for CsrRef<'a, T, I>
    {
        type Error = InputError;
    }
    #[cfg(feature = "faer")]
    impl<'a, T: Real, I: PrimInt + faer::Index> TryFrom<faer::sparse::SparseColMatRef<'a, I, T>>
        for CsrRef<'a, T, I>
    {
        type Error = InputError;
    }
    #[cfg(feature = "nalgebra")]
    impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CsrMatrix<T>> for CsrRef<'a, T, usize> {
        type Error = InputError;
    }
    #[cfg(feature = "nalgebra")]
    impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CscMatrix<T>> for CsrRef<'a, T, usize> {
        type Error = InputError;
    }

    #[non_exhaustive]
    pub enum InputError {
        VertexCountOverflow { n_left: usize, n_right: usize },
        LengthMismatch { parameter: &'static str, expected: usize, found: usize },
        RowPtrs { parameter: &'static str, row: usize },
        IndexOutOfRange { parameter: &'static str, entry: usize },
        OutOfRange { parameter: &'static str, entry: usize },
        Negative { parameter: &'static str, entry: usize },
        SelfLoop { edge: usize },
        RepeatedRow { column: usize },
        NotSymmetric { row: usize },
        OutsideTriangle { row: usize },
        Diagonal { row: usize },
        NotSquare { rows: usize, cols: usize },
        RowGap { row: usize },
    }

    pub enum Side { Left, Right }
}
```

`Input` also converts `From` `TriangleCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2` and
`BipartiteFactorWidth2`. `T` is `f32` or `f64` (`Real`, sealed); `I` is any primitive integer,
`u32` unless named, so `Laplacian::<f64, i64>::new(n, &u, &v)` names both where no value fixes
`T`. Every type is `Copy` and `Debug`; every error type implements `Error` and `Display`.
`CsrRef::new`, `TriangleCsrRef::upper` and `TriangleCsrRef::lower` accept a row's entries in any
order, and repeated columns, which sum; a triangle's diagonal is stored once, in it. A CSC
triangle is passed as the CSR of the other, a faer `SparseColMatRef` holding the lower triangle
to `upper`. A sprs CSC view, a faer `SparseColMatRef` and a nalgebra-sparse `CscMatrix` are read
as the CSR of their transpose, positions in that storage. A sprs view converts when its pointers
and indices share one type, sprs's default. `NotSquare` and `RowGap` arise only from conversions,
whose `parameter` is the source's own field name; `RowGap` is a faer matrix stored with gaps
(`row_nnz` or `col_nnz` set). A first row pointer other than 0, an adapter's offset view
included, is `RowPtrs { parameter, row: 0 }`. `OutOfRange` is a stated value that is not finite, or neither
0 nor above the range floor; `Diagonal` a diagonal that is negative, past `MAX/4`, or zero beside
an edge. `VertexCountOverflow` is a `BipartiteFactorWidth2` whose sides' rows
sum past `usize`. A magnitude derived from a Gram form, a diagonal of $V W V^\top$, is the
build's to check (`BuildError::OutOfRange`), not the constructor's.

Python submits as part of the build call (`interface/build.md`): a matrix is checked where it is
passed.

```python
def factorize(matrix: sparray | spmatrix, *, ...) -> ApproxInverse: ...

class InputError(ValueError):
    kind: str            # the Rust variant in snake_case, e.g. "not_symmetric"
    parameter: str | None
    position: int | None # the entry, row, edge, column or vertex the variant names
```

Python takes any scipy sparse matrix, whole storage. CSR, and CSC as the CSR of its transpose,
with `f64` values and `int32` or `int64` indices are read without a copy; any other format goes
through `tocsr()` and any other dtype is converted to `f64` or `int64`, each at most once.
Anything else raises `TypeError`. `parameter` is `indptr`, `indices` or `data` of the matrix
read. A rectangular matrix raises `not_square`. The message carries every field of the Rust
variant; `kind`, `parameter` and `position` are for matching.

## Why

- **`n` first in every constructor from slices:** each form reads size, then data, as sprs, faer
  and nalgebra-sparse construct theirs.
- **Values by a second constructor (`weighted`, `valued`, `scaled`):** no `Option` argument, and
  each name says what the values mean. A constructor without values cannot infer `T`, so the
  caller names it there.
- **One flat `InputError`, each variant with its own position:** no defect can carry a position
  that makes no sense for it. A slice is named by the constructor's own parameter, the caller's
  exact terms, for reading, not matching; whether an index or a value is wrong is the variant. A
  bipartite form numbers its vertices left side first, so a vertex is a position on both sides.
- **`#[non_exhaustive]`:** variants can be added without a breaking release.
- **sprs, faer and nalgebra-sparse through `TryFrom` into `CsrRef`:** their matrices are
  compressed storage; a symmetric matrix's CSC storage is the CSR of its transpose, so the column
  forms, faer's and nalgebra-sparse's main ones, convert too.
- **Any order and repeated columns:** they store the same matrix; Submit judges the matrix, not
  its storage.
- **One `TriangleCsrRef`, `upper` and `lower`:** both triangles carry the same fact, so one type;
  the constructor names the triangle, as `weighted` names the values.
- **One `I` for row pointers and column indices:** one index type per input; past $2^{32}$ nonzeros
  the caller widens both.
- **`Side` in `input`:** it names a part of `BipartiteFactorWidth2`; Build names a side with it.
- **Python: any scipy sparse matrix, inside the build call:** Python users hold scipy matrices,
  networkx's Laplacians among them; within, the one consumer of the other forms, calls Rust. A
  Python input object would be one more step that carries nothing.
- **`InputError` a `ValueError`:** existing `except ValueError` keeps working; `kind` and the
  position mirror the Rust variant.
