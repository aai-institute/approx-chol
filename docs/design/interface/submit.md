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
        pub fn named(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T],
            names: CsrNames) -> Result<Self, InputError>;
    }
    pub struct CsrNames {
        pub row_ptrs: &'static str,
        pub col_indices: &'static str,
        pub values: &'static str,
    }
    impl CsrNames { pub const CSR_REF: CsrNames; }
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
    #[cfg(feature = "nalgebra")]
    impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CsrMatrix<T>> for CsrRef<'a, T, usize> {
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
        ColumnMajor,
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
triangle is passed as the CSR of the other, the arrays of a faer `SparseColMatRef` holding the lower triangle
to `upper`. Only row-major storage converts: a sprs CSR view, a faer `SparseRowMatRef`, a
nalgebra-sparse `CsrMatrix`; a sprs CSC view is `ColumnMajor`, and a CSC reaches `CsrRef` through
its library's own conversion (`to_csr()`, `to_row_major()`, `CsrMatrix::from`). A sprs view
converts when its pointers and indices share one type, sprs's default. A conversion's errors name
each array by its source's field name (sprs `indptr`, `indices`, `data`; faer `row_ptr`,
`col_idx`, `val`; nalgebra-sparse `row_offsets`, `col_indices`, `values`) through
`CsrRef::named`; `new` is `named` with `CsrNames::CSR_REF`. `NotSquare`, `RowGap` and
`ColumnMajor` arise only from conversions; `RowGap` is a faer matrix stored with gaps
(`row_nnz` set). A first row pointer other than 0, an adapter's offset view
included, is `RowPtrs { parameter, row: 0 }`. `OutOfRange` is a stated value that is not finite, or neither
0 nor above the range floor; `Diagonal` a diagonal that is negative, past `MAX/4`, or zero beside
an edge. `VertexCountOverflow` is a `BipartiteFactorWidth2` whose sides' rows
sum past `usize`. A magnitude derived from a Gram form, a diagonal of $V W V^\top$, is the
build's to check (`BuildError::OutOfRange`), not the constructor's.

Python submits as part of the build call (`interface/build.md`): a matrix is checked where it is
passed.

```python
def factorize(matrix: sparray | spmatrix | CsrLike, *, ...) -> ApproxInverse: ...

class InputError(ValueError):
    kind: str            # the Rust variant in snake_case, e.g. "not_symmetric"
    parameter: str | None
    position: int | None # the entry, row, edge, column or vertex the variant names
```

Python takes any scipy sparse matrix, whole storage, through `tocsr()`, so a CSC factors its own
matrix; any other object with `indptr`, `indices`, `data` and `shape` is read as a CSR as it
stands. Anything else raises `TypeError`. Both index arrays take the integer dtype they share,
`int32`, `uint32` and `uint64` as they are, any other, narrower or mixed, as `int64`, since a
CSR has one index type; values are converted to `f64`, and the arrays are copied once into Rust, so the build runs with
the GIL released. `parameter` is `indptr`, `indices` or `data` of the CSR read, and a position
is in that CSR's storage, after any conversion. A rectangular matrix raises `not_square`. The message carries every field of the Rust
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
- **sprs, faer and nalgebra-sparse through `TryFrom` into `CsrRef`, row-major only:** their CSR
  storage is a `CsrRef`'s, borrowed. A CSC is not read as its transpose: symmetry is checked to a
  tolerance, so a nearly symmetric matrix's transpose is another matrix the check accepts, and the
  build would factor that one (#472, as Python's #432); the caller's conversion copies once.
- **Names passed in, `CsrRef::named`:** each check names the array it finds wrong as it raises
  the error, so no error is renamed afterwards and each source's names have one owner.
- **Any order and repeated columns:** they store the same matrix; Submit judges the matrix, not
  its storage.
- **One `TriangleCsrRef`, `upper` and `lower`:** both triangles carry the same fact, so one type;
  the constructor names the triangle, as `weighted` names the values.
- **One `I` for row pointers and column indices:** one index type per input; past $2^{32}$ nonzeros
  the caller widens both.
- **`Side` in `input`:** it names a part of `BipartiteFactorWidth2`; Build names a side with it.
- **Python: any scipy sparse matrix, inside the build call:** Python users hold scipy matrices,
  networkx's Laplacians among them; within, the one consumer of the other forms, calls Rust. A
  Python input object would be one more step that carries nothing. Arrays held without scipy
  pass as an object with a CSR's four attributes.
- **Python: a CSC through `tocsr()`, not as its transpose:** for the reason the Rust conversions
  take rows only (#432).
- **`InputError` a `ValueError`:** existing `except ValueError` keeps working; `kind` and the
  position mirror the Rust variant.
