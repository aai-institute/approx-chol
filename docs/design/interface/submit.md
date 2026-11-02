# Submit interface

The contract is `../contracts/submit.md`.

## Decision

```rust
pub mod input {
    pub struct CsrRef<'a, T, I = u32> { /* private */ }
    pub struct UpperCsrRef<'a, T, I = u32> { /* private */ }
    pub struct Laplacian<'a, T, I = u32> { /* private */ }
    pub struct Sddm<'a, T, I = u32> { /* private */ }
    pub struct FactorWidth2<'a, T, I = u32> { /* private */ }
    pub struct FactorWidth1<'a, T, I = u32> { /* private */ }
    pub struct BipartiteFactorWidth2<'a, T, I = u32> { /* private */ }

    impl<'a, T: Real, I: PrimInt> CsrRef<'a, T, I> {
        pub fn new(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T])
            -> Result<Self, InputError>;
    }
    impl<'a, T: Real, I: PrimInt> UpperCsrRef<'a, T, I> {
        pub fn new(n: usize, row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T])
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
        pub fn new(n: usize, a: (&'a [I], &'a [T]), b: (&'a [I], &'a [T]))
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
    impl<'a, T: Real, I: SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
        type Error = InputError;
    }
    #[cfg(feature = "faer")]
    impl<'a, T: Real, I: faer::Index> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
        for CsrRef<'a, T, I>
    {
        type Error = InputError;
    }

    #[non_exhaustive]
    pub enum InputError {
        LengthMismatch { parameter: &'static str, expected: usize, found: usize },
        RowPtrs { row: usize },
        IndexOutOfRange { parameter: &'static str, entry: usize },
        NotFinite { parameter: &'static str, entry: usize },
        Negative { parameter: &'static str, entry: usize },
        Magnitude { vertex: usize },
        SelfLoop { edge: usize },
        RepeatedRow { column: usize },
        ZeroSurplus,
        NotSymmetric { row: usize },
        BelowDiagonal { row: usize },
        InadmissibleRow { row: usize },
        NotSquare { rows: usize, cols: usize },
        RowGap { row: usize },
    }

    pub enum Side { Left, Right }
}
```

`Input` also converts `From` `UpperCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2` and
`BipartiteFactorWidth2`. `T` is `f32` or `f64` (`Real`, sealed); `I` is any primitive integer,
`u32` unless named. Every type is `Copy` and `Debug`; `InputError` implements `Error` and
`Display`. `CsrRef::new` and `UpperCsrRef::new` accept a row's entries in any order, and
repeated columns, which sum. A sprs CSC view is read as the CSR of its transpose, its positions in
that storage.
`NotSquare` and `RowGap` arise only from the sprs and faer conversions.

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
- **`#[non_exhaustive]`:** variants and slices can be added without a breaking release.
- **`Input` opaque, built only by `From`:** it hands nothing back.
- **sprs and faer through `TryFrom` into `CsrRef`:** their matrices are compressed storage; a
  symmetric matrix's CSC storage is the CSR of its transpose.
- **Any order and repeated columns:** they store the same matrix; Submit judges the matrix, not
  its storage.
- **One `I` for row pointers and column indices:** one index type per input; past $2^{32}$ nonzeros
  the caller widens both.
- **`Side` in `input`:** it names a part of `BipartiteFactorWidth2`; Build names a side with it.
