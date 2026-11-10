# Migrating

## From 0.7 to 0.8

Each 0.7.0 item removed or renamed and what takes its place; [CHANGELOG.md](CHANGELOG.md) states the behavior. No 0.7.0 factor loads, and factors change bits where a row's residue is rounding, so a stored factor or a recorded solution is built again from its input.

### Inputs

- `CsrRef::new(row_ptrs, col_indices, values, n)`, `n: u32`, is `input::CsrRef::new(n, row_ptrs, col_indices, values)`, `n: usize`, and checks the matrix there.
- `input::UpperCsrRef::new(row_ptrs, col_indices, values, n)` is `input::TriangleCsrRef::upper(n, row_ptrs, col_indices, values)`.
- `FactorWidth1::new(n, rows, None)` is `FactorWidth1::new(n, rows)`; `FactorWidth1::new(n, rows, Some(values))` is `FactorWidth1::valued(n, rows, values)`.
- `Laplacian::new`, `Laplacian::weighted`, `Sddm::new`, `FactorWidth2::new`, `BipartiteFactorWidth2::new` and `scaled` take the same arguments and return `InputError`, not `ViewError`.
- The `Input` enum and its variants are the opaque `input::Input`, built by `From` from any input; `factorize`, `factorize_with` and `Sgdd::reduce` take `Into<Input>`, so a `TryInto<Input>` bound becomes `Into<Input>`.
- `OwnedCsr`, `OwnedCsr::try_from_usize`, `as_csr_ref` and `CsrRef::to_owned_u32`: none; a `CsrRef<T, usize>` borrows `usize` arrays as they are.
- `CsrRef::row_ptrs`, `col_indices`, `values` and `n`: none; the caller keeps the slices it passed.
- `factorize(&sprs_matrix)`, `Input: TryFrom<&CsMatI>` and `Input: TryFrom<CsMatViewI>` are `CsrRef::try_from(matrix.view())`; `&faer::sparse::SparseRowMat` and `Input: TryFrom<SparseRowMatRef>` are `CsrRef::try_from(matrix.as_ref())`, each passed to `factorize` as a `CsrRef`.
- `CsrRef`, `UpperCsrRef` and `OwnedCsr` defaulted `T` to `f64`; no input does, so `CsrRef<'_>` is `CsrRef<'_, f64>`.
- `approx_chol::{CsrRef, Input, Side}` are `approx_chol::input::{CsrRef, Input, Side}`.
- A Laplacian weight, Gram loading or scale, or stated SDDM surplus that is nonzero below `MIN_POSITIVE/EPSILON` is rejected at construction, `InputError::OutOfRange`, as a CSR entry was at `factorize`. A CSR value is checked as stored, before repeated columns sum, so duplicates below the floor that 0.7 summed into range are rejected; they are summed by the caller first.

```rust
// 0.7
use approx_chol::input::FactorWidth1;
use approx_chol::{factorize, CsrRef};

let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
let factor = factorize(csr)?;
let side = FactorWidth1::new(n_levels, &levels, Some(&loadings))?;
let factor = factorize(&sprs_matrix)?;
```

```rust
// 0.8
use approx_chol::factorize;
use approx_chol::input::{CsrRef, FactorWidth1};

let csr = CsrRef::new(4, &row_ptrs, &col_indices, &values)?;
let factor = factorize(csr)?;
let side = FactorWidth1::valued(n_levels, &levels, &loadings)?;
let factor = factorize(CsrRef::try_from(sprs_matrix.view())?)?;
```

### Errors

- `Error` is two types: `input::InputError` from each input's constructor, `BuildError` from `factorize`, `factorize_with`, `factorize_complement`, `Sgdd::reduce` and `Complement::reduce_onto`. `Error::InvalidCsr` is the constructor's `InputError`, `Error::Reduce` a `BuildError`; `From<Infallible> for Error` goes with `Error`.
- `ReduceError::TooLarge` and `OutOfRange` are `BuildError::TooLarge` and `OutOfRange`; `ReduceError::Unscalable` is `BuildError::OutOfRange`, its `vertex` the row that left the range, not the block's first vertex; `ReduceError::NonpositiveDiagonal` is `InputError::Diagonal { row }` at construction.
- `CsrError` and `ViewError` are `InputError`, each defect at its argument's `parameter` and position: `RowPtrsLenMismatch`, `ColIndicesValuesLenMismatch`, `RowPtrsEndMismatchNnz` and `LengthMismatch` are `LengthMismatch { parameter, expected, found }`; `RowPtrsMustStartAtZero` is `RowPtrs { parameter, row: 0 }`, `RowPtrsNotNonDecreasing` `RowPtrs { parameter, row }`; `ColumnIndexOutOfBounds`, `IndexNotRepresentableAsUsize` and `ViewError::IndexOutOfRange` are `IndexOutOfRange { parameter, entry }`; `CsrError::OutOfRange { row }` is `OutOfRange { parameter: "values", entry }`; `InvalidWeight`, `InvalidScale` and `InvalidSurplus` are `OutOfRange` or `Negative { parameter, entry }`; `RepeatedVertex` is `SelfLoop { edge }` or `RepeatedRow { column }`; `BelowDiagonal` is `OutsideTriangle`, `ExpectedSquareMatrix` `NotSquare`; `NotSymmetric` and `RowGap` keep their names.
- `IndexKind`: none; `parameter` names the slice.
- `IndexExceedsIndexType`: none; it came from narrowing indices in the removed owned conversions, and a `CsrRef` borrows the caller's index type as it is.
- `CsrError::MatrixDimensionExceedsIndexType` and `ViewError::ZeroSurplus`: none; both inputs are accepted.
- `SolveError` is `ApplyError`: `LengthMismatch` is `Rows`, `ScratchTooShort` `Scratch`.

```rust
// 0.7
use approx_chol::{factorize, CsrError, CsrRef, Error, ReduceError};

let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
match factorize(csr) {
    Ok(factor) => {}
    Err(Error::InvalidCsr(CsrError::NotSymmetric { row })) => return Err(format!("row {row}").into()),
    Err(Error::Reduce(ReduceError::Unscalable { vertex })) => return Err(format!("vertex {vertex}").into()),
    Err(e) => return Err(e.into()),
}
```

```rust
// 0.8
use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, BuildError};

let csr = match CsrRef::new(4, &row_ptrs, &col_indices, &values) {
    Ok(csr) => csr,
    Err(InputError::NotSymmetric { row }) => return Err(format!("row {row}").into()),
    Err(e) => return Err(e.into()),
};
match factorize(csr) {
    Ok(factor) => {}
    Err(BuildError::OutOfRange { vertex }) => return Err(format!("vertex {vertex}").into()),
    Err(e) => return Err(e.into()),
}
```

### Applying

- `Factor` is `ApproxInverse`.
- `Factor::solve(&b)` and `solve_into(&b, &mut x)` are a copy of `b` into `x` and `apply(&mut x, threads)`; `solve_in_place(&mut x, &mut scratch)` is `apply_with(&mut x, threads, stack)`.
- `Factor::scratch_len` is `scratch(columns, threads)`, a `StackReq`; a `MemBuffer` of it gives the `MemStack`.
- `InnerParallelism::Allow` is `Threads::All`, `Forbid` `Threads::Count(NonZeroUsize::MIN)`.
- A block of right-hand sides is one `MatrixMut`: a `Vec` or slice of `[T; K]`, `ColumnMajor::new(data, rows)`, `Columns::new(columns)`, or a faer, nalgebra or ndarray matrix under its feature, an ndarray array or view passed as `&mut *a`.

```rust
// 0.7
let x = factor.solve(&b)?;
let mut x = b.clone();
let mut scratch = vec![0.0; factor.scratch_len()];
factor.solve_in_place(&mut x, &mut scratch)?;
```

```rust
// 0.8
use approx_chol::{ColumnMajor, MemBuffer, MemStack, Threads};

let mut x = b.clone();
factor.apply(&mut x, Threads::All)?;
let mut y = b.clone();
let mut buffer = MemBuffer::new(factor.scratch(1, Threads::All));
factor.apply_with(&mut y, Threads::All, MemStack::new(&mut buffer))?;
let mut block = vec![0.0; factor.n() * 3];
factor.apply(&mut ColumnMajor::new(&mut block, factor.n())?, Threads::All)?;
```

### Building and blocks

- `Sgdd::reduce(input, scaling)` is `Sgdd::reduce(input, scaling, Threads::All)`; `Sgdd::factor(&config)` is `factor(config, Threads::All)`.
- `factorize_complement` stays; its two stages are `Complement::reduce_onto(input, kept, scaling, threads)` and `Complement::factor(config, threads)`.
- `Factor::blocks`, `into_blocks`, `BlockFactor` and `Vertices` are `ApproxInverse::components()`, one `Component` per block: `BlockFactor::vertices` is `Component::vertices`, `floats` `Component::floats`, and `Component::grounding_unproven` is new. `components()` lists blocks by their component's lowest vertex in the input's numbering, not in 0.7's block order, so a block is matched by its `vertices`, not by its position.
- `BlockFactor::solve_local`, `scratch_len` and `splittable_work`: none; a block solves only within its `ApproxInverse`, so what a caller solves apart, a subdomain of its own, is factored as an input of its own.
- `BlockFactor::scaling_report` and `ScalingReport { iterations, settled, violation }` are `Component::clamp()`, an `Option<&Clamp>` present only where the reduction clamped past rounding: `violation`, and `scaling`, a `ScalingOutcome`, in place of `settled`; `iterations` has none. Before factoring, `Sgdd::clamps()` and `Complement::clamps()` yield them as `Clamped`.

```rust
// 0.7
use approx_chol::{factorize_complement, Sgdd, Side};

let factor = Sgdd::reduce(gram, scaling)?.factor(&config);
for block in factor.blocks() {
    if let Some(report) = block.scaling_report() {
        let (vertices, violation) = (block.vertices(), report.violation);
    }
}
let schur = factorize_complement(gram, Side::Left, config, scaling)?;
```

```rust
// 0.8
use approx_chol::input::Side;
use approx_chol::{Complement, Sgdd, Threads};

let sgdd = Sgdd::reduce(gram, scaling, Threads::All)?;
for clamped in sgdd.clamps() {
    let (vertices, violation) = (clamped.vertices(), clamped.clamp().violation);
}
let factor = sgdd.factor(config, Threads::All);
for component in factor.components() {
    let (vertices, clamp) = (component.vertices(), component.clamp());
}
let schur = Complement::reduce_onto(gram, Side::Left, scaling, Threads::All)?
    .factor(config, Threads::All);
```

### Config

- `Config::split_merge` is `Config::sampling`: `None`, `Some(0)` and `Some(1)` are `Sampling::CliqueTree { copies: 1 }`, `Some(k)` `copies: k`; `SchurMode::Approximate { split_merge }` is `{ sampling }`.
- A `Config` literal names `frustrated`; `FrustratedRoute::Cover(CoverSurplus::Auto)` is the route every 0.7.0 build took.
- A 0.7.0 JSON `Config` is rejected; it is written again with `sampling` and `frustrated`.

```rust
// 0.7
use approx_chol::{Config, SchurMode};

let config = Config {
    seed: 0,
    split_merge: Some(2),
    exact_max_dim: 24,
    complement: SchurMode::Approximate { split_merge: Some(2) },
};
```

```rust
// 0.8
use std::num::NonZeroU32;

use approx_chol::{Config, CoverSurplus, FrustratedRoute, Sampling, SchurMode};

let ac2 = Sampling::CliqueTree { copies: NonZeroU32::new(2).unwrap() };
let config = Config {
    seed: 0,
    sampling: ac2,
    frustrated: FrustratedRoute::Cover(CoverSurplus::Auto),
    exact_max_dim: 24,
    complement: SchurMode::Approximate { sampling: ac2 },
};
```

### Persistence

- `FACTOR_FORMAT_VERSION` is `Header::of::<T>().version`; a header read apart is checked by `Header::expect::<T>()`, whose `HeaderMismatch` names what differs.
- A persisted `ApproxInverse` serializes as `Factor` did, led by its `Header`: postcard's bytes lead with the version, then the scalar type.
- A 0.7.0 payload does not load: postcard rejects it at its version, and a JSON one fails as a type error, ``invalid type: integer `1094909961`, expected struct Header``. A persisted `BlockFactor` or list of them has no successor; the `ApproxInverse` is persisted whole.

```rust
// 0.7
use approx_chol::{Factor, FACTOR_FORMAT_VERSION};

let bytes = postcard::to_stdvec(&factor)?;
let (version, _) = postcard::take_from_bytes::<u32>(&bytes)?;
assert_eq!(version, FACTOR_FORMAT_VERSION);
let factor: Factor = postcard::from_bytes(&bytes)?;
```

```rust
// 0.8
use approx_chol::{ApproxInverse, Header};

let bytes = postcard::to_stdvec(&factor)?;
let (header, _) = postcard::take_from_bytes::<Header>(&bytes)?;
header.expect::<f64>()?;
let factor: ApproxInverse = postcard::from_bytes(&bytes)?;
```

### Python

- `approx_chol.Factor` is `approx_chol.ApproxInverse`.
- `Config.split` is the clique-tree copies, `1` by default where it was `None`; a `split` of `0`, which 0.7 took as one copy, raises `ValueError`, and one outside `u32` raises `ValueError`, not `OverflowError`.
- An input error's message is `InputError`'s `Display`, and a value outside the range at build reads "a value at vertex …", not "an entry at vertex …".
- A `Config` compares and hashes by value.
- `factorize(matrix, config, scaling)` passed positionally is `factorize(matrix, config=config, scaling=scaling)`; `threads=` sets the build's budget, `None` every core as 0.7 built.
- `ApproxInverse` is a `scipy.sparse.linalg.LinearOperator`, not a duck type: `matvec`, `rmatvec`, `matmat`, `rmatmat` and `@` are scipy's, and stacks `(..., n)` and `(..., n, k)` apply.
- `ApproxInverse.solve(b)` is `apply(b)`; `solve_into(b, out, workspace)` is `apply(b, out=out)`, whose `out` has `b`'s shape and `float64` dtype, aligned, any order, no longer only contiguous with at least `n` entries; `workspace()` and `Workspace`: none, each apply allocates its own scratch. An `n × k` `b` solves as a block of columns, each bit for bit as alone, a NaN's sign and payload aside.
- A complex or other non-real `b` raises `TypeError`, not `ValueError`; a bool `split` or `threads` raises `TypeError`.
- `ApproxInverse.threads` reads the budget every apply runs at, and `with_threads(threads)` gives one at another, sharing the factor; a loaded or copied one holds `None`.
- An input or build error is an `approx_chol.InputError` or `approx_chol.BuildError`, both `ValueError`s: match `e.kind` and its fields (`parameter` and `position`, or `vertex`) where 0.7 matched the message, `except approx_chol.BuildError as e: e.kind, e.vertex` ([#430]).
- `factorize` reads a scipy sparse matrix through `tocsr()`: a CSC factors its own matrix, where 0.7 factored its transpose, and every other scipy format is accepted. An object that is neither scipy sparse nor has `indptr`, `indices`, `data` and `shape` raises `TypeError`, not `AttributeError` ([#432]).

```python
# 0.7
factor = approx_chol.factorize(laplacian, approx_chol.Config(seed=1))
x = factor.solve(b)
workspace = factor.workspace()
factor.solve_into(b, out, workspace)
```

```python
# 0.8
factor = approx_chol.factorize(laplacian, config=approx_chol.Config(seed=1), threads=4)
x = factor.apply(b)
factor.apply(b, out=out)
X = factor.with_threads(1).apply(B)  # an n × k block, serially, to the same bits
```

[#430]: https://github.com/aai-institute/approx-chol/issues/430
[#432]: https://github.com/aai-institute/approx-chol/issues/432
