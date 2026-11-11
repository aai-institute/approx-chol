# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `CoverSurplus::Auto` decides a sampled complement's cover on the complement's exact off-diagonal, before sampling.

### Fixed

- A sampled complement no longer carries the extra fill of 0.8.0: its clique trees draw each star's exact ground again and drop what lands on it, except on a twisted cover. ([#437])

## [0.8.0] - 2026-11-11

Upgrading from 0.7: [MIGRATING.md](MIGRATING.md) maps each removed or renamed item to what takes its place.

### Added

- `ApproxInverse::apply` and `apply_with` solve a block of right-hand sides at a `Threads` budget, each column bit for bit as it solves alone; `scratch(columns, threads)` sizes `apply_with`'s stack. A block is a `MatrixMut`: a slice, `[T; K]` rows, `ColumnMajor` or `Columns`.
- `Config::frustrated`, a `FrustratedRoute`: factor a frustrated block on its Gremban cover, `Cover(CoverSurplus)` with `Cover(Auto)` the default, or eliminate it signed, `Signed`. In Python, `Config(frustrated=...)`. ([#433], [#435])
- `CsrRef::named` and `CsrNames` name the arrays in an `InputError` by the caller's names. ([#472])
- `faer`: faer matrices are `MatrixMut` blocks, `ApproxInverse` implements `LinOp`, `BiLinOp`, `Precond` and `BiPrecond`, and `Threads: From<faer::Par>`. ([#426])
- `nalgebra`: a nalgebra-sparse `&CsrMatrix` converts into a `CsrRef`, and nalgebra matrices are `MatrixMut` blocks; the feature needs Rust 1.89. ([#427])
- `ndarray`: `ArrayRef2` and `ArrayRef1` are `MatrixMut` blocks; `sprs` enables it. ([#428])
- `Complement::reduce_onto` and `Complement::factor` split a Schur build into its two stages, and `Sgdd::clamps` and `Complement::clamps` report each clamped component before factoring. ([#424])
- `Component::grounding_unproven` flags a block grounded or floated on a scaling that did not settle.
- Python `ApproxInverse.components`, `threads` and `with_threads`. ([#429])
- Python `ApproxInverse` pickles and deep-copies. ([#431])

### Changed (breaking)

- `Factor` is renamed `ApproxInverse`, in Rust and Python.
- Inputs are in the `input` module and built size first: `CsrRef::new(n, row_ptrs, col_indices, values)`, `TriangleCsrRef::upper` for `UpperCsrRef`, `FactorWidth1::new` and `valued`. `factorize` takes any `Into<Input>`. `OwnedCsr` and the direct sprs and faer matrix conversions are removed; convert their `view()` or `as_ref()`. ([#420])
- `InputError` replaces `CsrError` and `ViewError`, and input constructors check every defect, diagonals and value range included, before the build. ([#419], [#421])
- `BuildError` replaces `Error` and `ReduceError`. ([#422])
- Only row-major storage converts into a `CsrRef`; a column-stored sprs view is `InputError::StoredByColumns`. ([#472])
- `solve`, `solve_into`, `solve_in_place` and `scratch_len` are removed in favor of `apply` and `apply_with`; `SolveError` is `ApplyError`.
- `Threads` replaces `InnerParallelism`; `Sgdd::reduce` and `Sgdd::factor` take a `Threads` budget, and `Sgdd::factor` takes `Config` by value. ([#424])
- `BlockFactor`, `blocks` and `into_blocks` are removed; `ApproxInverse::components` lists a `Component` per block, and `Component::clamp` replaces `ScalingReport`.
- `Config::sampling`, a `Sampling`, replaces `Config::split_merge`; in Python, `Config(split=0)` raises `ValueError`.
- `Header` replaces `FACTOR_FORMAT_VERSION`; a persisted factor is format `0x4143_000F`, and earlier payloads are rejected. ([#425], [#433])
- A row's surplus within rounding is no longer a ground, so factors change bits on such rows, and an `f32` input with weights spanning `1e14` no longer grounds on rounding. ([#311], [#343])
- `Sddm::new` accepts an all-zero surplus. ([#423])
- Python `ApproxInverse` is a scipy `LinearOperator`, and `apply(b, *, out=None)` replaces `solve`, `solve_into`, `workspace` and `Workspace`. ([#429])
- Python `factorize` takes its options by keyword, `threads` among them, and reads any scipy sparse format through `tocsr()`; `factorize_raw` is removed. ([#429], [#432])
- Python raises `InputError` and `BuildError`, both `ValueError` subclasses, with `kind`, `parameter` and `position`. ([#430])

### Fixed

- `BipartiteFactorWidth2` rejects side lengths whose sum overflows `usize`. ([#421])
- An approximate step's heaviest share no longer loses accuracy at high degree, up to 8e-6 relative at degree 10⁶. ([#306])
- Decoding rejects a negative share or an invalid clamp, which decoded before. ([#418])
- The exact arm counts a frustrated block's vertices once, so blocks up to `exact_max_dim` factor exactly. ([#327])
- A sampled complement under `CoverSurplus::SurplusEdges` no longer panics. ([#437])

## [0.7.0] - 2026-10-30

### Changed (breaking)

- `Config` gains `complement: SchurMode`. ([#182])
- `factorize_with` takes a `ScalingPolicy` after its `Config`. ([#267])
- `Backend` and `Config::backend` are removed: `Config::exact_max_dim` is the largest block dimension factored exactly, `0` none; a `Config` serialized by 0.6.0 is rejected from JSON and unsupported from postcard. ([#322])
- `ExactFailure`, `Fallback`, `DenseFailure`, `UnusablePivot` and `Factor::fallbacks` are removed, as the exact arm never fails a pivot. ([#182])
- `Error` is `InvalidCsr(CsrError)` or `Reduce(ReduceError)`; a value defect is `CsrError::{NotSymmetric, BelowDiagonal, OutOfRange}`, naming its row. ([#182])
- `CsrRef::new`, `CsrRef::to_owned_u32`, `OwnedCsr::try_from_usize` and `CsrRef`'s `sprs`/`faer` conversions return `CsrError`, not `Error`.
- A column outside `0..n` is reported by `factorize`, not `CsrRef::new`; `CsrError::InputConversionPanicked` is removed. ([#182])
- `factorize` takes any `TryInto<Input>`; a type converting only into `CsrRef` converts first. ([#182])
- `factorize` and `Factor`'s methods take `T: Real`, implemented only for `f32` and `f64`, and `factorize` an index `I: Sync`. ([#182])
- `low_level` is removed. ([#182])
- `CsrError::ExpectedCsrMatrixGotCsc` is removed: a CSC `sprs` matrix converts into `CsrRef` as its transpose, which a symmetric one equals; a CSC `faer` matrix converts as `matrix.as_ref().transpose()`.
- `Factor::n` is the input dimension; `Factor::original_n` is removed. ([#140])
- `Factor::solve_in_place` takes `x` of length `Factor::n` and caller scratch of at least `Factor::scratch_len`. ([#146])
- `Factor::solve` and `Factor::solve_into` reject a right-hand side or output whose length is not `Factor::n`. ([#146])
- `SolveError` is `LengthMismatch` or `ScratchTooShort`. ([#146])
- An entry below `MIN_POSITIVE / EPSILON` in magnitude is `CsrError::OutOfRange`; 0.6.0 accepted it, and solved scales below about 1e-307 wrongly without an error. ([#163])
- `Factor::solve_in_place` returns `solve`'s `A⁺b` on each floating block; 0.6.0 pinned an elimination-dependent vertex of each floating block to zero instead. ([#135])
- `FACTOR_FORMAT_VERSION` is `0x41430009`, and no other version decodes. ([#148], [#179])
- Python `Backend` is removed: `Config(exact_max_dim=...)` replaces it; `ExactFailure`, `DenseFailure`, `Fallback` and `Factor.fallbacks` are removed. ([#182])
- Python `Factor.n` is the matrix dimension, and `Factor.solve`, `solve_into` and `matvec` reject a `b` whose length is not `n`. ([#140], [#146])

### Changed

- `Config::exact_max_dim` bounds a block's dimension, not its solved variables: a floating block of `exact_max_dim + 1` vertices factors approximately. ([#182])
- `factorize` admits symmetric generalized diagonally dominant input: positive off-diagonals fold by a ±1 scaling or factor on a double cover, and a block short of dominance is scaled under `factorize_with`'s `ScalingPolicy`. ([#183])
- Python `Factor.solve`, `solve_into`, `matvec` and `rmatvec` take any 1-D integer or float array-like, and `matvec`/`rmatvec` scipy's `(n, 1)` column. ([#224])
- Python `Factor.solve_into` accepts `out` overlapping `b`, so `solve_into(v, v)` solves in place. ([#300])

### Added

- `input`: `Input`, taking `UpperCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2` and `BipartiteFactorWidth2`, with `FactorWidth1` and `ViewError`. ([#183])
- `SchurMode`, `ScalingPolicy`, `ScalingReport` and `ReduceError`. ([#182])
- `Factor::blocks`, `Factor::into_blocks`, `BlockFactor` and `InnerParallelism`: a `BlockFactor` solves one block in place and persists on its own. ([#183])
- `Factor::scratch_len`. ([#146])
- A factor of more than `(u32::MAX - 1) / 2` vertices stores `u64` indices. ([#182])
- Python `factorize` warns when a block's deficit is clamped or its scaling does not settle. ([#181])
- Python `factorize_raw` takes arrays of any integer index dtype. ([#181])
- Python `Factor.workspace` returns a `Workspace`, which `Factor.solve_into` reuses across calls. ([#255])
- `Sgdd`: `factorize_with`'s two steps, `Sgdd::reduce` an `Input`, then `Sgdd::factor` it. ([#267])
- Python `ScalingPolicy`, which `factorize` and `factorize_raw` take as `scaling`. ([#301])
- `factorize_complement` factors a `BipartiteFactorWidth2`'s Schur complement onto a `Side`, formed under `Config::complement` as `factorize_with` forms it; the factor's vertex `k` is that side's `k`-th. ([#315])

### Fixed

- A `faer` matrix with unused slots inside a row is `CsrError::RowGap`; 0.6.0 read them as entries. ([#180])
- A grounded block whose weights span past `1/ε` ends its approximate elimination at the ground; it could solve inaccurately. ([#228])
- Python `factorize`, `factorize_raw`, `Factor.solve` (so `matvec`/`rmatvec`) and `Factor.solve_into` release the GIL for a matrix of more than 32768 stored entries, working only on their own copies meanwhile. ([#255])
- `Factor<f32>` keeps a high-degree pivot's `1/d` and fill weights; they drifted by `~mε` at degree `m`, up to a negative pivot. ([#275])
- AC2's degree queue takes memory linear in the input for any `split_merge`. ([#277])

## [0.6.0] - 2026-08-24

### Changed (breaking)

- A persisted `Factor` stores each neighbor's share of the original pivot, and
  `FACTOR_FORMAT_VERSION` is `0x41430004`. ([#131])
- A persisted column names the neighbor taking the remaining pivot rather than storing its
  share. ([#131])

### Changed

- Faster `Factor::solve` on dense factors, from composing elimination fractions at build
  time. ([#131])

## [0.5.0] - 2026-08-11

### Changed (breaking)

- A persisted `Factor` no longer carries `original_n`; it is derived from the blocks, and
  `FACTOR_FORMAT_VERSION` is `0x41430003`. ([#118])
- A deserialized `Factor` no longer rejects a tampered block anchor, which 0.4.0
  documented as rejected. ([#118])

### Changed

- A block routed to the exact dense backend no longer gets an elimination graph built for
  it. ([#84])
- `Error::NonFiniteValue` no longer takes precedence over `Error::Asymmetric`,
  `Error::PositiveOffDiagonal` and `Error::NotDiagonallyDominant`. ([#84])
- The Python bindings build against pyo3 and numpy 0.29. ([#89])

### Fixed

- The star sampler draws from `[0, 1)`; the previous mapping could return exactly `1.0`. ([#109])

## [0.4.0] - 2026-07-30

### Changed (breaking)

- Each connected component of a disconnected input is factored, grounded and projected
  on its own. ([#35], [#36])
- `solve`/`solve_into` cap the right-hand side at the original dimension;
  `solve_in_place` leaves one variable pinned per component.
- The serde representation of `Factor` is incompatible with 0.3.1, and a fixed seed can
  produce a different factor.
- `Config` gains a `backend` field, defaulting to exact dense Cholesky for a component
  of at most 24 solved variables. ([#83])
- `Error::Asymmetric`, `Error::NonFiniteValue`, `Error::NonFiniteRow` and
  `Error::NotDiagonallyDominant` reject non-symmetric, non-finite and non-dominant
  input, and duplicate entries are summed before the off-diagonal sign check.
- `Config::split_merge` of `Some(0)` or `Some(1)` selects standard AC.
- `low_level::CliqueTreeSampler` replaces `clique_tree_sample`/`clique_tree_sample_multi`:
  `seed` and `split_merge` move to `new`, `sample` takes the star index, and entries of
  equal weight order by neighbor index.
- `OwnedCsr::try_as_ref` is replaced by the infallible `as_csr_ref`, and `CsrRef`
  accessors return slices with the view's lifetime.
- A row surplus above `epsilon * scale * terms` is grounded, a deficit below it is
  `Error::NotDiagonallyDominant`. ([#78], [#85], [#91])
- A persisted `Factor` declares `format_version` and rejects any other version. ([#50])
- The sampler judges weights by sign rather than against an absolute floor, so a fixed
  seed's factor differs below `1e-14` (`f64`) / `1e-6` (`f32`). ([#92])

### Removed

- `Error::InvalidConfig` and `ConfigError`.

### Added

- `Config::backend`: `Backend::ExactBelow { max_dim, on_failure }` or
  `Backend::Approximate`, with `ExactFailure` choosing fallback or
  `Error::DenseFactorizationFailed`. ([#83])
- `Factor::fallbacks`, `Fallback`, `UnusablePivot` and `DenseFailure`. ([#83])
- Python `Backend`, `ExactFailure`, `DenseFailure`, `Fallback`, `Config(backend=...)`,
  `Factor.fallbacks`, and a `RuntimeWarning` per fallback. ([#83])
- `From<&OwnedCsr> for CsrRef`. ([#41])
- One `cp39-abi3` wheel per platform, plus `cp314t` free-threaded wheels. ([#61])
- `requires-python` drops to 3.9 and the `scipy` floor rises to 1.12. ([#61])
- `FACTOR_FORMAT_VERSION`. ([#50])
- An MSRV of Rust 1.85. ([#62])

### Fixed

- An invalid `Factor` — tampered anchor, mismatched `original_n`, non-finite or
  out-of-range values — is rejected at deserialize time. ([#37], [#80])
- `CliqueTreeSampler` no longer panics or emits non-finite fill on degenerate
  weights. ([#38])
- `u32` nonzero/edge overflow panics instead of truncating the factor. ([#39])
- A sub-unit-scale strictly-dominant SDDM is augmented, and elimination keeps the scale
  of an underflowing diagonal or per-copy share. ([#36])
- Approximate elimination keeps a small pivot's reciprocal wherever it is
  representable. ([#75])
- The approximate solve zeroes the uneliminated vertex between passes, so a small
  solution is no longer annihilated. ([#93])
- Each block draws from its own sampler stream, so a fixed seed factors it the same way
  under either backend. ([#82])
- A large diagonal no longer overflows the row-scale computation.

## [0.3.1] - 2026-07-10

### Fixed

- Reject positive off-diagonals (`Error::PositiveOffDiagonal`) at ingestion
  instead of silently dropping them, which had produced a wrong factor on
  non-SDDM input.

## [0.3.0] - 2026-06-15

### Changed

- Faster factorization via batched minimum-degree priority-queue updates. The
  randomized factor produced for a given seed may differ from 0.2.0;
  correctness is unaffected.

## [0.2.0] - 2026-05-26

### Changed (breaking)

- `low_level::clique_tree_sample` no longer takes a `pivot_diag` parameter.
- `CsrError` variants consolidated (20 → 12). `RowPtr*ExceedsU32`, `ColIndex*ExceedsU32`, `*ExceedsTargetIndexType` collapse into `IndexExceedsIndexType { kind: IndexKind }`. `NExceedsU32`, `NExceedsTargetIndexType`, `MatrixDimensionExceedsU32` collapse into `MatrixDimensionExceedsIndexType { n }`. `RowPtrNotRepresentableAsUsize` and `ColIndexNotRepresentableAsUsize` collapse into `IndexNotRepresentableAsUsize { kind, position }`. `RowIndexOutOfBounds` removed.
- `Error`, `ConfigError`, `CsrError`, `SolveError` are now `#[non_exhaustive]`; external `match` sites must add a wildcard arm.

### Added

- `IndexKind { RowPtr, ColIndex }` for disambiguating which CSR array an index error refers to.

### Removed

- `low_level::CdfSampler` — the `WeightedSampler` trait it implemented was crate-private, so external code could not wire it into anything.
- `low_level::EliminationSequence`, `low_level::EliminationStep` — factor-internal types with no external callers.
- `Factor::solve_into_with_projection` — folded into `solve_into` (always projects). For non-projecting solves, copy the RHS into the work buffer and call `solve_in_place`.
- `CsrRef::try_from_sprs`, `try_from_sprs_view`, `try_from_faer`, `try_from_faer_view` inherent methods — use the `TryFrom` impls instead (same conversions, same errors).
- `CsrRef::try_row` and `CsrRef::debug_validate` are no longer part of the public API.

### Fixed

- AC factorization no longer panics on marginally-SDD Laplacian inputs.

## [0.1.0] - 2026-03-10

Initial release of `approx-chol`, providing approximate Cholesky factorization
for graph Laplacians in Rust with Python bindings.

### Added

- Core approximate Cholesky factorization algorithm for SDD/SDDM matrices
- Fallible APIs with structured error types throughout
- CSR sparse matrix representation with checked accessors
- Triangular solve (`solve` and `solve_into`) for factored systems
- Clique sampling and star/clique-tree internals
- Serialization support via optional `serde` feature
- Python bindings (via PyO3/maturin) exposing factorization and solve
- Input validation and borrow-safe `solve_into` on the Python side
- Proptest suites for CSR construction and factorization invariants
- Preconditioner effectiveness tests
- Criterion benchmarks for factorization, sampling, solve, and CSR conversion
- Multi-platform CI with coverage reporting
- Dual MIT license for Rust crate and Python package

[Unreleased]: https://github.com/aai-institute/approx-chol/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.7.0
[0.6.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.6.0
[0.5.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.5.0
[0.4.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.4.0
[0.3.1]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.1
[0.3.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.0
[0.2.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.2.0
[0.1.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.1.0

[#35]: https://github.com/aai-institute/approx-chol/issues/35
[#36]: https://github.com/aai-institute/approx-chol/issues/36
[#37]: https://github.com/aai-institute/approx-chol/issues/37
[#38]: https://github.com/aai-institute/approx-chol/issues/38
[#39]: https://github.com/aai-institute/approx-chol/issues/39
[#41]: https://github.com/aai-institute/approx-chol/issues/41
[#50]: https://github.com/aai-institute/approx-chol/issues/50
[#61]: https://github.com/aai-institute/approx-chol/issues/61
[#62]: https://github.com/aai-institute/approx-chol/issues/62
[#75]: https://github.com/aai-institute/approx-chol/issues/75
[#78]: https://github.com/aai-institute/approx-chol/issues/78
[#80]: https://github.com/aai-institute/approx-chol/issues/80
[#82]: https://github.com/aai-institute/approx-chol/issues/82
[#83]: https://github.com/aai-institute/approx-chol/issues/83
[#84]: https://github.com/aai-institute/approx-chol/issues/84
[#85]: https://github.com/aai-institute/approx-chol/issues/85
[#89]: https://github.com/aai-institute/approx-chol/issues/89
[#91]: https://github.com/aai-institute/approx-chol/issues/91
[#92]: https://github.com/aai-institute/approx-chol/issues/92
[#93]: https://github.com/aai-institute/approx-chol/issues/93
[#109]: https://github.com/aai-institute/approx-chol/issues/109
[#118]: https://github.com/aai-institute/approx-chol/issues/118
[#131]: https://github.com/aai-institute/approx-chol/pull/131
[#135]: https://github.com/aai-institute/approx-chol/issues/135
[#140]: https://github.com/aai-institute/approx-chol/issues/140
[#146]: https://github.com/aai-institute/approx-chol/issues/146
[#163]: https://github.com/aai-institute/approx-chol/issues/163
[#179]: https://github.com/aai-institute/approx-chol/issues/179
[#180]: https://github.com/aai-institute/approx-chol/issues/180
[#181]: https://github.com/aai-institute/approx-chol/issues/181
[#224]: https://github.com/aai-institute/approx-chol/pull/224
[#183]: https://github.com/aai-institute/approx-chol/issues/183
[#182]: https://github.com/aai-institute/approx-chol/issues/182
[#148]: https://github.com/aai-institute/approx-chol/issues/148
[#228]: https://github.com/aai-institute/approx-chol/pull/228
[#255]: https://github.com/aai-institute/approx-chol/issues/255
[#267]: https://github.com/aai-institute/approx-chol/issues/267
[#275]: https://github.com/aai-institute/approx-chol/issues/275
[#277]: https://github.com/aai-institute/approx-chol/issues/277
[#300]: https://github.com/aai-institute/approx-chol/issues/300
[#301]: https://github.com/aai-institute/approx-chol/issues/301
[#306]: https://github.com/aai-institute/approx-chol/pull/306
[#311]: https://github.com/aai-institute/approx-chol/issues/311
[#315]: https://github.com/aai-institute/approx-chol/pull/315
[#322]: https://github.com/aai-institute/approx-chol/pull/322
[#327]: https://github.com/aai-institute/approx-chol/issues/327
[#343]: https://github.com/aai-institute/approx-chol/issues/343
[#418]: https://github.com/aai-institute/approx-chol/issues/418
[#419]: https://github.com/aai-institute/approx-chol/issues/419
[#420]: https://github.com/aai-institute/approx-chol/issues/420
[#421]: https://github.com/aai-institute/approx-chol/issues/421
[#422]: https://github.com/aai-institute/approx-chol/issues/422
[#423]: https://github.com/aai-institute/approx-chol/issues/423
[#424]: https://github.com/aai-institute/approx-chol/issues/424
[#425]: https://github.com/aai-institute/approx-chol/issues/425
[#426]: https://github.com/aai-institute/approx-chol/issues/426
[#427]: https://github.com/aai-institute/approx-chol/issues/427
[#428]: https://github.com/aai-institute/approx-chol/issues/428
[#429]: https://github.com/aai-institute/approx-chol/issues/429
[#430]: https://github.com/aai-institute/approx-chol/issues/430
[#431]: https://github.com/aai-institute/approx-chol/issues/431
[#432]: https://github.com/aai-institute/approx-chol/issues/432
[#433]: https://github.com/aai-institute/approx-chol/issues/433
[#435]: https://github.com/aai-institute/approx-chol/issues/435
[#437]: https://github.com/aai-institute/approx-chol/issues/437
[#472]: https://github.com/aai-institute/approx-chol/issues/472
