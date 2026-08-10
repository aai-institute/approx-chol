# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Ingestion builds every adjacency list in one allocation instead of one per row, which
  is 11% of the build at low degree. ([#66])
- A block routed to the exact dense backend no longer gets an elimination graph built for
  it. ([#84])
- `Error::NonFiniteValue` no longer takes precedence over `Error::Asymmetric`,
  `Error::PositiveOffDiagonal` and `Error::NotDiagonallyDominant`. ([#84])
- The Python bindings build against pyo3 and numpy 0.29. ([#89])

### Fixed

- The star sampler draws from `[0, 1)`; the previous mapping could return exactly `1.0`. ([#109])
- A deserialized `Factor` whose dimension leaves no room for its ground vertex is rejected
  instead of overflowing. ([#59])

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

[Unreleased]: https://github.com/aai-institute/approx-chol/compare/v0.4.0...HEAD
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
[#59]: https://github.com/aai-institute/approx-chol/issues/59
[#61]: https://github.com/aai-institute/approx-chol/issues/61
[#62]: https://github.com/aai-institute/approx-chol/issues/62
[#75]: https://github.com/aai-institute/approx-chol/issues/75
[#78]: https://github.com/aai-institute/approx-chol/issues/78
[#80]: https://github.com/aai-institute/approx-chol/issues/80
[#82]: https://github.com/aai-institute/approx-chol/issues/82
[#83]: https://github.com/aai-institute/approx-chol/issues/83
[#66]: https://github.com/aai-institute/approx-chol/issues/66
[#84]: https://github.com/aai-institute/approx-chol/issues/84
[#85]: https://github.com/aai-institute/approx-chol/issues/85
[#89]: https://github.com/aai-institute/approx-chol/issues/89
[#91]: https://github.com/aai-institute/approx-chol/issues/91
[#92]: https://github.com/aai-institute/approx-chol/issues/92
[#93]: https://github.com/aai-institute/approx-chol/issues/93
[#109]: https://github.com/aai-institute/approx-chol/issues/109
