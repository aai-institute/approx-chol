# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `TryFrom<&OwnedCsr> for CsrRef`, so `factorize`/`Builder::build` take `&OwnedCsr` directly. (#41)
- Exact Cholesky for small blocks via `Config::backend`
  (`Backend::ExactBelow { max_dim }`, default `24`).
- `ExactFailure` selects fallback or error on an invalid exact pivot.
- `Factor::exact_fallbacks` records fallbacks; Python warns with `RuntimeWarning`.
- `Error::Asymmetric`, `Error::NonFiniteValue`, `Error::NonFiniteRow` and
  `Error::NotDiagonallyDominant` reject input that was previously accepted.

### Fixed

- `solve`/`solve_into` ground SDDM systems against the auxiliary vertex instead of
  applying a global zero-mean projection. (#35)
- Each connected component of a disconnected Laplacian is factored independently. (#36)
- A strictly-dominant SDDM scaled below unit magnitude is augmented. (#36)
- A structurally-invalid `Factor` is rejected at deserialize time. (#37)
- `low_level` clique-tree samplers no longer panic or emit non-finite fill on degenerate weights. (#38)
- `u32` nonzero/edge overflow now panics instead of silently truncating the factor. (#39)
- `solve`/`solve_into` project an out-of-range right-hand side onto the range.
- Edge splitting no longer reaches the exact backend, where `weight / k` underflowed
  a subnormal weight to zero.
- A dense block whose `anchor` or `ground` does not match its omitted vertex is
  rejected at deserialize time.
- A block too large to assemble densely falls back to approximate elimination
  instead of failing the factorization.

### Changed

- `Config` gains a public `backend` field, breaking exhaustive struct literals.
- The serde representation of `Factor` is incompatible with earlier releases.
- `solve`/`solve_into` cap the right-hand side at the original matrix dimension,
  not the augmented one.
- `solve_in_place` leaves one variable pinned per block; for a floating block which
  variable that is depends on `Backend`.
- A row surplus at or below `min(1e-10 * row_scale, sqrt(f64::EPSILON))` is
  rounding noise, not dominance, so the system is not augmented.
- Duplicate entries are summed before the off-diagonal sign check.
- `Error::Disconnected` is removed; disconnected input is factored per component.

### Performance

- Canonical CSR is ingested without a reordering buffer.
- Connectivity is tested with one traversal instead of enumerating components.

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

[Unreleased]: https://github.com/aai-institute/approx-chol/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.1
[0.3.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.0
[0.2.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.2.0
[0.1.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.1.0
