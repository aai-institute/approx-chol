# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.2.0
[0.1.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.1.0
