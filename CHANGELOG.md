# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)

- Disconnected input is supported: each connected component is factored, grounded and
  projected on its own in place of a global zero-mean projection. 0.3.1 coupled the
  components and silently solved a different system when a right-hand side was not
  zero-sum within each; solve output differs even where 0.3.1 was correct. (#35, #36)
- `solve`/`solve_into` cap the right-hand side at the original matrix dimension, and
  `solve_in_place` leaves one variable pinned per component.
- The serde representation of `Factor` is incompatible with 0.3.1, and the exact
  factor for a fixed seed can differ.
- `Config` gains a `backend` field, so struct literals need `..Config::default()`.
- `Error::Asymmetric`, `Error::NonFiniteValue`, `Error::NonFiniteRow` and
  `Error::NotDiagonallyDominant` reject input that 0.3.1 accepted, and duplicate
  entries are summed before the off-diagonal sign check.
- `Config::split_merge` of `Some(1)` selects standard AC in place of AC2 with one
  edge copy, and `Some(0)` selects it instead of erroring.
- `low_level::CliqueTreeSampler` replaces `low_level::clique_tree_sample` and
  `low_level::clique_tree_sample_multi`: `seed` and `split_merge` move to
  `CliqueTreeSampler::new`, and `sample` takes the star index and borrows its entries.
  Fill edges are unchanged for a given base seed and star index.
- `OwnedCsr::try_as_ref` is replaced by the infallible `as_csr_ref`, and
  `CsrRef::row_ptrs`/`col_indices`/`values` return slices with the view's lifetime.

### Removed

- `Error::InvalidConfig` and `ConfigError` — no `split_merge` value is invalid.

### Added

- `Config::backend` selects the per-component factorization: `Backend::ExactBelow { max_dim, on_failure }` by default, or `Backend::Approximate`. `ExactFailure` chooses whether an unusable pivot falls back to approximate elimination or fails with `Error::DenseFactorizationFailed`.
- `Factor::fallbacks`, `Fallback`, `UnusablePivot` and `DenseFailure` name each component that fell back and why its pivot was unusable.
- Python `Backend`, `ExactFailure`, `DenseFailure`, `Config(backend=...)`, `Factor.fallbacks` and `Fallback`; `factorize`/`factorize_raw` emit a `RuntimeWarning` per fallback.
- `From<&OwnedCsr> for CsrRef`, so `factorize`/`Builder::build` take `&OwnedCsr` directly. (#41)

### Fixed

- A structurally or numerically invalid `Factor` is rejected at deserialize time. (#37)
- `low_level::CliqueTreeSampler` no longer panics or emits non-finite fill on degenerate weights. (#38)
- `u32` nonzero/edge overflow now panics instead of silently truncating the factor. (#39)
- A strictly-dominant SDDM scaled below unit magnitude is augmented, and elimination
  keeps the scale of an underflowing diagonal or per-copy share. (#36)
- Approximate elimination keeps a small pivot's reciprocal wherever it is representable
  instead of substituting a scale of one. (#75)
- `validate_structure` rejects a tampered `Anchor`, an `original_n` unrelated to `n`, and
  non-finite or out-of-range factor values. (#80)
- Each block draws from its own sampler stream, so a fixed seed factors a block the same
  way under either backend. (#82)

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
