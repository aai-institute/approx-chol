# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.1.0
