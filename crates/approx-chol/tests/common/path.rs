//! Canonical 4-node path-graph Laplacian (0-1-2-3) fixture data.
//!
//! Shared across the CSR-validation, generic, and sprs/faer integration suites
//! so the matrix lives in exactly one place; each suite wraps it into the index
//! and value types it exercises.

/// Matrix dimension.
pub const N: u32 = 4;
/// CSR row pointers (length `N + 1`).
pub const ROW_PTRS: [usize; 5] = [0, 2, 5, 8, 10];
/// CSR column indices (length `nnz`).
pub const COL_INDICES: [usize; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// CSR values: tridiagonal path Laplacian with unit edge weights.
pub const VALUES: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
