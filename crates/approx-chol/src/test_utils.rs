//! Unit-test support sourced from the fixtures the integration suites, benches and examples share.

#[allow(dead_code)]
#[path = "../tests/common/path.rs"]
mod path;

/// 4-node path Laplacian `(row_ptrs, col_indices, values)`; zero row sums keep `n() == 4`.
pub(crate) fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    (
        path::ROW_PTRS.iter().map(|&v| v as u32).collect(),
        path::COL_INDICES.iter().map(|&v| v as u32).collect(),
        path::VALUES.to_vec(),
    )
}
