//! Unit-test support, sourced from the files the integration suites, benches and
//! examples already share, so each fixture and helper exists once in the tree.

#[allow(dead_code)]
#[path = "../tests/common/path.rs"]
mod path;

/// 4-node path-graph Laplacian CSR `(row_ptrs, col_indices, values)`. Zero row
/// sums → no Gremban augmentation, so the factor keeps `n() == 4`.
pub(crate) fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    (
        path::ROW_PTRS.iter().map(|&v| v as u32).collect(),
        path::COL_INDICES.iter().map(|&v| v as u32).collect(),
        path::VALUES.to_vec(),
    )
}
