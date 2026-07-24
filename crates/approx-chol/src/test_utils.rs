pub(crate) trait OrPanic<T> {
    fn or_panic(self, context: &str) -> T;
}

impl<T, E: core::fmt::Debug> OrPanic<T> for Result<T, E> {
    fn or_panic(self, context: &str) -> T {
        match self {
            Ok(value) => value,
            Err(err) => panic!("{context}: {err:?}"),
        }
    }
}

impl<T> OrPanic<T> for Option<T> {
    fn or_panic(self, context: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{context}"),
        }
    }
}

/// 4-node path-graph Laplacian CSR `(row_ptrs, col_indices, values)`. Zero row
/// sums → no Gremban augmentation, so the factor keeps `n() == 4`.
pub(crate) fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    (
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    )
}
