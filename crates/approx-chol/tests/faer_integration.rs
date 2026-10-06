#![cfg(feature = "faer")]

#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use path_solve::assert_view_and_factor_match_fixture;

use approx_chol::{Config, CsrError, CsrRef};
use faer::sparse::SparseRowMat;
use num_traits::{cast, PrimInt};

fn path_laplacian_faer<I: faer::Index + PrimInt>() -> SparseRowMat<I, f64> {
    let nrows = path::N as usize;
    let ncols = path::N as usize;
    let row_ptrs = path::ROW_PTRS
        .into_iter()
        .map(|v| cast::<usize, I>(v).expect("index conversion"))
        .collect();
    let col_indices = path::COL_INDICES
        .into_iter()
        .map(|v| cast::<usize, I>(v).expect("index conversion"))
        .collect();

    let symbolic = faer::sparse::SymbolicSparseRowMat::<I>::new_checked(
        nrows,
        ncols,
        row_ptrs,
        None,
        col_indices,
    );
    SparseRowMat::new(symbolic, path::VALUES.to_vec())
}

fn run_case<I: faer::Index + PrimInt + 'static>() {
    let mat = path_laplacian_faer::<I>();
    assert_view_and_factor_match_fixture(&mat, Config::default());
}

/// One factorization per index type the adapter converts. The scalar is
/// forwarded untouched, so `generic_api` owns that axis.
#[test]
fn faer_csr_factorizes_over_index_types() {
    run_case::<u32>();
    run_case::<usize>();
    run_case::<u64>();
}

#[test]
fn faer_view_rejects_non_square_with_error() {
    let symbolic = faer::sparse::SymbolicSparseRowMat::<u32>::new_checked(
        3,
        4,
        vec![0u32, 1, 2, 3],
        None,
        vec![0u32, 1, 0],
    );
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = CsrRef::try_from(&mat).expect_err("non-square matrix must be rejected");
    assert_eq!(err, CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 });
}
