#![cfg(feature = "faer")]

#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use path_solve::assert_view_and_factor_match_fixture;

use approx_chol::{CsrError, CsrRef};
use faer::sparse::SparseRowMat;
use num_traits::{cast, PrimInt};

fn run_case<I: faer::Index + PrimInt + 'static>() {
    let index = |v| cast::<usize, I>(v).expect("index conversion");
    let symbolic = faer::sparse::SymbolicSparseRowMat::<I>::new_checked(
        path::N as usize,
        path::N as usize,
        path::ROW_PTRS.into_iter().map(index).collect(),
        None,
        path::COL_INDICES.into_iter().map(index).collect(),
    );
    let mat = SparseRowMat::new(symbolic, path::VALUES.to_vec());
    assert_view_and_factor_match_fixture(&mat);
}

/// One factorization per index type the adapter converts; the scalar passes through untouched.
#[test]
fn faer_csr_factorizes_over_index_types() {
    run_case::<u32>();
    run_case::<usize>();
    run_case::<u64>();
}

#[test]
fn faer_factorize_rejects_non_square_with_error() {
    let symbolic = faer::sparse::SymbolicSparseRowMat::<u32>::new_checked(
        3,
        4,
        vec![0u32, 1, 2, 3],
        None,
        vec![0u32, 1, 0],
    );
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = CsrRef::try_from(&mat).expect_err("non-square matrix must be rejected");
    assert!(matches!(
        err,
        CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 }
    ));
}
