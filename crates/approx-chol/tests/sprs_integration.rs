#![cfg(feature = "sprs")]

#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use path_solve::assert_view_and_factor_match_fixture;

use approx_chol::{CsrError, CsrRef};

fn path_laplacian_sprs<I: sprs::SpIndex>() -> sprs::CsMatI<f64, I> {
    let n = path::N as usize;
    let indptr = path::ROW_PTRS.into_iter().map(I::from_usize).collect();
    let indices = path::COL_INDICES.into_iter().map(I::from_usize).collect();
    sprs::CsMatI::new((n, n), indptr, indices, path::VALUES.to_vec())
}

fn run_case<I: sprs::SpIndex + num_traits::PrimInt + 'static>() {
    let mat = path_laplacian_sprs::<I>();
    assert_view_and_factor_match_fixture(&mat);
}

/// One factorization per index type the adapter converts; the scalar passes through untouched.
#[test]
fn sprs_csr_factorizes_over_index_types() {
    run_case::<u32>();
    run_case::<usize>();
    run_case::<u64>();
}

#[test]
fn sprs_factorize_rejects_csc_with_error() {
    let csr = path_laplacian_sprs::<u32>();
    let csc = csr.to_csc();
    let err = CsrRef::try_from(&csc).expect_err("CSC must be rejected");
    assert!(matches!(err, CsrError::ExpectedCsrMatrixGotCsc));
}

#[test]
fn sprs_try_from_non_square_returns_error() {
    let mat = sprs::CsMatI::<f64, u32>::new((3, 4), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3]);
    let err = CsrRef::try_from(&mat).expect_err("non-square matrix must be rejected");
    assert!(matches!(
        err,
        CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 }
    ));
}
