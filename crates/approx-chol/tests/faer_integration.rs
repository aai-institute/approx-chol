#![cfg(feature = "faer")]

#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use path_solve::assert_view_and_factor_match_fixture;

use approx_chol::{factorize, Config, CsrError, Error};
use faer::sparse::SparseRowMat;
use num_traits::{cast, Float, FromPrimitive, PrimInt};

/// Build a 4-node path graph Laplacian (0-1-2-3) as a faer sparse CSR matrix.
fn path_laplacian_faer<T, I>() -> SparseRowMat<I, T>
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: faer::Index + PrimInt,
{
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
    let values = path::VALUES
        .into_iter()
        .map(|v| T::from_f64(v).expect("value conversion"))
        .collect();

    let symbolic = faer::sparse::SymbolicSparseRowMat::<I>::new_checked(
        nrows,
        ncols,
        row_ptrs,
        None,
        col_indices,
    );
    SparseRowMat::new(symbolic, values)
}

fn run_case<T, I>()
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: faer::Index + PrimInt + 'static,
{
    let mat = path_laplacian_faer::<T, I>();
    assert_view_and_factor_match_fixture(&mat, Config::default());
}

/// One factorization per index type the adapter converts. The scalar is
/// forwarded untouched, so `generic_low_level_api` owns that axis.
#[test]
fn faer_csr_factorizes_over_index_types() {
    run_case::<f64, u32>();
    run_case::<f64, usize>();
    run_case::<f64, u64>();
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
    let err = factorize(&mat).expect_err("non-square matrix must be rejected");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 })
    ));
}
