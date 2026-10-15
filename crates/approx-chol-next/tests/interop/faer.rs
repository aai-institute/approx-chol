use approx_chol_next::input::CsrError;
use approx_chol_next::FromCsrError;
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use num_traits::{cast, PrimInt};
use rstest::rstest;

use super::{assert_view_and_factor_match_fixture, factorize_matrix};

fn run_case<I: faer::Index + PrimInt>() {
    let index = |v| cast::<usize, I>(v).expect("index conversion");
    let symbolic = SymbolicSparseRowMat::<I>::new_checked(
        super::N,
        super::N,
        super::ROW_PTRS.into_iter().map(index).collect(),
        None,
        super::COL_INDICES.into_iter().map(index).collect(),
    );
    let mat = SparseRowMat::new(symbolic, super::VALUES.to_vec());
    assert_view_and_factor_match_fixture(&mat);
}

/// One factorization per index type the converter borrows; the scalar passes through untouched.
#[rstest]
#[case::u32(run_case::<u32>)]
#[case::usize(run_case::<usize>)]
#[case::u64(run_case::<u64>)]
fn faer_csr_factorizes_over_index_types(#[case] run: fn()) {
    run();
}

/// The identity, its unused slots holding `-1`s that would connect the two vertices if read.
#[test]
fn faer_factorize_rejects_rows_with_unused_slots_with_error() {
    let symbolic = SymbolicSparseRowMat::<u32>::new_checked(
        2,
        2,
        vec![0, 2, 4],
        Some(vec![1, 1]),
        vec![0, 1, 1, 0],
    );
    let mat = SparseRowMat::new(symbolic, vec![1.0, -1.0, 1.0, -1.0]);
    let err = factorize_matrix(&mat).expect_err("unused slots must be rejected");
    assert_eq!(err, FromCsrError::Csr(CsrError::RowGap { row: 0 }));
}

#[test]
fn faer_csr_with_full_row_counts_factorizes() {
    let row_ptrs: Vec<u32> = super::ROW_PTRS.iter().map(|&p| p as u32).collect();
    let row_nnz = row_ptrs.windows(2).map(|span| span[1] - span[0]).collect();
    let symbolic = SymbolicSparseRowMat::<u32>::new_checked(
        super::N,
        super::N,
        row_ptrs,
        Some(row_nnz),
        super::COL_INDICES.iter().map(|&j| j as u32).collect(),
    );
    let mat = SparseRowMat::new(symbolic, super::VALUES.to_vec());
    assert_view_and_factor_match_fixture(&mat);
}

#[test]
fn faer_factorize_rejects_non_square_with_error() {
    let symbolic =
        SymbolicSparseRowMat::<u32>::new_checked(3, 4, vec![0, 1, 2, 3], None, vec![0, 1, 0]);
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = factorize_matrix(&mat).expect_err("non-square matrix must be rejected");
    assert_eq!(
        err,
        FromCsrError::Csr(CsrError::NotSquare { rows: 3, cols: 4 })
    );
}
