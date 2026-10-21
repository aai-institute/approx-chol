use approx_chol::input::{CsrError, CsrRef};
use approx_chol::Error;
use rstest::rstest;
use sprs::{CsMatI, SpIndex};

use super::{assert_solves_fixture, indices, COL_INDICES, N, ROW_PTRS, VALUES};

fn path_laplacian<I: SpIndex>() -> CsMatI<f64, I> {
    CsMatI::new(
        (N, N),
        indices(&ROW_PTRS),
        indices(&COL_INDICES),
        VALUES.to_vec(),
    )
}

fn run_case<I: SpIndex>() {
    assert_solves_fixture(&path_laplacian::<I>());
}

#[rstest]
#[case::u32(run_case::<u32>)]
#[case::usize(run_case::<usize>)]
#[case::u64(run_case::<u64>)]
fn sprs_csr_factorizes_over_index_types(#[case] run: fn()) {
    run();
}

/// A symmetric matrix's CSC arrays are its CSR arrays, so both factor alike.
#[test]
fn sprs_csc_factorizes_as_its_csr() {
    assert_solves_fixture(&path_laplacian::<u32>().to_csc());
}

#[test]
fn sprs_csc_of_an_asymmetric_matrix_is_rejected() {
    let csc =
        CsMatI::<f64, u32>::new_csc((2, 2), vec![0, 1, 3], vec![0, 0, 1], vec![2.0, -1.0, 2.0]);
    let err = approx_chol::factorize(&csc).expect_err("asymmetric CSC must be rejected");
    assert!(
        matches!(err, Error::InvalidCsr(CsrError::NotSymmetric { .. })),
        "{err:?}"
    );
}

/// Rows 2..4 of a 4x2 matrix are a square view whose row pointers start past zero.
#[test]
fn sprs_outer_slice_with_offset_rows_is_rejected() {
    let mat = CsMatI::<f64, u32>::new(
        (4, 2),
        vec![0, 1, 2, 4, 6],
        vec![0, 1, 0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
    );
    let err = CsrRef::try_from(mat.slice_outer(2..4)).expect_err("offset rows must be rejected");
    assert_eq!(
        err,
        Error::InvalidCsr(CsrError::RowPtrsMustStartAtZero { got: 2 })
    );
}

#[test]
fn sprs_try_from_non_square_returns_error() {
    let mat = CsMatI::<f64, u32>::new((3, 4), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3]);
    let err = CsrRef::try_from(&mat).expect_err("non-square matrix must be rejected");
    assert_eq!(
        err,
        Error::InvalidCsr(CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 })
    );
}
