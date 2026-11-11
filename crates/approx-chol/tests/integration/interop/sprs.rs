use approx_chol::input::{CsrRef, InputError};
use rstest::rstest;
use sprs::{CsMatI, SpIndex};

use super::path::{assert_solves_fixture, COL_INDICES, N, ROW_PTRS, VALUES};
use crate::common::entry::cast_all;

fn path_laplacian<I: SpIndex>() -> CsMatI<f64, I> {
    CsMatI::new(
        (N, N),
        cast_all(&ROW_PTRS),
        cast_all(&COL_INDICES),
        VALUES.to_vec(),
    )
}

fn run_case<I: SpIndex>() {
    assert_solves_fixture(path_laplacian::<I>().view());
}

#[rstest]
#[case::u32(run_case::<u32>)]
#[case::usize(run_case::<usize>)]
#[case::u64(run_case::<u64>)]
fn sprs_csr_factorizes_over_index_types(#[case] run: fn()) {
    run();
}

/// A CSC read as a CSR would be its transpose, which the 8-ulp symmetry check cannot tell from the matrix.
#[test]
fn sprs_csc_is_rejected_as_stored_by_columns_and_converts_by_to_csr() {
    let csc = path_laplacian::<u32>().to_csc();
    let err = CsrRef::try_from(csc.view()).expect_err("a CSC view is rejected");
    assert_eq!(err, InputError::StoredByColumns);
    assert_solves_fixture(csc.to_csr().view());
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
    let err = CsrRef::try_from(mat.slice_outer(2..4)).expect_err("offset rows are rejected");
    assert_eq!(
        err,
        InputError::RowPtrs {
            parameter: "row_ptrs",
            row: 0
        }
    );
}

#[test]
fn sprs_try_from_non_square_returns_error() {
    let mat = CsMatI::<f64, u32>::new((3, 4), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3]);
    let err = CsrRef::try_from(mat.view()).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}

/// The shape is checked before the storage order.
#[test]
fn sprs_try_from_non_square_csc_returns_not_square() {
    let mat = CsMatI::<f64, u32>::new_csc((3, 4), vec![0, 1, 2, 3, 3], vec![0, 1, 2], vec![1.0; 3]);
    let err = CsrRef::try_from(mat.view()).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}
