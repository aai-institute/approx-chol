use approx_chol_next::input::{ConvertError, Csr};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use num_traits::PrimInt;
use rstest::rstest;

use super::{assert_solves_fixture, indices, COL_INDICES, N, ROW_PTRS, VALUES};

fn path_laplacian<I: faer::Index + PrimInt>(row_nnz: Option<Vec<I>>) -> SparseRowMat<I, f64> {
    let symbolic =
        SymbolicSparseRowMat::new_checked(N, N, indices(&ROW_PTRS), row_nnz, indices(&COL_INDICES));
    SparseRowMat::new(symbolic, VALUES.to_vec())
}

fn run_case<I: faer::Index + PrimInt>() {
    assert_solves_fixture(&path_laplacian::<I>(None));
}

#[rstest]
#[case::u32(run_case::<u32>)]
#[case::usize(run_case::<usize>)]
#[case::u64(run_case::<u64>)]
fn faer_csr_factorizes_over_index_types(#[case] run: fn()) {
    run();
}

#[test]
fn faer_csr_with_full_row_counts_factorizes() {
    let spans: Vec<usize> = ROW_PTRS.windows(2).map(|span| span[1] - span[0]).collect();
    assert_solves_fixture(&path_laplacian::<u32>(Some(indices(&spans))));
}

/// The spare slots hold `-1`s that would connect vertices 0 and 3 if read.
#[test]
fn faer_csr_with_spare_capacity_factorizes() {
    let mut col_indices: Vec<u32> = indices(&COL_INDICES);
    col_indices.extend([3, 0]);
    let mut values = VALUES.to_vec();
    values.extend([-1.0, -1.0]);
    let symbolic = SymbolicSparseRowMat::new_checked(N, N, indices(&ROW_PTRS), None, col_indices);
    assert_solves_fixture(&SparseRowMat::new(symbolic, values));
}

/// The identity, its unused slots holding `-1`s that would connect the two vertices if read.
#[test]
fn faer_try_from_rows_with_unused_slots_returns_error() {
    let symbolic = SymbolicSparseRowMat::<u32>::new_checked(
        2,
        2,
        vec![0, 2, 4],
        Some(vec![1, 1]),
        vec![0, 1, 1, 0],
    );
    let mat = SparseRowMat::new(symbolic, vec![1.0, -1.0, 1.0, -1.0]);
    let err = Csr::try_from(&mat).expect_err("unused slots must be rejected");
    assert_eq!(err, ConvertError::RowGap { row: 0 });
}

/// A 2x2 whose rows occupy slots 2..6, behind two slots no row owns.
#[test]
fn faer_try_from_offset_rows_returns_error() {
    let symbolic =
        SymbolicSparseRowMat::<u32>::new_checked(2, 2, vec![2, 4, 6], None, vec![0, 1, 0, 1, 0, 1]);
    let mat = SparseRowMat::new(symbolic, vec![9.0, 9.0, 1.0, -1.0, -1.0, 1.0]);
    let err = Csr::try_from(&mat).expect_err("offset rows must be rejected");
    assert_eq!(err, ConvertError::OffsetRows);
}

#[test]
fn faer_try_from_non_square_returns_error() {
    let symbolic =
        SymbolicSparseRowMat::<u32>::new_checked(3, 4, vec![0, 1, 2, 3], None, vec![0, 1, 0]);
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = Csr::try_from(&mat).expect_err("non-square matrix must be rejected");
    assert_eq!(err, ConvertError::NotSquare { rows: 3, cols: 4 });
}
