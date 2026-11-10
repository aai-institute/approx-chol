use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, Threads};
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, SymbolicSparseRowMat};
use faer::Par;
use num_traits::PrimInt;
use rstest::rstest;

use super::{assert_solves_fixture, COL_INDICES, N, ROW_PTRS, VALUES};
use crate::common::bits::bits;
use crate::common::entry::cast_all;
use crate::common::graph::Graph;
use crate::common::grid::grid;

fn path_laplacian<I: faer::Index + PrimInt>(row_nnz: Option<Vec<I>>) -> SparseRowMat<I, f64> {
    let symbolic = SymbolicSparseRowMat::new_checked(
        N,
        N,
        cast_all(&ROW_PTRS),
        row_nnz,
        cast_all(&COL_INDICES),
    );
    SparseRowMat::new(symbolic, VALUES.to_vec())
}

fn run_case<I: faer::Index + PrimInt>() {
    assert_solves_fixture(path_laplacian::<I>(None).as_ref());
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
    assert_solves_fixture(path_laplacian::<u32>(Some(cast_all(&spans))).as_ref());
}

/// A symmetric matrix's CSC arrays are its CSR arrays.
#[test]
fn faer_csc_factorizes() {
    let symbolic = SymbolicSparseColMat::<u32>::new_checked(
        N,
        N,
        cast_all(&ROW_PTRS),
        None,
        cast_all(&COL_INDICES),
    );
    assert_solves_fixture(SparseColMat::new(symbolic, VALUES.to_vec()).as_ref());
}

/// A `rows × cols` grid grounded at vertex 0, so its matrix is SDDM and nonsingular.
fn grounded_grid(rows: usize, cols: usize) -> Graph {
    let mut graph = grid(rows, cols);
    graph.ground[0] = 0.5;
    graph
}

/// A grounded grid stored in faer's row and column layouts, and in the same arrays as a `CsrRef`.
#[test]
fn faer_row_and_column_storage_factor_as_the_same_csr() {
    let graph = grounded_grid(9, 7);
    let (row_ptrs, col_indices, values) = graph.csr(false);
    let n = graph.n;
    let rows = SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(n, n, row_ptrs.clone(), None, col_indices.clone()),
        values.clone(),
    );
    let columns = SparseColMat::new(
        SymbolicSparseColMat::new_checked(n, n, row_ptrs.clone(), None, col_indices.clone()),
        values.clone(),
    );
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let solve = |csr: CsrRef<'_, f64, usize>| {
        let mut x = b.clone();
        factorize(csr)
            .expect("SDDM")
            .apply(&mut x, Threads::All)
            .expect("n rows");
        bits(&x)
    };
    let direct = solve(CsrRef::new(n, &row_ptrs, &col_indices, &values).expect("valid"));
    assert_eq!(solve(rows.as_ref().try_into().expect("valid")), direct);
    assert_eq!(solve(columns.as_ref().try_into().expect("valid")), direct);
}

/// The identity stored by columns, its unused slots holding `-1`s that would connect the two vertices if read.
#[test]
fn faer_try_from_columns_with_unused_slots_returns_error() {
    let symbolic = SymbolicSparseColMat::<u32>::new_checked(
        2,
        2,
        vec![0, 2, 4],
        Some(vec![1, 1]),
        vec![0, 1, 1, 0],
    );
    let mat = SparseColMat::new(symbolic, vec![1.0, -1.0, 1.0, -1.0]);
    let err = CsrRef::try_from(mat.as_ref()).expect_err("unused slots must be rejected");
    assert_eq!(err, InputError::RowGap { row: 0 });
}

/// A column-major matrix's shape is checked as stored, before its arrays are read as its transpose's.
#[test]
fn faer_try_from_non_square_columns_returns_error() {
    let symbolic =
        SymbolicSparseColMat::<u32>::new_checked(3, 4, vec![0, 1, 2, 3, 3], None, vec![0, 1, 2]);
    let mat = SparseColMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = CsrRef::try_from(mat.as_ref()).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}

/// The spare slots hold `-1`s that would connect vertices 0 and 3 if read.
#[test]
fn faer_csr_with_spare_capacity_factorizes() {
    let mut col_indices: Vec<u32> = cast_all(&COL_INDICES);
    col_indices.extend([3, 0]);
    let mut values = VALUES.to_vec();
    values.extend([-1.0, -1.0]);
    let symbolic = SymbolicSparseRowMat::new_checked(N, N, cast_all(&ROW_PTRS), None, col_indices);
    assert_solves_fixture(SparseRowMat::new(symbolic, values).as_ref());
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
    let err = CsrRef::try_from(mat.as_ref()).expect_err("unused slots must be rejected");
    assert_eq!(err, InputError::RowGap { row: 0 });
}

/// A 2x2 whose rows occupy slots 2..6, behind two slots no row owns.
#[test]
fn faer_try_from_offset_rows_returns_error() {
    let symbolic =
        SymbolicSparseRowMat::<u32>::new_checked(2, 2, vec![2, 4, 6], None, vec![0, 1, 0, 1, 0, 1]);
    let mat = SparseRowMat::new(symbolic, vec![9.0, 9.0, 1.0, -1.0, -1.0, 1.0]);
    let err = CsrRef::try_from(mat.as_ref()).expect_err("offset rows must be rejected");
    assert_eq!(err, InputError::RowPtrs { row: 0 });
}

#[test]
fn faer_try_from_non_square_returns_error() {
    let symbolic =
        SymbolicSparseRowMat::<u32>::new_checked(3, 4, vec![0, 1, 2, 3], None, vec![0, 1, 0]);
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = CsrRef::try_from(mat.as_ref()).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}

#[test]
fn a_faer_par_is_a_thread_budget() {
    assert_eq!(Threads::from(Par::Seq), Threads::Count(NonZeroUsize::MIN));
    for n in [1, 3, 12] {
        let count = NonZeroUsize::new(n).unwrap();
        assert_eq!(Threads::from(Par::Rayon(count)), Threads::Count(count));
    }
    let pool = NonZeroUsize::new(rayon::current_num_threads()).unwrap();
    assert_eq!(Threads::from(Par::rayon(0)), Threads::Count(pool));
}
