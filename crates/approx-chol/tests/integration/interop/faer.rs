use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, ApplyError, ApproxInverse, MemBuffer, MemStack, Threads};
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, SymbolicSparseRowMat};
use faer::{Col, Mat, MatRef, Par};
use num_traits::PrimInt;
use rstest::rstest;

use super::{assert_solves_fixture, COL_INDICES, N, ROW_PTRS, VALUES};
use crate::common::bits::bits;
use crate::common::entry::cast_all;
use crate::common::graph::Graph;
use crate::common::grid::grid;
use crate::common::scratch::least_slots;

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

/// A grounded grid's approximate inverse and its column `j` of right-hand sides.
fn grid_inverse() -> (ApproxInverse<f64>, impl Fn(usize) -> Vec<f64>) {
    let graph = grounded_grid(9, 7);
    let (row_ptrs, col_indices, values) = graph.csr(false);
    let csr = CsrRef::new(graph.n, &row_ptrs, &col_indices, &values).expect("valid");
    let n = graph.n;
    let column = move |j: usize| (0..n).map(|i| ((i * 7 + j * 13) as f64).sin()).collect();
    (factorize(csr).expect("SDDM"), column)
}

/// Each column applied alone as a `Vec`.
fn alone(factor: &ApproxInverse<f64>, column: Vec<f64>) -> Vec<u64> {
    let mut x = column;
    factor
        .apply(&mut x, Threads::Count(NonZeroUsize::MIN))
        .expect("n rows");
    bits(&x)
}

fn column_bits(block: MatRef<'_, f64>) -> Vec<Vec<u64>> {
    let column = |j| bits(&block.col(j).iter().copied().collect::<Vec<_>>());
    (0..block.ncols()).map(column).collect()
}

/// faer blocks of 1 to 9 columns, contiguous or strided, solve each column as it solves alone.
#[test]
fn faer_blocks_apply_each_column_as_alone() {
    let (factor, column) = grid_inverse();
    let n = factor.n();
    let threads = Threads::Count(NonZeroUsize::new(3).unwrap());
    for k in [1, 3, 4, 9] {
        let expected: Vec<Vec<u64>> = (0..k).map(|j| alone(&factor, column(j))).collect();
        let columns: Vec<Vec<f64>> = (0..k).map(&column).collect();
        let fresh = Mat::from_fn(n, k, |i, j| columns[j][i]);

        let mut mat = fresh.clone();
        factor.apply(&mut mat, threads).expect("n rows");
        assert_eq!(column_bits(mat.as_ref()), expected, "Mat, {k} columns");

        let mut buffer = MemBuffer::new(factor.scratch(k, Threads::Count(NonZeroUsize::MIN)));
        let mut mat = fresh.clone();
        let stack = MemStack::new(&mut buffer);
        factor
            .apply_with(&mut mat.as_mut(), threads, stack)
            .expect("held");
        assert_eq!(column_bits(mat.as_ref()), expected, "MatMut, {k} columns");

        // Row-major storage: a row stride of `k`, a column stride of 1.
        let mut rows = fresh.transpose().to_owned();
        factor
            .apply(&mut rows.as_mut().transpose_mut(), threads)
            .expect("n rows");
        assert_eq!(
            column_bits(rows.as_ref().transpose()),
            expected,
            "row-major, {k} columns"
        );

        // Rows 1..=n of a taller block, the padding rows untouched.
        let mut tall = Mat::from_fn(n + 2, k, |i, j| match (1..=n).contains(&i) {
            true => columns[j][i - 1],
            false => f64::NAN,
        });
        factor
            .apply(&mut tall.as_mut().subrows_mut(1, n), threads)
            .expect("n rows");
        assert_eq!(
            column_bits(tall.as_ref().subrows(1, n)),
            expected,
            "inner rows, {k} columns"
        );
        assert!((0..k).all(|j| tall[(0, j)].is_nan() && tall[(n + 1, j)].is_nan()));
    }
}

/// Contiguous faer columns fill packs from where they are, a single one solved in place as a `Vec` is; strided ones are staged through a column more.
#[test]
fn faer_contiguous_columns_solve_where_they_are() {
    let (factor, column) = grid_inverse();
    let n = factor.n();
    let one = Threads::Count(NonZeroUsize::MIN);
    for k in [1, 9] {
        let columns: Vec<Vec<f64>> = (0..k).map(&column).collect();
        let contiguous = Mat::from_fn(n, k, |i, j| columns[j][i]);
        let row_major = contiguous.transpose().to_owned();
        assert_ne!(row_major.as_ref().transpose().row_stride(), 1);

        let most = factor.scratch(k, one);
        let packed = least_slots::<f64>(most, |stack| {
            factor.apply_with(&mut contiguous.clone(), one, stack)
        });
        let staged = least_slots::<f64>(most, |stack| {
            let mut fresh = row_major.clone();
            factor.apply_with(&mut fresh.as_mut().transpose_mut(), one, stack)
        });
        assert_eq!(
            staged,
            packed + n,
            "{k} columns: a staged block takes one column more"
        );
        if k == 1 {
            let in_place = least_slots::<f64>(most, |stack| {
                factor.apply_with(&mut columns[0].clone(), one, stack)
            });
            assert_eq!(packed, in_place, "a contiguous column solves where it is");
            let col = Col::from_fn(n, |i| columns[0][i]);
            let as_col = least_slots::<f64>(most, |stack| {
                factor.apply_with(&mut col.clone(), one, stack)
            });
            assert_eq!(as_col, in_place, "a Col solves where it is");
            let as_col_mut = least_slots::<f64>(most, |stack| {
                let mut fresh = col.clone();
                factor.apply_with(&mut fresh.as_mut(), one, stack)
            });
            assert_eq!(as_col_mut, in_place, "a ColMut solves where it is");
        }
    }
}

/// faer's columns, owned and borrowed, solve as a `Vec` does.
#[test]
fn faer_columns_apply_as_a_vec() {
    let (factor, column) = grid_inverse();
    let expected = vec![alone(&factor, column(0))];
    let fresh = Col::from_fn(factor.n(), |i| column(0)[i]);
    let mut col = fresh.clone();
    factor.apply(&mut col, Threads::All).expect("n rows");
    assert_eq!(column_bits(col.as_mat()), expected, "Col");
    let mut col = fresh.clone();
    factor
        .apply(&mut col.as_mut(), Threads::All)
        .expect("n rows");
    assert_eq!(column_bits(col.as_mat()), expected, "ColMut");
}

/// A faer block of the wrong row count errs and is left as it was.
#[test]
fn faer_block_of_other_rows_returns_error() {
    let (factor, _) = grid_inverse();
    let n = factor.n();
    let mut mat = Mat::from_fn(n + 1, 2, |i, j| (i + j) as f64);
    let error = factor.apply(&mut mat, Threads::All);
    assert_eq!(
        error,
        Err(ApplyError::Rows {
            expected: n,
            found: n + 1
        })
    );
    assert!((0..2).all(|j| (0..=n).all(|i| mat[(i, j)] == (i + j) as f64)));
}
