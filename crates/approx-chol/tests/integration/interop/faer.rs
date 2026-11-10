use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, ApplyError, ApproxInverse, MemBuffer, MemStack, StackReq, Threads};
use faer::matrix_free::conjugate_gradient::{
    conjugate_gradient, conjugate_gradient_scratch, CgParams,
};
use faer::matrix_free::{BiLinOp, BiPrecond, IdentityPrecond, LinOp, Precond};
use faer::sparse::{
    SparseColMat, SparseRowMat, SparseRowMatRef, SymbolicSparseColMat, SymbolicSparseRowMat,
};
use faer::{Col, ColRef, Mat, MatMut, MatRef, Par};
use num_traits::PrimInt;
use rstest::rstest;

use super::path::{assert_solves_fixture, COL_INDICES, N, ROW_PTRS, VALUES};
use crate::common::bits::bits;
use crate::common::entry::cast_all;
use crate::common::graph::Graph;
use crate::common::grid::grid;
use crate::common::scratch::{least_slots, nan_cells, nan_slots};

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
    assert_eq!(
        err,
        InputError::RowPtrs {
            parameter: "row_ptrs",
            row: 0
        }
    );
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

type Apply = fn(&ApproxInverse<f64>, MatMut<'_, f64>, MatRef<'_, f64>, Par, &mut MemStack);
type InPlace = fn(&ApproxInverse<f64>, MatMut<'_, f64>, Par, &mut MemStack);
type Scratch = fn(&ApproxInverse<f64>, usize, Par) -> StackReq;

/// faer's out-of-place methods, each with its scratch query.
const APPLIES: [(&str, Apply, Scratch); 4] = [
    ("apply", LinOp::apply, LinOp::apply_scratch),
    ("conj_apply", LinOp::conj_apply, LinOp::apply_scratch),
    (
        "transpose_apply",
        BiLinOp::transpose_apply,
        BiLinOp::transpose_apply_scratch,
    ),
    (
        "adjoint_apply",
        BiLinOp::adjoint_apply,
        BiLinOp::transpose_apply_scratch,
    ),
];

/// faer's in-place methods, each with its scratch query.
const IN_PLACE: [(&str, InPlace, Scratch); 4] = [
    (
        "apply_in_place",
        Precond::apply_in_place,
        Precond::apply_in_place_scratch,
    ),
    (
        "conj_apply_in_place",
        Precond::conj_apply_in_place,
        Precond::apply_in_place_scratch,
    ),
    (
        "transpose_apply_in_place",
        BiPrecond::transpose_apply_in_place,
        BiPrecond::transpose_apply_in_place_scratch,
    ),
    (
        "adjoint_apply_in_place",
        BiPrecond::adjoint_apply_in_place,
        BiPrecond::transpose_apply_in_place_scratch,
    ),
];

/// Each of faer's operator methods, on a NaN-filled stack its own scratch query sizes, equals `apply`, bit for bit.
#[test]
fn faer_operator_methods_equal_apply() {
    let (factor, column) = grid_inverse();
    let n = factor.n();
    for par in [Par::Seq, Par::rayon(3)] {
        for k in [1, 5, 9] {
            let columns: Vec<Vec<f64>> = (0..k).map(&column).collect();
            let rhs = Mat::from_fn(n, k, |i, j| columns[j][i]);
            let mut expected = rhs.clone();
            factor.apply(&mut expected, par.into()).expect("n rows");
            let expected = column_bits(expected.as_ref());
            let own = factor.scratch(k, par.into());
            for (name, _, scratch) in APPLIES.iter().copied() {
                assert_eq!(scratch(&factor, k, par), own, "{name}'s scratch");
            }
            for (name, _, scratch) in IN_PLACE.iter().copied() {
                assert_eq!(
                    scratch(&factor, k, par),
                    own,
                    "{name}'s scratch, no temporary"
                );
            }
            for (name, apply, scratch) in APPLIES {
                let mut cells = nan_cells::<f64>(scratch(&factor, k, par));
                let mut out = Mat::from_fn(n, k, |_, _| f64::NAN);
                apply(
                    &factor,
                    out.as_mut(),
                    rhs.as_ref(),
                    par,
                    MemStack::new_any(&mut cells),
                );
                assert_eq!(
                    column_bits(out.as_ref()),
                    expected,
                    "{name}, {k} columns, {par:?}"
                );
            }
            for (name, apply, scratch) in IN_PLACE {
                let mut cells = nan_cells::<f64>(scratch(&factor, k, par));
                let mut block = rhs.clone();
                apply(&factor, block.as_mut(), par, MemStack::new_any(&mut cells));
                assert_eq!(
                    column_bits(block.as_ref()),
                    expected,
                    "{name}, {k} columns, {par:?}"
                );
            }
        }
    }
}

/// faer's operators return nothing, so a block of other rows panics with `ApplyError`'s message.
#[test]
#[should_panic(expected = "the block has 64 rows, expected 63")]
fn faer_operator_on_other_rows_panics() {
    let (factor, _) = grid_inverse();
    let mut buffer = MemBuffer::new(factor.scratch(1, Threads::All));
    let mut block = Mat::<f64>::zeros(factor.n() + 1, 1);
    Precond::apply_in_place(
        &factor,
        block.as_mut(),
        Par::Seq,
        MemStack::new(&mut buffer),
    );
}

/// `LinOp::apply` checks `out`'s rows before copying `rhs` into it, so a mismatch panics with `ApplyError`'s message.
#[test]
#[should_panic(expected = "the block has 64 rows, expected 63")]
fn faer_apply_into_other_rows_panics() {
    let (factor, column) = grid_inverse();
    let mut cells = nan_cells::<f64>(factor.scratch(1, Threads::All));
    let rhs = ColRef::from_slice(&column(0)).as_mat().to_owned();
    let mut out = Mat::<f64>::zeros(factor.n() + 1, 1);
    LinOp::apply(
        &factor,
        out.as_mut(),
        rhs.as_ref(),
        Par::Seq,
        MemStack::new_any(&mut cells),
    );
}

/// A stack too small for the block, here a pack of four columns, panics with `ApplyError`'s message.
#[test]
#[should_panic(expected = "the apply's scratch does not fit")]
fn faer_operator_on_a_short_stack_panics() {
    let (factor, column) = grid_inverse();
    let mut block = Mat::from_fn(factor.n(), 4, |i, j| column(j)[i]);
    Precond::apply_in_place(
        &factor,
        block.as_mut(),
        Par::Seq,
        MemStack::new_any(&mut nan_slots::<f64>(0)),
    );
}

/// faer's CG on `a` preconditioned by `precond`, from zero, to `‖Ax − b‖ ≤ 1e-10 ‖b‖`: its solution and iterations.
fn faer_cg(
    precond: impl Precond<f64>,
    a: SparseRowMatRef<'_, usize, f64>,
    b: &[f64],
) -> (Vec<f64>, usize) {
    let par = Par::rayon(2);
    let mut buffer = MemBuffer::new(conjugate_gradient_scratch(&precond, a, 1, par));
    let mut x = Col::zeros(b.len());
    let params = CgParams {
        abs_tolerance: 0.0,
        rel_tolerance: 1e-10,
        max_iters: 1000,
        ..Default::default()
    };
    let info = conjugate_gradient(
        x.as_mat_mut(),
        precond,
        a,
        ColRef::from_slice(b).as_mat(),
        params,
        |_| {},
        par,
        MemStack::new(&mut buffer),
    )
    .expect("converges");
    (x.iter().copied().collect(), info.iter_count)
}

/// faer's CG preconditioned by an approximate inverse, passed by reference, converges on a grounded grid in a third of the unpreconditioned iterations.
#[test]
fn faer_cg_preconditioned_by_an_approx_inverse_converges() {
    let graph = grounded_grid(30, 30);
    let (row_ptrs, col_indices, values) = graph.csr(false);
    let n = graph.n;
    let a = SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(n, n, row_ptrs, None, col_indices),
        values,
    );
    let factor = factorize(CsrRef::try_from(a.as_ref()).expect("valid")).expect("SDDM");
    let b: Vec<f64> = (0..n).map(|i| ((i * 7) as f64).sin()).collect();
    let (x, preconditioned) = faer_cg(&factor, a.as_ref(), &b);
    assert!(graph.relative_residual(&x, &b) < 1e-9, "true residual");
    let (_, plain) = faer_cg(IdentityPrecond { dim: n }, a.as_ref(), &b);
    assert!(
        preconditioned * 3 < plain,
        "{preconditioned} iterations preconditioned, {plain} plain"
    );
}
