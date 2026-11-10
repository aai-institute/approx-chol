use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, ApplyError, ApproxInverse, Threads};
use nalgebra::{DMatrix, DMatrixViewMut, DVector};
use nalgebra_sparse::CsrMatrix;

use super::{assert_solves_fixture, COL_INDICES, N, ROW_PTRS, VALUES};
use crate::common::bits::bits;
use crate::common::grid::grid;
use crate::common::solve::Solve;

/// A 5 × 6 grid, three vertices grounded, so it has floating and grounded parts alike.
fn grounded_grid() -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut graph = grid(5, 6);
    for (v, g) in [(0, 0.5), (13, 1.0), (29, 0.25)] {
        graph.ground[v] = g;
    }
    let (ptrs, columns, values) = graph.csr(false);
    (graph.n, ptrs, columns, values)
}

fn csr_matrix(n: usize, ptrs: &[usize], columns: &[usize], values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::try_from_csr_data(n, n, ptrs.to_vec(), columns.to_vec(), values.to_vec())
        .expect("valid nalgebra CSR")
}

fn rhs(n: usize, j: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7 + j * 13) % 11) as f64 - 5.0)
        .collect()
}

fn factor() -> ApproxInverse<f64> {
    let (n, ptrs, columns, values) = grounded_grid();
    factorize(CsrRef::new(n, &ptrs[..], &columns[..], &values[..]).unwrap()).unwrap()
}

#[test]
fn nalgebra_csr_factorizes() {
    assert_solves_fixture(&csr_matrix(N, &ROW_PTRS, &COL_INDICES, &VALUES));
}

#[test]
fn nalgebra_csr_factors_bit_for_bit_as_its_arrays() {
    let (n, ptrs, columns, values) = grounded_grid();
    let matrix = csr_matrix(n, &ptrs, &columns, &values);
    let viewed = factorize(CsrRef::try_from(&matrix).unwrap()).unwrap();
    let direct = factor();
    for j in 0..4 {
        let b = rhs(n, j);
        assert_eq!(
            bits(&viewed.solve(&b).unwrap()),
            bits(&direct.solve(&b).unwrap()),
            "column {j}"
        );
    }
}

#[test]
fn nalgebra_try_from_non_square_returns_error() {
    let matrix =
        CsrMatrix::try_from_csr_data(3, 4, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0; 3]).unwrap();
    let err = CsrRef::try_from(&matrix).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}

/// Each of `columns` right-hand sides applied alone, as a `Vec`.
fn alone(factor: &ApproxInverse<f64>, columns: usize) -> Vec<Vec<u64>> {
    (0..columns)
        .map(|j| bits(&factor.solve_at(&rhs(factor.n(), j), TWO).unwrap()))
        .collect()
}

const TWO: Threads = Threads::Count(NonZeroUsize::new(2).unwrap());

/// Nine columns: a pack of eight and one alone.
const K: usize = 9;

#[test]
fn apply_on_a_nalgebra_matrix_equals_apply_on_each_column() {
    let factor = factor();
    let n = factor.n();
    let mut block = DMatrix::from_fn(n, K, |i, j| rhs(n, j)[i]);
    factor.apply(&mut block, TWO).unwrap();
    let solved: Vec<_> = block.column_iter().map(|c| bits(c.as_slice())).collect();
    assert_eq!(solved, alone(&factor, K));
}

#[test]
fn apply_on_nalgebra_views_equals_apply_on_each_column() {
    let factor = factor();
    let n = factor.n();
    // A view of rows 1..=n of a larger matrix: columns apart by n + 2.
    let mut outer = DMatrix::from_fn(n + 2, K, |i, j| match (1..=n).contains(&i) {
        true => rhs(n, j)[i - 1],
        false => f64::NAN,
    });
    factor
        .apply(&mut outer.view_mut((1, 0), (n, K)), TWO)
        .unwrap();
    let solved: Vec<_> = (0..K)
        .map(|j| bits(&outer.column(j).as_slice()[1..=n]))
        .collect();
    assert_eq!(solved, alone(&factor, K));
    assert!(outer
        .row(0)
        .iter()
        .chain(outer.row(n + 1).iter())
        .all(|x| x.is_nan()));
    // Rows two entries apart, the slots between them NaN.
    let mut data: Vec<f64> = (0..2 * n * K)
        .map(|k| match k % 2 {
            0 => rhs(n, k / (2 * n))[k % (2 * n) / 2],
            _ => f64::NAN,
        })
        .collect();
    let mut strided = DMatrixViewMut::from_slice_with_strides_mut(&mut data, n, K, 2, 2 * n);
    factor.apply(&mut strided, TWO).unwrap();
    let solved: Vec<_> = (0..K)
        .map(|j| bits(&strided.column(j).iter().copied().collect::<Vec<_>>()))
        .collect();
    assert_eq!(solved, alone(&factor, K));
    assert!(data.iter().skip(1).step_by(2).all(|x| x.is_nan()));
}

#[test]
fn apply_on_a_nalgebra_vector_equals_apply() {
    let factor = factor();
    let mut x = DVector::from_vec(rhs(factor.n(), 0));
    factor.apply(&mut x, TWO).unwrap();
    assert_eq!(bits(x.as_slice()), alone(&factor, 1)[0]);
}

#[test]
fn apply_on_a_nalgebra_matrix_of_other_rows_leaves_it() {
    let factor = factor();
    let mut block = DMatrix::from_fn(factor.n() - 1, 2, |i, j| (i + j) as f64);
    let before = block.clone();
    let err = factor
        .apply(&mut block, Threads::All)
        .expect_err("rows differ");
    assert_eq!(
        err,
        ApplyError::Rows {
            expected: factor.n(),
            found: factor.n() - 1
        }
    );
    assert_eq!(block, before);
}
