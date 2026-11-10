use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, InputError};
use approx_chol::{factorize, ApproxInverse, Threads};
use nalgebra::{DMatrix, DMatrixViewMut, SMatrix};
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

const GRID: usize = 30;

fn csr_matrix(n: usize, ptrs: &[usize], columns: &[usize], values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::try_from_csr_data(n, n, ptrs.to_vec(), columns.to_vec(), values.to_vec())
        .expect("valid nalgebra CSR")
}

fn factor() -> ApproxInverse<f64> {
    let (n, ptrs, columns, values) = grounded_grid();
    factorize(CsrRef::new(n, &ptrs[..], &columns[..], &values[..]).unwrap()).unwrap()
}

/// Each of `factor`'s solves of four right-hand sides, bit for bit.
fn solved_bits(factor: &ApproxInverse<f64>) -> Vec<Vec<u64>> {
    let block = block(factor.n());
    (0..4)
        .map(|j| {
            bits(
                &factor
                    .solve(&block.column(j).iter().copied().collect::<Vec<_>>())
                    .unwrap(),
            )
        })
        .collect()
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
    assert_eq!(solved_bits(&viewed), solved_bits(&factor()));
}

#[test]
fn nalgebra_try_from_non_square_returns_error() {
    let csr =
        CsrMatrix::try_from_csr_data(3, 4, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0; 3]).unwrap();
    let err = CsrRef::try_from(&csr).expect_err("non-square matrix must be rejected");
    assert_eq!(err, InputError::NotSquare { rows: 3, cols: 4 });
}

const TWO: Threads = Threads::Count(NonZeroUsize::new(2).unwrap());

/// Nine columns: a pack of eight and one alone.
const K: usize = 9;

fn block(n: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, K, |i, j| ((i * 7 + j * 13) % 11) as f64 - 5.0)
}

/// The block applied as an owned matrix, whose columns the shared layouts check against each alone.
fn owned_bits(factor: &ApproxInverse<f64>) -> Vec<u64> {
    let mut owned = block(factor.n());
    factor.apply(&mut owned, TWO).unwrap();
    bits(owned.as_slice())
}

#[test]
fn apply_on_a_nalgebra_sub_view_leaves_the_rows_outside_it() {
    let factor = factor();
    let n = factor.n();
    let mut outer = DMatrix::from_element(n + 2, K, f64::NAN);
    let mut view = outer.view_mut((1, 0), (n, K));
    view.copy_from(&block(n));
    factor.apply(&mut view, TWO).unwrap();
    assert_eq!(bits(view.clone_owned().as_slice()), owned_bits(&factor));
    assert!(outer
        .row(0)
        .iter()
        .chain(outer.row(n + 1).iter())
        .all(|x| x.is_nan()));
}

#[test]
fn apply_on_a_nalgebra_strided_view_leaves_the_slots_between_rows() {
    let factor = factor();
    let n = factor.n();
    let mut data = vec![f64::NAN; 2 * n * K];
    let mut strided = DMatrixViewMut::from_slice_with_strides_mut(&mut data, n, K, 2, 2 * n);
    strided.copy_from(&block(n));
    factor.apply(&mut strided, TWO).unwrap();
    assert_eq!(bits(strided.clone_owned().as_slice()), owned_bits(&factor));
    assert!(data.iter().skip(1).step_by(2).all(|x| x.is_nan()));
}

#[test]
fn apply_on_a_static_nalgebra_matrix_equals_the_dynamic_one() {
    let factor = factor();
    let dynamic = block(GRID);
    let mut fixed = SMatrix::<f64, GRID, K>::from_fn(|i, j| dynamic[(i, j)]);
    factor.apply(&mut fixed, TWO).unwrap();
    assert_eq!(bits(fixed.as_slice()), owned_bits(&factor));
}
