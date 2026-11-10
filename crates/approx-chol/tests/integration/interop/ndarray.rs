use approx_chol::input::CsrRef;
use approx_chol::{factorize, ApproxInverse, MatrixMut, MemStack};
use ndarray::{s, Array1, Array2, ArrayRef2, ShapeBuilder};

use crate::common::bits::bits;
use crate::common::grid::grid;
use crate::common::scratch::least_slots;
use crate::common::threads::ONE;

/// A 5 × 6 grid, three vertices grounded, so it has floating and grounded parts alike.
fn factor() -> ApproxInverse<f64> {
    let mut graph = grid(5, 6);
    for (v, g) in [(0, 0.5), (13, 1.0), (29, 0.25)] {
        graph.ground[v] = g;
    }
    let (ptrs, columns, values) = graph.csr(false);
    factorize(CsrRef::new(graph.n, &ptrs[..], &columns[..], &values[..]).unwrap()).unwrap()
}

/// Nine columns: a pack of eight and one alone.
const K: usize = 9;

fn value(i: usize, j: usize) -> f64 {
    ((i * 7 + j * 13) % 11) as f64 - 5.0
}

/// The block's columns, bit for bit, read through the trait.
fn column_bits(block: &ArrayRef2<f64>) -> Vec<Vec<u64>> {
    let read = |j| {
        let mut column = vec![0.0; block.nrows()];
        MatrixMut::read_column(block, j, &mut column);
        bits(&column)
    };
    (0..block.ncols()).map(read).collect()
}

/// The block applied as an owned column-major array.
fn owned_bits(factor: &ApproxInverse<f64>) -> Vec<Vec<u64>> {
    let mut owned = Array2::from_shape_fn((factor.n(), K).f(), |(i, j)| value(i, j));
    factor.apply(&mut *owned, ONE).unwrap();
    column_bits(&owned)
}

/// Rows `1..=n` of a taller array in either order, its padding rows NaN, solve as the owned block and leave the padding.
#[test]
fn apply_on_an_ndarray_row_slice_leaves_the_rows_outside_it() {
    let factor = factor();
    let n = factor.n();
    for shape in [(n + 2, K).set_f(false), (n + 2, K).f()] {
        let mut outer = Array2::from_elem(shape, f64::NAN);
        let mut view = outer.slice_mut(s![1..n + 1, ..]);
        view.assign(&Array2::from_shape_fn((n, K), |(i, j)| value(i, j)));
        factor.apply(&mut *view, ONE).unwrap();
        assert_eq!(column_bits(&view), owned_bits(&factor));
        assert!(outer
            .row(0)
            .iter()
            .chain(outer.row(n + 1))
            .all(|x| x.is_nan()));
    }
}

#[test]
fn apply_on_an_ndarray_stepped_view_leaves_the_rows_between() {
    let factor = factor();
    let n = factor.n();
    let mut outer = Array2::from_elem((2 * n, K).f(), f64::NAN);
    let mut stepped = outer.slice_mut(s![..;2, ..]);
    stepped.assign(&Array2::from_shape_fn((n, K), |(i, j)| value(i, j)));
    factor.apply(&mut *stepped, ONE).unwrap();
    assert_eq!(column_bits(&stepped), owned_bits(&factor));
    assert!(outer.slice(s![1..;2, ..]).iter().all(|x| x.is_nan()));
}

/// A view with its rows reversed, row stride -1, or its columns reversed, solves to the bits of the same values owned.
#[test]
fn apply_on_a_reversed_ndarray_view_equals_the_values_owned() {
    let factor = factor();
    let n = factor.n();
    let fresh = Array2::from_shape_fn((n, K), |(i, j)| value(i, j));
    for shape in [(n, K).set_f(false), (n, K).f()] {
        for reversed in [s![..;-1, ..], s![.., ..;-1]] {
            let mut outer = Array2::from_elem(shape, f64::NAN);
            let mut view = outer.slice_mut(reversed);
            view.assign(&fresh);
            factor.apply(&mut *view, ONE).unwrap();
            assert_eq!(column_bits(&view), owned_bits(&factor));
        }
    }
}

/// An array without columns, of the operator's rows or not, in either order, is left as it is, even on an empty stack.
#[test]
fn an_ndarray_array_without_columns_is_left_as_it_is() {
    let factor = factor();
    let n = factor.n();
    for rows in [n, n + 1] {
        for shape in [(rows, 0).set_f(false), (rows, 0).f()] {
            let mut empty = Array2::<f64>::zeros(shape);
            factor
                .apply_with(&mut *empty, ONE, MemStack::new(&mut []))
                .unwrap();
            assert_eq!(empty.dim(), (rows, 0));
        }
    }
}

/// An `Array1`, contiguous or a stepped view, is one column, solved as a `Vec` is.
#[test]
fn an_ndarray_array1_applies_as_a_vec() {
    let factor = factor();
    let n = factor.n();
    let column: Vec<f64> = (0..n).map(|i| value(i, 0)).collect();
    let mut vec = column.clone();
    factor.apply(&mut vec, ONE).unwrap();
    let mut array = Array1::from(column.clone());
    factor.apply(&mut *array, ONE).unwrap();
    assert_eq!(bits(array.as_slice().unwrap()), bits(&vec));
    let mut outer = Array1::from_elem(2 * n, f64::NAN);
    let mut stepped = outer.slice_mut(s![..;2]);
    stepped.assign(&Array1::from(column));
    factor.apply(&mut *stepped, ONE).unwrap();
    assert_eq!(bits(&stepped.to_vec()), bits(&vec));
    assert!(outer.slice(s![1..;2]).iter().all(|x| x.is_nan()));
}

/// Columns of row stride 1 fill packs from where they are, a single one solved in place as a `Vec` is; strided ones are staged through a column more.
#[test]
fn ndarray_contiguous_columns_solve_where_they_are() {
    let factor = factor();
    let n = factor.n();
    for k in [1, K] {
        let most = factor.scratch(k, ONE);
        let columns = Array2::from_shape_fn((n, k).f(), |(i, j)| value(i, j));
        let packed = least_slots::<f64>(most, |stack| {
            factor.apply_with(&mut *columns.clone(), ONE, stack)
        });
        let mut outer = Array2::from_elem((2 * n, k).f(), 0.0);
        let staged = least_slots::<f64>(most, |stack| {
            factor.apply_with(&mut *outer.slice_mut(s![..;2, ..]), ONE, stack)
        });
        assert_eq!(
            staged,
            packed + n,
            "{k} columns: a strided block takes one column more"
        );
        if k == 1 {
            let vec: Vec<f64> = (0..n).map(|i| value(i, 0)).collect();
            let in_place = least_slots::<f64>(most, |stack| {
                factor.apply_with(&mut vec.clone(), ONE, stack)
            });
            assert_eq!(packed, in_place, "a contiguous column solves where it is");
            let array = Array1::from(vec);
            let as_array1 = least_slots::<f64>(most, |stack| {
                factor.apply_with(&mut *array.clone(), ONE, stack)
            });
            assert_eq!(as_array1, in_place, "an Array1 solves where it is");
        }
    }
}
