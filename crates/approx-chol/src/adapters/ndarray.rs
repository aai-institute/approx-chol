//! ndarray arrays and views as blocks of right-hand sides.

use ndarray::{ArrayRef1, ArrayRef2, Axis, Zip};

use crate::factor::{Kernel, Slices};
use crate::{ApplyError, MatrixMut, Real};

/// An `Array2` or a view of one, at any strides, passed as `&mut *a`: columns of row stride 1 are solved as [`ColumnMajor`](crate::ColumnMajor)'s are, others through `read_column` and `write_column`. An `ArrayD` converts with `into_dimensionality::<Ix2>()`.
///
/// ndarray's inherent `rows` and `columns` shadow this trait's: call `MatrixMut::rows(&*a)`.
impl<T: Real> MatrixMut<T> for ArrayRef2<T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        self.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        Zip::from(into)
            .and(self.column(column))
            .for_each(|to, &from| *to = from);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        Zip::from(self.column_mut(column))
            .and(from)
            .for_each(|to, &from| *to = from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        match self.stride_of(Axis(0)) {
            1 => kernel.by_slices(&mut Contiguous(self)),
            _ => kernel.by_columns(self),
        }
    }
}

/// An ndarray block of row stride 1, each column a slice.
const SLICE: &str = "a column of row stride 1 is a slice";

struct Contiguous<'a, T>(&'a mut ArrayRef2<T>);

impl<T: Real> MatrixMut<T> for Contiguous<'_, T> {
    fn rows(&self) -> usize {
        self.0.nrows()
    }

    fn columns(&self) -> usize {
        self.0.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(self.0.column(column).as_slice().expect(SLICE));
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.column_mut(column).copy_from_slice(from);
    }
}

impl<T: Real> Slices<T> for Contiguous<'_, T> {
    fn column_mut(&mut self, column: usize) -> &mut [T] {
        (self.0.column_mut(column).into_slice()).expect(SLICE)
    }
}

/// An `Array1` or a view of one, at any stride, passed as `&mut *a`: one column, solved as its two-dimensional view of one column is.
impl<T: Real> MatrixMut<T> for ArrayRef1<T> {
    fn rows(&self) -> usize {
        self.len()
    }

    fn columns(&self) -> usize {
        1
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        (self.view().insert_axis(Axis(1))).read_column(column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        (self.view_mut().insert_axis(Axis(1))).write_column(column, from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        (self.view_mut().insert_axis(Axis(1))).solve_by(kernel)
    }
}
