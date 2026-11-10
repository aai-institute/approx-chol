//! faer's sparse matrices as [`CsrRef`] views, its dense ones as blocks of right-hand sides, and its parallelism as a [`Threads`] budget.

use std::num::NonZeroUsize;

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::{SparseColMatRef, SparseRowMatRef};
use faer::{Col, ColMut, ContiguousFwd, Mat, MatMut, MatRef, Par};
use num_traits::PrimInt;

use super::square;
use crate::factor::{Kernel, Slices};
use crate::input::{CsrRef, InputError};
use crate::{ApplyError, MatrixMut, Real, Threads};

impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: SparseRowMatRef<'a, I, T>) -> Result<Self, InputError> {
        rows(square(matrix.nrows(), matrix.ncols())?, matrix)
    }
}

impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<SparseColMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: SparseColMatRef<'a, I, T>) -> Result<Self, InputError> {
        rows(square(matrix.nrows(), matrix.ncols())?, matrix.transpose())
    }
}

/// The `n` rows of a square `matrix` as a [`CsrRef`].
fn rows<'a, T: Real, I: faer::Index + PrimInt>(
    n: usize,
    matrix: SparseRowMatRef<'a, I, T>,
) -> Result<CsrRef<'a, T, I>, InputError> {
    let symbolic = matrix.symbolic();
    let row_ptrs = symbolic.row_ptr();
    // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
    if let Some(row) = (0..n).find(|&i| symbolic.row_range(i).end != row_ptrs[i + 1].zx()) {
        return Err(InputError::RowGap { row });
    }
    // faer may keep spare capacity past the last row.
    let end = row_ptrs[n].zx();
    let (col_indices, values) = (&symbolic.col_idx()[..end], &matrix.val()[..end]);
    CsrRef::new(n, row_ptrs, col_indices, values)
}

/// faer's `Par::Seq` is `Count(1)`, `Par::Rayon(n)` is `Count(n)`.
impl From<Par> for Threads {
    fn from(par: Par) -> Self {
        Threads::Count(NonZeroUsize::new(par.degree()).unwrap_or(NonZeroUsize::MIN))
    }
}

fn read_column<T: Real>(block: MatRef<'_, T>, column: usize, into: &mut [T]) {
    assert_eq!(into.len(), block.nrows(), "one entry per row");
    (into.iter_mut().zip(block.col(column).iter())).for_each(|(to, &from)| *to = from);
}

impl<T: Real> MatrixMut<T> for MatMut<'_, T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        self.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        read_column(self.rb(), column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        assert_eq!(from.len(), self.nrows(), "one entry per row");
        (self.rb_mut().col_mut(column).iter_mut().zip(from)).for_each(|(to, &from)| *to = from);
    }

    /// The path [`MatrixMut`] states for a faer block, by its row stride.
    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        match self.rb_mut().try_as_col_major_mut() {
            Some(columns) => kernel.by_slices(&mut Contiguous(columns)),
            None => kernel.by_columns(self),
        }
    }
}

/// A faer block of contiguous columns, each a slice.
struct Contiguous<'a, T>(MatMut<'a, T, usize, usize, ContiguousFwd>);

impl<T: Real> MatrixMut<T> for Contiguous<'_, T> {
    fn rows(&self) -> usize {
        self.0.nrows()
    }

    fn columns(&self) -> usize {
        self.0.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(self.0.rb().col(column).as_slice());
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.column_mut(column).copy_from_slice(from);
    }
}

impl<T: Real> Slices<T> for Contiguous<'_, T> {
    fn column_mut(&mut self, column: usize) -> &mut [T] {
        self.0.rb_mut().col_mut(column).as_slice_mut()
    }
}

impl<T: Real> MatrixMut<T> for Mat<T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        self.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        read_column(self.as_ref(), column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.as_mut().write_column(column, from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        self.as_mut().solve_by(kernel)
    }
}

impl<T: Real> MatrixMut<T> for ColMut<'_, T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        1
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        read_column(self.rb().as_mat(), column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.rb_mut().as_mat_mut().write_column(column, from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        self.rb_mut().as_mat_mut().solve_by(kernel)
    }
}

impl<T: Real> MatrixMut<T> for Col<T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        1
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        read_column(self.as_mat(), column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.as_mat_mut().write_column(column, from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        self.as_mat_mut().solve_by(kernel)
    }
}
