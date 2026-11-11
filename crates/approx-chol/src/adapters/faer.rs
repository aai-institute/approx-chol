//! faer's row-major sparse matrices as [`CsrRef`] views, its dense ones as blocks of right-hand sides, its parallelism as a [`Threads`] budget, and an [`ApproxInverse`] as faer's operator.

use std::num::NonZeroUsize;

use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::SparseRowMatRef;
use faer::traits::ComplexField;
use faer::{Col, ColMut, ContiguousFwd, Mat, MatMut, MatRef, Par};
use num_traits::PrimInt;

use super::square;
use crate::factor::{Kernel, Slices};
use crate::input::{CsrNames, CsrRef, InputError};
use crate::{ApplyError, ApproxInverse, MatrixMut, MemStack, Real, StackReq, Threads};

/// faer's names of a `SparseRowMatRef`'s arrays.
const NAMES: CsrNames = CsrNames {
    row_ptrs: "row_ptr",
    col_indices: "col_idx",
    values: "val",
};

impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: SparseRowMatRef<'a, I, T>) -> Result<Self, InputError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let symbolic = matrix.symbolic();
        let row_ptrs = symbolic.row_ptr();
        // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
        if let Some(row) = (0..n).find(|&i| symbolic.row_range(i).end != row_ptrs[i + 1].zx()) {
            return Err(InputError::RowGap { row });
        }
        // faer may keep spare capacity past the last row.
        let end = row_ptrs[n].zx();
        let (col_indices, values) = (&symbolic.col_idx()[..end], &matrix.val()[..end]);
        CsrRef::named(n, row_ptrs, col_indices, values, NAMES)
    }
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

/// faer's `MatMut`, and through it `Mat`, `Col` and `ColMut`, at any strides: columns of row stride 1 are solved as [`ColumnMajor`](crate::ColumnMajor)'s are, others through `read_column` and `write_column`; a typed faer view converts with `as_dyn_mut().as_dyn_stride_mut()`.
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

/// faer's operators return nothing, so an error panics, as faer's own do.
#[track_caller]
fn or_panic(result: Result<(), ApplyError>) {
    if let Err(error) = result {
        panic!("{error}");
    }
}

#[track_caller]
fn apply_in_place<T: Real>(
    inverse: &ApproxInverse<T>,
    mut rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) {
    or_panic(inverse.apply_with(&mut rhs, par.into(), stack));
}

/// `rhs` copied into `out` and applied there; with columns, either of other rows than `n` panics as the apply errs, before the copy.
#[track_caller]
fn apply<T: Real + ComplexField>(
    inverse: &ApproxInverse<T>,
    mut out: MatMut<'_, T>,
    rhs: MatRef<'_, T>,
    par: Par,
    stack: &mut MemStack,
) {
    let n = inverse.n();
    if rhs.ncols() > 0 {
        for found in [rhs.nrows(), out.nrows()] {
            if found != n {
                or_panic(Err(ApplyError::Rows { expected: n, found }));
            }
        }
    }
    out.copy_from(rhs);
    apply_in_place(inverse, out, par, stack);
}

// B is real and symmetric: its conjugate, transpose and adjoint all apply as B does.
impl<T: Real + ComplexField> LinOp<T> for ApproxInverse<T> {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.scratch(rhs_ncols, par.into())
    }

    fn nrows(&self) -> usize {
        self.n()
    }

    fn ncols(&self) -> usize {
        self.n()
    }

    #[track_caller]
    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        apply(self, out, rhs, par, stack);
    }

    #[track_caller]
    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        apply(self, out, rhs, par, stack);
    }
}

impl<T: Real + ComplexField> BiLinOp<T> for ApproxInverse<T> {
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }

    #[track_caller]
    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        apply(self, out, rhs, par, stack);
    }

    #[track_caller]
    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        apply(self, out, rhs, par, stack);
    }
}

impl<T: Real + ComplexField> Precond<T> for ApproxInverse<T> {
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }

    #[track_caller]
    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply_in_place(self, rhs, par, stack);
    }

    #[track_caller]
    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply_in_place(self, rhs, par, stack);
    }
}

impl<T: Real + ComplexField> BiPrecond<T> for ApproxInverse<T> {
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }

    #[track_caller]
    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply_in_place(self, rhs, par, stack);
    }

    #[track_caller]
    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply_in_place(self, rhs, par, stack);
    }
}
