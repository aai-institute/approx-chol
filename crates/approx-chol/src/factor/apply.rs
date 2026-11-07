//! Blocks of right-hand sides an [`ApproxInverse`] applies to, and how each reaches the solve.

use super::ApproxInverse;
use crate::{Batch, MemStack, Real, Threads};

/// Errors an apply returns, leaving the block as it was.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// A block whose row count is not the operator's dimension.
    Rows {
        /// The operator's dimension.
        expected: usize,
        /// The block's row count.
        found: usize,
    },
    /// Scratch the apply cannot hold: a stack too small, or a requirement past memory.
    Scratch,
}

impl core::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Rows { expected, found } => {
                write!(f, "the block has {found} rows, expected {expected}")
            }
            Self::Scratch => write!(
                f,
                "the apply's scratch does not fit: a stack too small, or past memory"
            ),
        }
    }
}

impl std::error::Error for ApplyError {}

/// A block of right-hand sides, the solutions written over them.
///
/// It keeps its shape during an apply; `read_column` copies one column into `into`, `write_column` copies `from` into one column and touches nothing else, and neither allocates. Each panics unless `column < columns()` and the slice has `rows()` entries. A slice or `Vec` of `T` is one column; one of `[T; K]` holds `K` interleaved columns, solved where they are.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a block of right-hand sides of `{T}`",
    label = "not a `MatrixMut<{T}>`",
    note = "a slice or `Vec` of `{T}` is one column, one of `[{T}; K]` holds `K`; pass an array as `&mut a[..]`"
)]
pub trait MatrixMut<T: Real> {
    /// Rows, the operator's dimension when applied.
    fn rows(&self) -> usize;
    /// Columns, each a right-hand side.
    fn columns(&self) -> usize;
    /// Copies column `column` into `into`.
    fn read_column(&self, column: usize, into: &mut [T]);
    /// Copies `from` into column `column`.
    fn write_column(&mut self, column: usize, from: &[T]);

    /// Solves the block through `kernel`; a layout the solve reads in place overrides it.
    #[doc(hidden)]
    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        kernel.by_columns(self);
    }
}

impl<T: Real> MatrixMut<T> for [T] {
    fn rows(&self) -> usize {
        self.len()
    }

    fn columns(&self) -> usize {
        1
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        assert_eq!(column, 0, "a slice is one column");
        into.copy_from_slice(self);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        assert_eq!(column, 0, "a slice is one column");
        self.copy_from_slice(from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        kernel.in_place(self);
    }
}

impl<T: Real, const K: usize> MatrixMut<T> for [[T; K]] {
    fn rows(&self) -> usize {
        self.len()
    }

    fn columns(&self) -> usize {
        K
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        assert!(column < K, "column {column} of {K}");
        assert_eq!(into.len(), self.len(), "one entry per row");
        (into.iter_mut().zip(self)).for_each(|(to, row)| *to = row[column]);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        assert!(column < K, "column {column} of {K}");
        assert_eq!(from.len(), self.len(), "one entry per row");
        (self.iter_mut().zip(from)).for_each(|(row, &value)| row[column] = value);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        kernel.in_place(self);
    }
}

impl<T: Real, E> MatrixMut<T> for Vec<E>
where
    [E]: MatrixMut<T>,
{
    fn rows(&self) -> usize {
        self.as_slice().rows()
    }

    fn columns(&self) -> usize {
        self.as_slice().columns()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        self.as_slice().read_column(column, into);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.as_mut_slice().write_column(column, from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        self.as_mut_slice().solve_by(kernel);
    }
}

/// One apply's solve, handed to [`MatrixMut::solve_by`]; only the crate makes one, for a block it checked.
#[doc(hidden)]
pub struct Kernel<'a, T> {
    pub(super) inverse: &'a ApproxInverse<T>,
    pub(super) threads: Threads,
    pub(super) stack: &'a mut MemStack,
}

impl<T: Real> Kernel<'_, T> {
    /// Solves rows of `V`, one right-hand side per lane, where they are.
    pub(crate) fn in_place<V: Batch<T>>(self, rows: &mut [V]) {
        self.inverse.solve_unchecked(rows, self.threads, self.stack);
    }

    /// Solves `block` one column at a time through a column of scratch.
    pub(crate) fn by_columns<M: MatrixMut<T> + ?Sized>(self, block: &mut M) {
        // An empty system takes no column, so a stack that holds none serves it.
        if self.inverse.n() == 0 {
            return;
        }
        let (mut column, rest) = self.stack.make_with(self.inverse.n(), |_| T::zero());
        for j in 0..block.columns() {
            block.read_column(j, &mut column);
            self.inverse
                .solve_unchecked(&mut column, self.threads, rest);
            block.write_column(j, &column);
        }
    }
}
