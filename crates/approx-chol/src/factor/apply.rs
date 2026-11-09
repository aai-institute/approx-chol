//! Blocks of right-hand sides an [`ApproxInverse`] applies to, and how each reaches the solve.

use super::spread::Spread;
use super::ApproxInverse;
use crate::{Batch, MemStack, Real, StackReq, Threads};

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

/// Errors forming a [`ColumnMajor`] or [`Columns`] block.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Column-major data that is no whole number of columns.
    Length {
        /// The rows of a column.
        rows: usize,
        /// The data's entries.
        found: usize,
    },
    /// A column of other rows than the first.
    Column {
        /// Its index.
        column: usize,
        /// The first column's rows.
        expected: usize,
        /// Its entries.
        found: usize,
    },
}

impl core::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length { rows, found } => {
                write!(
                    f,
                    "{found} entries are no whole number of columns of {rows} rows"
                )
            }
            Self::Column {
                column,
                expected,
                found,
            } => write!(
                f,
                "column {column} has {found} entries, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// A block of right-hand sides, the solutions written over them.
///
/// It keeps its shape during an apply; `read_column` copies one column into `into`, `write_column` copies `from` into one column and touches nothing else, and neither allocates. Each panics unless `column < columns()` and the slice has `rows()` entries. A slice or `Vec` of `T` is one column; one of `[T; K]` holds `K` interleaved columns, solved where they are; [`ColumnMajor`] and [`Columns`] solve each column where it is.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a block of right-hand sides of `{T}`",
    label = "not a `MatrixMut<{T}>`",
    note = "a slice or `Vec` of `{T}` is one column, one of `[{T}; K]` holds `K`; pass an array as `&mut a[..]`; wrap column-major data in `ColumnMajor`, one slice per column in `Columns`"
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
        kernel.in_place([self]);
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
        kernel.in_place([self]);
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

/// Columns of `rows` entries one after another in `data`.
#[derive(Debug)]
pub struct ColumnMajor<'a, T> {
    data: &'a mut [T],
    rows: usize,
}

impl<'a, T> ColumnMajor<'a, T> {
    /// `data` read column-major, `rows` to a column; [`LayoutError::Length`] unless that makes a whole number of columns, none without data.
    pub fn new(data: &'a mut [T], rows: usize) -> Result<Self, LayoutError> {
        let whole = match rows {
            0 => data.is_empty(),
            rows => data.len() % rows == 0,
        };
        match whole {
            true => Ok(Self { data, rows }),
            false => Err(LayoutError::Length {
                rows,
                found: data.len(),
            }),
        }
    }
}

impl<T: Real> ColumnMajor<'_, T> {
    fn column(&self, column: usize) -> core::ops::Range<usize> {
        let columns = self.columns();
        assert!(column < columns, "column {column} of {columns}");
        column * self.rows..(column + 1) * self.rows
    }
}

impl<T: Real> MatrixMut<T> for ColumnMajor<'_, T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.data.len().checked_div(self.rows).unwrap_or(0)
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(&self.data[self.column(column)]);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        let column = self.column(column);
        self.data[column].copy_from_slice(from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        kernel.in_place(self.data.chunks_exact_mut(self.rows));
    }
}

/// One slice per column, all of the first's rows: `&mut [T]`, `Vec<T>`, `[T; N]` or any other `AsMut<[T]>`.
#[derive(Debug)]
pub struct Columns<'s, C> {
    columns: &'s mut [C],
}

impl<'s, C> Columns<'s, C> {
    /// `columns` as a block; [`LayoutError::Column`] at the first not of the first's rows.
    pub fn new<T>(columns: &'s mut [C]) -> Result<Self, LayoutError>
    where
        C: AsRef<[T]>,
    {
        let rows = first_rows(columns);
        let other = columns
            .iter()
            .position(|column| column.as_ref().len() != rows);
        match other {
            None => Ok(Self { columns }),
            Some(column) => Err(LayoutError::Column {
                column,
                expected: rows,
                found: columns[column].as_ref().len(),
            }),
        }
    }
}

fn first_rows<T, C: AsRef<[T]>>(columns: &[C]) -> usize {
    columns.first().map_or(0, |column| column.as_ref().len())
}

impl<T: Real, C: AsRef<[T]> + AsMut<[T]>> MatrixMut<T> for Columns<'_, C> {
    fn rows(&self) -> usize {
        first_rows(self.columns)
    }

    fn columns(&self) -> usize {
        self.columns.len()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(self.columns[column].as_ref());
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.columns[column].as_mut().copy_from_slice(from);
    }

    fn solve_by(&mut self, kernel: Kernel<'_, T>) {
        kernel.in_place(self.columns.iter_mut().map(AsMut::as_mut));
    }
}

/// What a solve at a budget takes from the stack, from which each path's need follows.
pub(super) struct Room {
    /// A column of the operator's rows.
    pub(super) column: StackReq,
    /// Per lane, the most a block past the budget's share takes alone.
    pub(super) alone: StackReq,
    /// Per lane, the most one of the others takes, on each thread solving them at once.
    pub(super) region: StackReq,
    /// Threads solving the others at once.
    pub(super) regions: usize,
}

impl Room {
    pub(super) fn of<T: Real>(inverse: &ApproxInverse<T>, threads: Threads) -> Self {
        Spread::of(inverse, threads).room(inverse)
    }

    /// Rows of `lanes` right-hand sides each, solved where they are.
    pub(super) fn in_place(&self, lanes: usize) -> StackReq {
        let regions = self.region.array(lanes).array(self.regions);
        self.alone.array(lanes).or(regions)
    }

    /// A block copied one column at a time through a column.
    pub(super) fn staged(&self) -> StackReq {
        self.column.and(self.in_place(1))
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
    /// Solves each run of rows of `V`, one right-hand side per lane, where it is, one after another.
    pub(crate) fn in_place<'r, V: Batch<T> + 'r>(
        self,
        runs: impl IntoIterator<Item = &'r mut [V]>,
    ) {
        for rows in runs {
            self.inverse.solve_unchecked(rows, self.threads, self.stack);
        }
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
