//! Blocks of right-hand sides an [`ApproxInverse`] applies to, and how each reaches the solve.

use super::spread::Spread;
use super::ApproxInverse;
use crate::threads::ONE;
use crate::{Batch, MemBuffer, MemStack, Real, StackReq, Threads};

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

// Built inline, either error cost within's solves 5–9%.
#[cold]
#[inline(never)]
pub(super) fn rows_error(found: usize, n: usize) -> ApplyError {
    ApplyError::Rows { expected: n, found }
}

#[cold]
#[inline(never)]
fn scratch_error() -> ApplyError {
    ApplyError::Scratch
}

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
/// It keeps its shape during an apply; `read_column` copies one column into `into`, `write_column` copies `from` into one column and touches nothing else, and neither allocates. Each panics unless `column < columns()` and the slice has `rows()` entries. A slice or `Vec` of `T` is one column; one of `[T; K]` holds `K` interleaved columns, solved where they are; any other block, [`ColumnMajor`] and [`Columns`] among them, is solved in interleaved packs of its columns copied into scratch.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a block of right-hand sides of `{T}`",
    label = "not a `MatrixMut<{T}>`",
    note = "a slice or `Vec` of `{T}` is one column, one of `[{T}; K]` holds `K`; pass an array as `&mut a[..]`; wrap column-major data in `ColumnMajor`, one slice per column in `Columns`; a nalgebra 0.35 matrix needs the `nalgebra` feature"
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

    /// Solves the block through `kernel`, or leaves it as it was when the scratch its path takes does not fit; a layout whose columns the solve reaches where they are overrides it.
    #[doc(hidden)]
    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        kernel.by_columns(self)
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

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        kernel.in_place(self)
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

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        kernel.in_place(self)
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

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        self.as_mut_slice().solve_by(kernel)
    }
}

/// Every nalgebra matrix, owned or a view, of any shape and strides.
#[cfg(feature = "nalgebra")]
impl<T, R, C, S> MatrixMut<T> for nalgebra::Matrix<T, R, C, S>
where
    T: Real,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorageMut<T, R, C>,
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        self.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        let shape = (nalgebra::Dyn(self.nrows()), nalgebra::Const::<1>);
        let rows = into.len();
        nalgebra::DVectorViewMut::from_slice(into, rows)
            .copy_from(&self.generic_view((0, column), shape));
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.column_mut(column).copy_from_slice(from);
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

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        kernel.by_slices(self)
    }
}

impl<T: Real> Slices<T> for ColumnMajor<'_, T> {
    fn column_mut(&mut self, column: usize) -> &mut [T] {
        let column = self.column(column);
        &mut self.data[column]
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

    fn solve_by(&mut self, kernel: Kernel<'_, T>) -> Result<(), ApplyError> {
        kernel.by_slices(self)
    }
}

impl<T: Real, C: AsRef<[T]> + AsMut<[T]>> Slices<T> for Columns<'_, C> {
    fn column_mut(&mut self, column: usize) -> &mut [T] {
        self.columns[column].as_mut()
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

    /// A block of `columns` columns whose slices fill its packs, a column alone solved where it is.
    pub(super) fn copied(&self, columns: usize) -> StackReq {
        match pack_lanes(columns) {
            1 => self.in_place(1),
            lanes => self.column.array(lanes).and(self.in_place(lanes)),
        }
    }

    /// A block of `columns` columns read and written through a column.
    pub(super) fn staged(&self, columns: usize) -> StackReq {
        self.column.and(self.copied(columns))
    }
}

/// Columns a copied block solves at once: the fused solve measured 2.5x faster per column at 8 than one at a time, 2.1x at 4.
const PACK: usize = 8;

/// The widest pack a remainder short of [`PACK`] takes; fewer columns go one at a time.
const HALF: usize = PACK / 2;

/// Columns of the widest pack a copied block with `left` columns to go takes next.
fn pack_lanes(left: usize) -> usize {
    match left {
        PACK.. => PACK,
        HALF.. => HALF,
        _ => 1,
    }
}

/// One apply's solve, handed to [`MatrixMut::solve_by`]; only the crate makes one, for a block of the operator's rows.
#[doc(hidden)]
pub struct Kernel<'a, T> {
    pub(super) inverse: &'a ApproxInverse<T>,
    pub(super) threads: Threads,
    pub(super) stack: Stack<'a>,
}

/// Where an apply's scratch comes from.
pub(super) enum Stack<'a> {
    /// The caller's, which holds what the block's path takes at one thread or the apply errs.
    Holds(&'a mut MemStack),
    /// Allocated for what the block's path takes at the budget.
    Allocate,
}

impl<T: Real> Kernel<'_, T> {
    /// Runs `solve` on a stack that holds what `need` asks of the solve's room, or errs before it touches the block.
    fn on(
        self,
        need: impl Fn(&Room) -> StackReq,
        solve: impl FnOnce(&ApproxInverse<T>, Threads, &mut MemStack),
    ) -> Result<(), ApplyError> {
        let Self {
            inverse,
            threads,
            stack,
        } = self;
        let mut allocated;
        let stack = match stack {
            Stack::Holds(stack) => {
                let need = need(&Room::of(inverse, ONE));
                // An overflowing requirement's alignment is 0, on which can_hold panics.
                if need.align_bytes() == 0 || !stack.can_hold(need) {
                    return Err(scratch_error());
                }
                stack
            }
            Stack::Allocate => {
                let need = need(&Room::of(inverse, threads));
                allocated = MemBuffer::try_new(need).map_err(|_| scratch_error())?;
                MemStack::new(&mut allocated)
            }
        };
        solve(inverse, threads, stack);
        Ok(())
    }

    /// Solves `rows` of `V`, one right-hand side per lane, where they are.
    fn in_place<V: Batch<T>>(self, rows: &mut [V]) -> Result<(), ApplyError> {
        self.on(
            |room| room.in_place(V::LANES),
            |inverse, threads, stack| inverse.solve_unchecked(rows, threads, stack),
        )
    }

    /// Solves `block` in packs filled from and drained into its own column slices.
    fn by_slices<S: Slices<T>>(self, block: &mut S) -> Result<(), ApplyError> {
        let columns = block.columns();
        self.on(
            |room| room.copied(columns),
            |inverse, threads, stack| packs(inverse, &mut Sliced(block), threads, stack),
        )
    }

    /// Solves `block` in packs, each column read and written through a column of scratch.
    fn by_columns<M: MatrixMut<T> + ?Sized>(self, block: &mut M) -> Result<(), ApplyError> {
        let columns = block.columns();
        self.on(
            |room| room.staged(columns),
            |inverse, threads, stack| {
                let (mut column, rest) = stack.make_with(inverse.n(), |_| T::zero());
                let mut staged = Staged {
                    block,
                    column: &mut column,
                };
                packs(inverse, &mut staged, threads, rest);
            },
        )
    }
}

/// A copied block's columns as its packs reach them.
trait Packed<T: Real> {
    fn columns(&self) -> usize;
    /// Copies columns `first..first + W` into the lanes of `rows`.
    fn fill<const W: usize>(&mut self, first: usize, rows: &mut [[T; W]]);
    /// Copies the lanes of `rows` back into columns `first..first + W`.
    fn drain<const W: usize>(&mut self, first: usize, rows: &[[T; W]]);
    /// Solves column `column` alone through `solve`.
    fn one(&mut self, column: usize, solve: impl FnOnce(&mut [T]));
}

/// A block whose columns are slices of its own, so packs fill from them and a column alone is solved where it is.
trait Slices<T: Real>: MatrixMut<T> {
    fn column_mut(&mut self, column: usize) -> &mut [T];
}

/// A block of [`Slices`] its packs reach where its columns are.
struct Sliced<'b, S>(&'b mut S);

impl<T: Real, S: Slices<T>> Packed<T> for Sliced<'_, S> {
    fn columns(&self) -> usize {
        self.0.columns()
    }

    fn fill<const W: usize>(&mut self, first: usize, rows: &mut [[T; W]]) {
        for lane in 0..W {
            rows.write_column(lane, self.0.column_mut(first + lane));
        }
    }

    fn drain<const W: usize>(&mut self, first: usize, rows: &[[T; W]]) {
        for lane in 0..W {
            rows.read_column(lane, self.0.column_mut(first + lane));
        }
    }

    fn one(&mut self, column: usize, solve: impl FnOnce(&mut [T])) {
        solve(self.0.column_mut(column));
    }
}

/// A block reached only through `read_column` and `write_column`, by way of `column`.
struct Staged<'b, T, M: ?Sized> {
    block: &'b mut M,
    column: &'b mut [T],
}

impl<T: Real, M: MatrixMut<T> + ?Sized> Packed<T> for Staged<'_, T, M> {
    fn columns(&self) -> usize {
        self.block.columns()
    }

    fn fill<const W: usize>(&mut self, first: usize, rows: &mut [[T; W]]) {
        for lane in 0..W {
            self.block.read_column(first + lane, self.column);
            rows.write_column(lane, self.column);
        }
    }

    fn drain<const W: usize>(&mut self, first: usize, rows: &[[T; W]]) {
        for lane in 0..W {
            rows.read_column(lane, self.column);
            self.block.write_column(first + lane, self.column);
        }
    }

    fn one(&mut self, column: usize, solve: impl FnOnce(&mut [T])) {
        self.block.read_column(column, self.column);
        solve(self.column);
        self.block.write_column(column, self.column);
    }
}

/// Solves every column of `block`, the widest packs first.
fn packs<T: Real, P: Packed<T>>(
    inverse: &ApproxInverse<T>,
    block: &mut P,
    threads: Threads,
    stack: &mut MemStack,
) {
    let columns = block.columns();
    let mut first = 0;
    while first < columns {
        first += match pack_lanes(columns - first) {
            PACK => whole_packs::<T, PACK, P>(inverse, block, first, threads, stack),
            HALF => whole_packs::<T, HALF, P>(inverse, block, first, threads, stack),
            _ => {
                block.one(first, |x| inverse.solve_unchecked(x, threads, stack));
                1
            }
        };
    }
}

/// Solves each whole pack of `W` columns of `block` from `first` on through rows carved once; the columns it solved.
fn whole_packs<T: Real, const W: usize, P: Packed<T>>(
    inverse: &ApproxInverse<T>,
    block: &mut P,
    first: usize,
    threads: Threads,
    stack: &mut MemStack,
) -> usize {
    let solved = (block.columns() - first) / W * W;
    let (mut rows, rest) = stack.make_with(inverse.n(), |_| [T::zero(); W]);
    for start in (first..first + solved).step_by(W) {
        block.fill(start, &mut rows);
        inverse.solve_unchecked(&mut rows, threads, rest);
        block.drain(start, &rows);
    }
    solved
}
