use std::array;

use approx_chol::{
    ApplyError, ApproxInverse, ColumnMajor, Columns, MatrixMut, MemBuffer, MemStack, Real,
    StackReq, Threads,
};

use super::apart::Apart;
use super::entry::cast;
use super::scratch::{least_slots, nan_cells};
use super::threads;

/// How a block stores its columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    /// Rows of every column, solved where they are.
    Interleaved,
    /// A block of the caller's own type, reached through `read_column` and `write_column`.
    Apart,
    ColumnMajor,
    Columns,
}

impl Layout {
    pub const ALL: [Layout; 4] = [
        Layout::Interleaved,
        Layout::Apart,
        Layout::ColumnMajor,
        Layout::Columns,
    ];
}

/// The same `K` columns in every layout, each owning its data.
#[derive(Clone)]
pub struct Layouts<T, const K: usize> {
    rows: usize,
    interleaved: Vec<[T; K]>,
    apart: Apart<T>,
    major: Vec<T>,
    slices: Vec<Vec<T>>,
}

impl<T: Real, const K: usize> Layouts<T, K> {
    pub fn new(columns: [Vec<T>; K]) -> Self {
        let rows = columns[0].len();
        Self {
            rows,
            interleaved: (0..rows)
                .map(|i| array::from_fn(|j| columns[j][i]))
                .collect(),
            apart: Apart(columns.to_vec()),
            major: columns.concat(),
            slices: columns.to_vec(),
        }
    }

    /// `f` on the block in `layout`.
    pub fn on<R>(&mut self, layout: Layout, f: impl FnOnce(&mut dyn MatrixMut<T>) -> R) -> R {
        match layout {
            Layout::Interleaved => f(&mut self.interleaved),
            Layout::Apart => f(&mut self.apart),
            Layout::ColumnMajor => {
                f(&mut ColumnMajor::new(&mut self.major, self.rows).expect("whole columns"))
            }
            Layout::Columns => f(&mut Columns::new(&mut self.slices).expect("equal columns")),
        }
    }

    /// The columns in `layout`, read through the trait.
    pub fn columns(&mut self, layout: Layout) -> Vec<Vec<T>> {
        let rows = self.rows;
        self.on(layout, |block| {
            let read = |j| {
                let mut column = vec![T::zero(); rows];
                block.read_column(j, &mut column);
                column
            };
            (0..block.columns()).map(read).collect()
        })
    }

    /// The fewest slots `layout` takes from `apply_with`'s stack, each probe on a fresh copy.
    pub fn least(&self, factor: &ApproxInverse<T>, layout: Layout) -> usize {
        least_slots::<T>(factor.scratch(K, threads::ONE), |stack| {
            let mut fresh = self.clone();
            fresh.on(layout, |block| {
                factor.apply_with(block, threads::ONE, stack)
            })
        })
    }
}

/// A block of the wrong row count, or a stack a byte short of what its layout takes at one thread, errs and leaves the block as it was, in every layout.
pub fn assert_errors_leave_the_block<T: Real>(factor: &ApproxInverse<T>) {
    let n = factor.n();
    let mut cells = nan_cells::<T>(factor.scratch(2, threads::ONE));
    for found in [n - 1, n + 1, n] {
        let column = |sign: f64| {
            (0..found)
                .map(|i| cast(sign * i as f64))
                .collect::<Vec<T>>()
        };
        let fresh = Layouts::new([column(1.0), column(-1.0)]);
        for layout in Layout::ALL {
            let mut block = fresh.clone();
            if found == n {
                let least = fresh.least(factor, layout);
                if least > 0 {
                    let bytes = least * size_of::<T>();
                    let mut buffer = MemBuffer::new(StackReq::new::<T>(least));
                    let short = MemStack::new(&mut buffer[..bytes - 1]);
                    let error = block.on(layout, |b| factor.apply_with(b, Threads::All, short));
                    assert_eq!(error, Err(ApplyError::Scratch), "{layout:?}");
                }
            } else {
                let rows = Err(ApplyError::Rows { expected: n, found });
                let error = block.on(layout, |b| factor.apply(b, Threads::All));
                assert_eq!(error, rows, "{layout:?}");
                let stack = MemStack::new_any(&mut cells);
                let error = block.on(layout, |b| factor.apply_with(b, Threads::All, stack));
                assert_eq!(error, rows, "{layout:?}");
            }
            assert!(
                block.columns(layout) == fresh.clone().columns(layout),
                "{layout:?} of {found} rows is left as it was"
            );
        }
    }
}
