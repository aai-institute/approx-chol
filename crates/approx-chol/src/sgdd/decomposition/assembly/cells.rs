//! `A`'s cells summed from `V`'s columns, row by row in column order.

use num_traits::PrimInt;

use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::ReduceError;
use crate::storage::Rows;
use crate::{in_range, narrow, widen, Index, Real};

/// `A`'s strict upper triangle; `pair` sees each summed cell, a cancelled one too.
pub(super) fn upper<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    seen: impl FnMut(&Column<T, I>),
    pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let counts = counts::<T, J, I>(v, v.n(), seen);
    sum_cells::<T, J, I>(v, &counts, 0, pair)
}

/// The biadjacency `C` from the left part's rows, summed in a dense table when small (+54% without it).
pub(super) fn cross<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    n_left: usize,
    seen: impl FnMut(&Column<T, I>),
    pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let table = n_left.saturating_mul(v.n() - n_left);
    // `?` per arm: returning the arms' results directly reshuffled the whole build's inlining.
    let a = match table <= DENSE_TABLE_MAX_CELLS {
        true => dense_cross(v, n_left, seen, pair)?,
        false => {
            let counts = counts::<T, J, I>(v, n_left, seen);
            let columns: usize = counts.iter().sum();
            // Past the cap, the table still wins whenever the buckets would take more memory.
            match columns.saturating_mul(12) < table.saturating_mul(8) {
                true => sum_cells::<T, J, I>(v, &counts, n_left, pair)?,
                false => dense_cross(v, n_left, |_| {}, pair)?,
            }
        }
    };
    Ok(a)
}

/// Row `i`'s nonzero sum as `T` stores it; one cast to zero underflowed, so it is out of range too.
#[inline(always)]
fn admit<T: Real>(i: usize, x: f64) -> Result<T, ReduceError> {
    T::from(x)
        .filter(|&value| value != T::zero() && in_range(value))
        .ok_or(ReduceError::OutOfRange { vertex: i })
}

/// within's cap on the dense accumulator, about 40 MB.
pub(super) const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

/// `pair` sees each nonzero cell.
pub(super) fn dense_cross<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    n_left: usize,
    mut seen: impl FnMut(&Column<T, I>),
    mut pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let n_right = v.n() - n_left;
    let mut table = vec![0.0_f64; n_left * n_right];
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            table[column.a.index() * n_right + column.b.index() - n_left] += column.cell();
            seen(&column);
        },
    );
    let mut ptrs = Vec::with_capacity(n_left + 1);
    ptrs.push(0);
    let (mut indices, mut cells) = (Vec::new(), Vec::new());
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        for (j, &x) in row.iter().enumerate().filter(|&(_, &x)| x != 0.0) {
            pair(i, n_left + j, x);
            indices.push(I::from_index(j));
            cells.push(admit::<T>(i, x)?);
        }
        ptrs.push(indices.len());
    }
    drop(table);
    Ok(Rows::new(ptrs, indices, cells, n_right))
}

/// Each row's cell count, a column's row its lower end, `seen` visiting each column once.
pub(super) fn counts<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    n_rows: usize,
    mut seen: impl FnMut(&Column<T, I>),
) -> Vec<usize> {
    let mut counts = vec![0; n_rows];
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            counts[column.a.min(column.b).index()] += 1;
            seen(&column);
        },
    );
    counts
}

/// Cells bucketed at their lower end in column order, so each sums as a dense table would; `pair` sees each summed cell, a cancelled one too.
pub(super) fn sum_cells<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    counts: &[usize],
    first_column: usize,
    mut pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let starts: Vec<usize> = std::iter::once(0)
        .chain(counts.iter().scan(0, |end, &count| {
            *end += count;
            Some(*end)
        }))
        .collect();
    let entries = starts[counts.len()];
    let mut cursor = starts[..counts.len()].to_vec();
    let (mut columns, mut cells) = (vec![I::zero(); entries], vec![0.0; entries]);
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            let (lo, hi) = (column.a.min(column.b), column.a.max(column.b));
            let slot = &mut cursor[lo.index()];
            columns[*slot] = hi - I::from_index(first_column);
            cells[*slot] = column.cell();
            *slot += 1;
        },
    );
    drop(cursor);
    let n_cols = v.n() - first_column;
    let mut sums = vec![0.0_f64; n_cols];
    let mut touched: Vec<I> = Vec::new();
    let mut ptrs = Vec::with_capacity(starts.len());
    ptrs.push(0);
    let mut out = 0;
    for (i, span) in starts.windows(2).enumerate() {
        for p in span[0]..span[1] {
            let j = columns[p];
            // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
            if sums[j.index()] == 0.0 {
                touched.push(j);
            }
            sums[j.index()] += cells[p];
        }
        touched.sort_unstable();
        touched.dedup();
        // Written in place: a row's sums never outnumber its own bucket, already read.
        for &j in &touched {
            let cell = std::mem::take(&mut sums[j.index()]);
            pair(i, first_column + j.index(), cell);
            if cell != 0.0 {
                columns[out] = j;
                cells[out] = widen(admit::<T>(i, cell)?);
                out += 1;
            }
        }
        touched.clear();
        ptrs.push(out);
    }
    // Freed before the shrinks reallocate, as they set the reduction's peak.
    drop((starts, sums));
    columns.truncate(out);
    columns.shrink_to_fit();
    cells.truncate(out);
    cells.shrink_to_fit();
    let values = cells.into_iter().map(narrow).collect();
    Ok(Rows::new(ptrs, columns, values, n_cols))
}
