//! `A` summed from `V`'s columns, row by row in column order.

use num_traits::PrimInt;

use super::partition::{Partition, UnionFind};
use super::{indexable, Assembled, BipartiteGram, General, Partitioned};
use crate::input::{BipartiteColumns, Column, IntoColumns, SignedGraph};
use crate::sgdd::block::ReduceError;
use crate::storage::{Rows, UpperAdjacency};
use crate::{in_range, narrow, widen, Index, Real};

/// The signed graph's cells, each vertex's surplus its pairs' cancelled mass.
pub(in crate::sgdd) fn assemble_signed<T: Real, J: PrimInt, I: Index>(
    graph: SignedGraph<'_, T, J>,
) -> Result<(General<T, I>, Vec<T>), ReduceError> {
    let mut surplus = vec![0.0_f64; graph.n()];
    let a = assemble(graph, |i, j, masses: Masses| {
        let s = masses.surplus();
        surplus[i] += s;
        surplus[j] += s;
    })?;
    Ok((a, surplus.into_iter().map(narrow).collect()))
}

/// `A`'s strict upper triangle, diagonal and components, cells summed by pair in column order.
pub(in crate::sgdd) fn assemble_general<T: Real, I: Index>(
    v: impl IntoColumns<T>,
) -> Result<General<T, I>, ReduceError> {
    assemble::<T, I, f64>(v, |_, _, _| {})
}

/// As [`assemble_general`], each cell summed as `C` and seen by `pair` with its two ends.
// Out of line, the Laplacian entry measured 3–4% slower: the same work, laid out worse.
#[inline(always)]
fn assemble<T: Real, I: Index, C: CellSum>(
    v: impl IntoColumns<T>,
    mut pair: impl FnMut(usize, usize, C),
) -> Result<General<T, I>, ReduceError> {
    indexable::<I>(v.n())?;
    let mut diagonal = vec![0.0_f64; v.n()];
    let counts = counts::<T, I>(v, v.n(), |column| add_squares(&mut diagonal, column));
    let mut sets = UnionFind::<I>::new(v.n());
    let rows = sum_cells::<T, I, C>(v, &counts, 0, |i, j, cell| {
        if cell.value() != 0.0 {
            sets.union(i, j);
        }
        pair(i, j, cell);
    })?;
    let a = Assembled {
        adjacency: UpperAdjacency::new(rows),
        diagonal: diagonal_of(diagonal)?,
    };
    Ok(Partitioned {
        a,
        partition: sets.into_partition(),
    })
}

/// `C` and its components.
pub(in crate::sgdd) fn assemble_bipartite<T: Real, I: Index>(
    v: impl BipartiteColumns<T>,
) -> Result<Partitioned<Rows<T, I>, I>, ReduceError> {
    cross(v, |_| {})
}

/// As [`assemble_bipartite`], with `A`'s diagonal summed in the same column pass.
pub(in crate::sgdd) fn assemble_bipartite_gram<T: Real, I: Index>(
    v: impl BipartiteColumns<T>,
) -> Result<BipartiteGram<T, I>, ReduceError> {
    let mut sums = vec![0.0_f64; v.n()];
    let Partitioned { a, partition } = cross(v, |column| add_squares(&mut sums, column))?;
    let a = Assembled {
        adjacency: a,
        diagonal: diagonal_of(sums)?,
    };
    Ok(Partitioned { a, partition })
}

/// Adds `w v²` at both ends, `NaN` where a nonzero term underflowed in `f64`, so the range screen names its vertex.
#[inline(always)]
fn add_squares<T: Real, I: Index>(sums: &mut [f64], column: &Column<T, I>) {
    let square = |v: f64| match column.weight.bilinear(v, v) {
        0.0 if v != 0.0 && !column.weight.is_zero() => f64::NAN,
        term => term,
    };
    // Two straight-line ends: a loop over them costs signed_F 7–10% and bip_sparse 18–20%.
    sums[column.a.index()] += square(widen(column.v_a));
    sums[column.b.index()] += square(widen(column.v_b));
}

/// Each sum as `T` stores it; a nonzero one cast to zero underflowed, so it is out of range.
fn diagonal_of<T: Real>(sums: Vec<f64>) -> Result<Vec<T>, ReduceError> {
    (sums.into_iter().enumerate())
        .map(|(i, sum)| match narrow::<T>(sum) {
            d if sum != 0.0 && d == T::zero() => Err(ReduceError::OutOfRange { vertex: i }),
            d => Ok(d),
        })
        .collect()
}

/// Row `i`'s nonzero sum as `T` stores it; one cast to zero underflowed, so it is out of range too.
#[inline(always)]
fn admit<T: Real>(i: usize, x: f64) -> Result<T, ReduceError> {
    T::from(x)
        .filter(|&value| value != T::zero() && in_range(value))
        .ok_or(ReduceError::OutOfRange { vertex: i })
}

/// `C`'s summed cells and components, summed in a dense table when small (+54% without it), `seen` visiting each column once.
fn cross<T: Real, I: Index>(
    v: impl BipartiteColumns<T>,
    seen: impl FnMut(&Column<T, I>),
) -> Result<Partitioned<Rows<T, I>, I>, ReduceError> {
    indexable::<I>(v.n())?;
    let n_left = v.n_left();
    let table = n_left.saturating_mul(v.n() - n_left);
    let a = match table <= DENSE_TABLE_MAX_CELLS {
        true => dense_cross(v, seen)?,
        false => {
            let counts = counts::<T, I>(v, n_left, seen);
            let columns: usize = counts.iter().sum();
            // Past the cap, the table still wins whenever the buckets would take more memory.
            match columns.saturating_mul(12) < table.saturating_mul(8) {
                true => sum_cells::<T, I, f64>(v, &counts, n_left, |_, _, _| {})?,
                false => dense_cross(v, |_| {})?,
            }
        }
    };
    let partition = Partition::of_bipartite(&a);
    Ok(Partitioned { a, partition })
}

/// within's cap on the dense accumulator, about 40 MB.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

fn dense_cross<T: Real, I: Index>(
    v: impl BipartiteColumns<T>,
    mut seen: impl FnMut(&Column<T, I>),
) -> Result<Rows<T, I>, ReduceError> {
    let (n_left, n_right) = (v.n_left(), v.n() - v.n_left());
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
            indices.push(I::from_index(j));
            cells.push(admit::<T>(i, x)?);
        }
        ptrs.push(indices.len());
    }
    drop(table);
    Ok(Rows::new(ptrs, indices, cells, n_right))
}

/// Each row's cell count, a column's row its lower end, `seen` visiting each column once.
fn counts<T: Real, I: Index>(
    v: impl IntoColumns<T>,
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

/// What one cell's terms sum to: their value, or a signed graph's two masses kept apart.
trait CellSum: Copy + Default {
    fn add(&mut self, x: f64);
    fn value(self) -> f64;
}

impl CellSum for f64 {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        *self += x;
    }

    fn value(self) -> f64 {
        self
    }
}

/// A pair's positive and negative weight apart: cell `p − q`, each end's surplus `2·min(p, q)`.
#[derive(Clone, Copy, Default)]
struct Masses {
    positive: f64,
    negative: f64,
}

impl CellSum for Masses {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        match x >= 0.0 {
            true => self.positive += x,
            false => self.negative -= x,
        }
    }

    fn value(self) -> f64 {
        self.positive - self.negative
    }
}

impl Masses {
    fn surplus(self) -> f64 {
        2.0 * self.positive.min(self.negative)
    }
}

/// Cells bucketed at their lower end in column order, so each sums as a dense table would; `pair` sees each summed cell, a cancelled one too.
fn sum_cells<T: Real, I: Index, C: CellSum>(
    v: impl IntoColumns<T>,
    counts: &[usize],
    first_column: usize,
    mut pair: impl FnMut(usize, usize, C),
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
    let mut sums = vec![C::default(); n_cols];
    let mut touched: Vec<I> = Vec::new();
    let mut ptrs = Vec::with_capacity(starts.len());
    ptrs.push(0);
    let mut out = 0;
    for (i, span) in starts.windows(2).enumerate() {
        for p in span[0]..span[1] {
            let j = columns[p];
            // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
            if sums[j.index()].value() == 0.0 {
                touched.push(j);
            }
            sums[j.index()].add(cells[p]);
        }
        touched.sort_unstable();
        touched.dedup();
        // Written in place: a row's sums never outnumber its own bucket, already read.
        for &j in &touched {
            let cell = std::mem::take(&mut sums[j.index()]);
            pair(i, first_column + j.index(), cell);
            if cell.value() != 0.0 {
                columns[out] = j;
                cells[out] = widen(admit::<T>(i, cell.value())?);
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
    Ok(Rows::new(ptrs, columns, cells, n_cols).map(narrow))
}
