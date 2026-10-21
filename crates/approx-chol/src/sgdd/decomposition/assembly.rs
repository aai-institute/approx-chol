//! `A` summed from `V`'s columns, row by row in column order.

use std::borrow::Cow;

use num_traits::PrimInt;

use super::partition::{Partition, UnionFind};
use super::{Assembled, Dominant, Partitioned, PrincipalBlocks};
use crate::input::{Column, Gram, IntoColumns, VertexPartition};
use crate::sgdd::block::ReduceError;
use crate::storage::{OffDiagonal, Rows, UpperAdjacency};
use crate::{diagonal_in_range, in_range, narrow, widen, Index, Real};

/// How a shape's summed cells are stored, and its components found from them.
pub(in crate::sgdd) trait AdjacencyLayout: Copy {
    type Adjacency<T: Real, I: Index>: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>
        + Sync;

    /// The summed cells and components, `seen` visiting each column once and `pair` each cell summed as `C` with its two ends.
    fn sum<T: Real, I: Index, C: CellSum>(
        self,
        v: impl IntoColumns<T>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, C),
    ) -> Result<Partitioned<Self::Adjacency<T, I>, I>, ReduceError>;
}

/// A general `V`'s cells: `A`'s strict upper triangle.
#[derive(Clone, Copy, Debug)]
pub(in crate::sgdd) struct StrictUpperTriangle;

/// A bipartite `V`'s cells: the biadjacency `C` from the left part's rows.
#[derive(Clone, Copy, Debug)]
pub(in crate::sgdd) struct Biadjacency {
    n_left: usize,
}

impl Biadjacency {
    /// `v`'s own left part, so the layout and the columns it sums never disagree.
    pub(in crate::sgdd) fn of<T: Real, J: PrimInt>(v: &Gram<'_, T, J>) -> Option<Self> {
        match v.partition() {
            VertexPartition::Bipartite { n_left } => Some(Self { n_left }),
            VertexPartition::Unpartitioned => None,
        }
    }
}

impl AdjacencyLayout for StrictUpperTriangle {
    type Adjacency<T: Real, I: Index> = UpperAdjacency<T, I>;

    // Out of line, the Laplacian entry measured 3–4% slower: the same work, laid out worse.
    #[inline(always)]
    fn sum<T: Real, I: Index, C: CellSum>(
        self,
        v: impl IntoColumns<T>,
        seen: impl FnMut(&Column<T, I>),
        mut pair: impl FnMut(usize, usize, C),
    ) -> Result<Partitioned<UpperAdjacency<T, I>, I>, ReduceError> {
        let counts = counts::<T, I>(v, v.n(), seen);
        let mut sets = UnionFind::<I>::new(v.n());
        let rows = sum_cells::<T, I, C>(v, &counts, 0, |i, j, cell| {
            if cell.value() != 0.0 {
                sets.union(i, j);
            }
            pair(i, j, cell);
        })?;
        Ok(Partitioned {
            a: UpperAdjacency::new(rows),
            partition: sets.into_partition(),
        })
    }
}

impl AdjacencyLayout for Biadjacency {
    type Adjacency<T: Real, I: Index> = Rows<T, I>;

    /// `C`, summed in a dense table when small (+54% without it).
    fn sum<T: Real, I: Index, C: CellSum>(
        self,
        v: impl IntoColumns<T>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, C),
    ) -> Result<Partitioned<Rows<T, I>, I>, ReduceError> {
        let n_left = self.n_left;
        let table = n_left.saturating_mul(v.n() - n_left);
        let cell = std::mem::size_of::<C>();
        let a = match table.saturating_mul(cell) <= DENSE_TABLE_MAX_BYTES {
            true => dense_cross(v, n_left, seen, pair)?,
            false => {
                let counts = counts::<T, I>(v, n_left, seen);
                let columns: usize = counts.iter().sum();
                // Past the cap, the table still wins whenever the buckets would take more memory.
                match columns.saturating_mul(12) < table.saturating_mul(cell) {
                    true => sum_cells::<T, I, C>(v, &counts, n_left, pair)?,
                    false => dense_cross(v, n_left, |_| {}, pair)?,
                }
            }
        };
        let partition = Partition::of_bipartite(&a);
        Ok(Partitioned { a, partition })
    }
}

/// An incidence `V`'s summed cells beside `σ`, and their components.
type IncidenceCells<'a, T, I, S> =
    Partitioned<Dominant<'a, T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// A valued `V`'s summed cells beside `A`'s diagonal, and their components.
type ValueCells<T, I, S> = Partitioned<Assembled<T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// D: each pivot `Σ_j |c_ij| + σ_i`, plus its pairs' cancelled mass where cells cancel; `A`'s diagonal never stored.
pub(in crate::sgdd) fn assemble_incidence<'a, T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: Gram<'a, T, J>,
) -> Result<IncidenceCells<'a, T, I, S>, ReduceError> {
    let mut abs = vec![0.0_f64; v.n()];
    let ground = v.diagonal();
    if !v.cancels() {
        let a = cells.sum(
            v,
            |_| {},
            |i, j, c: f64| {
                (abs[i], abs[j]) = (abs[i] + c.abs(), abs[j] + c.abs());
            },
        )?;
        return dominant(a, &abs, ground.map(Cow::Borrowed));
    }
    let mut surplus = sigma_sums(ground, v.n());
    let a = cells.sum(
        v,
        |_| {},
        |i, j, masses: SignedMasses| {
            let (c, s) = (masses.value().abs(), masses.surplus());
            (abs[i], abs[j]) = (abs[i] + c, abs[j] + c);
            (surplus[i], surplus[j]) = (surplus[i] + s, surplus[j] + s);
        },
    )?;
    let surplus = surplus.into_iter().map(narrow).collect();
    dominant(a, &abs, Some(Cow::Owned(surplus)))
}

/// Each pivot `Σ_j |c_ij| + σ_i` screened in the assembly's own sums, `A`'s diagonal never stored.
fn dominant<'a, T: Real, I: Index, C>(
    a: Partitioned<C, I>,
    abs: &[f64],
    surplus: Option<Cow<'a, [T]>>,
) -> Result<Partitioned<Dominant<'a, T, C>, I>, ReduceError> {
    let s = |i: usize| surplus.as_ref().map_or(0.0, |s| widen(s[i]));
    if let Some(vertex) = (0..abs.len()).find(|&i| !diagonal_in_range(narrow::<T>(abs[i] + s(i)))) {
        return Err(ReduceError::OutOfRange { vertex });
    }
    let Partitioned { a, partition } = a;
    // An all-zero surplus grounds nothing: dropped once here, not copied into every block.
    let surplus = surplus.filter(|s| s.iter().any(|&x| x != T::zero()));
    let a = Dominant {
        adjacency: a,
        surplus,
    };
    Ok(Partitioned { a, partition })
}

/// `A`'s cells, its diagonal `σ_i + Σ_k w_k v_ik²` and its components, summed in one column pass.
pub(in crate::sgdd) fn assemble_values<T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: Gram<'_, T, J>,
) -> Result<ValueCells<T, I, S>, ReduceError> {
    let mut sums = sigma_sums(v.diagonal(), v.n());
    let a = cells.sum::<T, I, f64>(v, |column| add_squares(&mut sums, column), |_, _, _| {})?;
    assembled(a, sums)
}

/// Each vertex's `σ_i` in `f64`, where its column sums start.
fn sigma_sums<T: Real>(sigma: Option<&[T]>, n: usize) -> Vec<f64> {
    match sigma {
        Some(s) => s.iter().map(|&s| widen(s)).collect(),
        None => vec![0.0; n],
    }
}

/// Adds `w v²` at both ends, so a row of only underflowed terms fails the range screen.
#[inline(always)]
fn add_squares<T: Real, I: Index>(sums: &mut [f64], column: &Column<T, I>) {
    let square = |v: f64| column.weight.bilinear_unflushed(v, v);
    // Two straight-line ends: a loop over them costs signed_F 7–10% and bip_sparse 18–20%.
    sums[column.a.index()] += square(widen(column.v_a));
    sums[column.b.index()] += square(widen(column.v_b));
}

/// The diagonal is each sum as `T` stores it; a nonzero one cast to zero underflowed, so it is out of range.
fn assembled<T: Real, I, C>(
    Partitioned { a, partition }: Partitioned<C, I>,
    sums: Vec<f64>,
) -> Result<Partitioned<Assembled<T, C>, I>, ReduceError> {
    let diagonal = (sums.into_iter().enumerate())
        .map(|(i, sum)| match narrow::<T>(sum) {
            d if sum != 0.0 && d == T::zero() => Err(ReduceError::OutOfRange { vertex: i }),
            d => Ok(d),
        })
        .collect::<Result<_, _>>()?;
    let a = Assembled {
        adjacency: a,
        diagonal,
    };
    Ok(Partitioned { a, partition })
}

/// Row `i`'s nonzero sum as `T` stores it; one cast to zero underflowed, so it is out of range too.
#[inline(always)]
fn admit<T: Real>(i: usize, x: f64) -> Result<T, ReduceError> {
    T::from(x)
        .filter(|&value| value != T::zero() && in_range(value))
        .ok_or(ReduceError::OutOfRange { vertex: i })
}

/// within's cap on the dense accumulator.
const DENSE_TABLE_MAX_BYTES: usize = 40_000_000;

/// `pair` sees each cell not [`CellSum::is_empty`].
fn dense_cross<T: Real, I: Index, C: CellSum>(
    v: impl IntoColumns<T>,
    n_left: usize,
    mut seen: impl FnMut(&Column<T, I>),
    mut pair: impl FnMut(usize, usize, C),
) -> Result<Rows<T, I>, ReduceError> {
    let n_right = v.n() - n_left;
    let mut table = vec![C::default(); n_left * n_right];
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            table[column.a.index() * n_right + column.b.index() - n_left].add(column.cell());
            seen(&column);
        },
    );
    let mut ptrs = Vec::with_capacity(n_left + 1);
    ptrs.push(0);
    let (mut indices, mut cells) = (Vec::new(), Vec::new());
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        for (j, &x) in row.iter().enumerate().filter(|&(_, &x)| !x.is_empty()) {
            pair(i, n_left + j, x);
            if x.value() != 0.0 {
                indices.push(I::from_index(j));
                cells.push(admit::<T>(i, x.value())?);
            }
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
pub(in crate::sgdd) trait CellSum: Copy + Default {
    fn add(&mut self, x: f64);
    fn value(self) -> f64;
    /// Nothing left to report: a plain sum that cancelled is empty too, as only kept masses outlive it.
    fn is_empty(self) -> bool;
}

impl CellSum for f64 {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        *self += x;
    }

    fn value(self) -> f64 {
        self
    }

    fn is_empty(self) -> bool {
        self == 0.0
    }
}

/// A pair's positive and negative weight apart: cell `p − q`, each end's surplus `2·min(p, q)`.
#[derive(Clone, Copy, Default)]
struct SignedMasses {
    positive: f64,
    negative: f64,
}

impl CellSum for SignedMasses {
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

    fn is_empty(self) -> bool {
        self.positive == 0.0 && self.negative == 0.0
    }
}

impl SignedMasses {
    fn surplus(self) -> f64 {
        2.0 * self.positive.min(self.negative)
    }
}

/// AdjacencyLayout bucketed at their lower end in column order, so each sums as a dense table would; `pair` sees each summed cell, a cancelled one too.
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
    let values = cells.into_iter().map(narrow).collect();
    Ok(Rows::new(ptrs, columns, values, n_cols))
}
