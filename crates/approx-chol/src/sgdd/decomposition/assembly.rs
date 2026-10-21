//! `A` summed from `V`'s columns, row by row in column order.

use num_traits::PrimInt;

use super::partition::{Partition, UnionFind};
use super::{Cells, LonePivot, Partitioned, PrincipalBlocks};
use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::{Grounding, ReduceError};
use crate::sgdd::diagonal_in_range;
use crate::storage::{OffDiagonal, Rows, UpperAdjacency};
use crate::{in_range, narrow, widen, Index, Real};

/// How a shape's summed cells are stored, and its components found from them.
pub(in crate::sgdd) trait AdjacencyLayout {
    type Adjacency<T: Real, I: Index>: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>
        + Sync;

    /// The summed cells and components, `seen` visiting each column once and `pair` each nonzero cell with its two ends.
    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, f64),
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
    pub(in crate::sgdd) fn of<T: Real, J: PrimInt>(v: &FactorWidth2<'_, T, J>) -> Option<Self> {
        v.n_left().map(|n_left| Self { n_left })
    }
}

impl AdjacencyLayout for StrictUpperTriangle {
    type Adjacency<T: Real, I: Index> = UpperAdjacency<T, I>;

    // Out of line, the Laplacian entry measured 3–4% slower: the same work, laid out worse.
    #[inline(always)]
    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        mut pair: impl FnMut(usize, usize, f64),
    ) -> Result<Partitioned<UpperAdjacency<T, I>, I>, ReduceError> {
        let counts = counts::<T, J, I>(v, v.n(), seen);
        let mut sets = UnionFind::<I>::new(v.n());
        let rows = sum_cells::<T, J, I>(v, &counts, 0, |i, j, cell| {
            if cell != 0.0 {
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
    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, f64),
    ) -> Result<Partitioned<Rows<T, I>, I>, ReduceError> {
        let n_left = self.n_left;
        let table = n_left.saturating_mul(v.n() - n_left);
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
        let partition = Partition::of_bipartite(&a);
        Ok(Partitioned { a, partition })
    }
}

/// A factor-width-2 `A`, assembled: general, or bipartite as `C`.
pub(in crate::sgdd) struct Assembled<T, C> {
    adjacency: C,
    diagonal: Vec<T>,
}

/// `A = L(C) + diag(σ)`, dominant by shape: no diagonal is stored, each pivot is `Σ_j |c_ij| + σ_i`.
pub(in crate::sgdd) struct Dominant<'a, T, C> {
    adjacency: C,
    surplus: Option<&'a [T]>,
}

/// An incidence `V`'s summed cells beside `σ`, and their components.
type IncidenceCells<'a, T, I, S> =
    Partitioned<Dominant<'a, T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// A valued `V`'s summed cells beside `A`'s diagonal, and their components.
type ValueCells<T, I, S> = Partitioned<Assembled<T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// Each pivot `Σ_j |c_ij| + σ_i`, screened in the assembly's own sums; `A`'s diagonal never stored.
pub(in crate::sgdd) fn assemble_incidence<'a, T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: FactorWidth2<'a, T, J>,
) -> Result<IncidenceCells<'a, T, I, S>, ReduceError> {
    let mut abs = vec![0.0_f64; v.n()];
    let a = cells.sum(
        v,
        |_| {},
        |i, j, c: f64| {
            (abs[i], abs[j]) = (abs[i] + c.abs(), abs[j] + c.abs());
        },
    )?;
    let surplus = v.surplus();
    let s = |i: usize| surplus.map_or(0.0, |s| widen(s[i]));
    if let Some(vertex) = (0..abs.len()).find(|&i| !diagonal_in_range(narrow::<T>(abs[i] + s(i)))) {
        return Err(ReduceError::OutOfRange { vertex });
    }
    let Partitioned { a, partition } = a;
    let a = Dominant {
        adjacency: a,
        surplus,
    };
    Ok(Partitioned { a, partition })
}

/// `A`'s cells, its diagonal `Σ_k w_k v_ik²` and its components, summed in one column pass.
pub(in crate::sgdd) fn assemble_values<T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: FactorWidth2<'_, T, J>,
) -> Result<ValueCells<T, I, S>, ReduceError> {
    let mut sums = vec![0.0; v.n()];
    let a = cells.sum::<T, J, I>(v, |column| add_squares(&mut sums, column), |_, _, _| {})?;
    assembled(a, sums)
}

/// Adds `w v²` at both ends, so a row of only underflowed terms fails the range screen.
#[inline(always)]
fn add_squares<T: Real, I: Index>(sums: &mut [f64], column: &Column<T, I>) {
    // Two straight-line ends: a loop over them costs signed_F 7–10% and bip_sparse 18–20%.
    sums[column.a.index()] += column.square(widen(column.v_a));
    sums[column.b.index()] += column.square(widen(column.v_b));
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

/// within's cap on the dense accumulator, about 40 MB.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

/// `pair` sees each nonzero cell.
fn dense_cross<T: Real, J: PrimInt, I: Index>(
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
fn counts<T: Real, J: PrimInt, I: Index>(
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
fn sum_cells<T: Real, J: PrimInt, I: Index>(
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

impl<T: Real, C> LonePivot<T> for Assembled<T, C> {
    fn lone(&self, v: usize) -> T {
        self.diagonal[v]
    }
}

impl<T: Real, C> LonePivot<T> for Dominant<'_, T, C> {
    fn lone(&self, v: usize) -> T {
        self.surplus.map_or(T::zero(), |s| s[v])
    }
}

/// The upper triangle, cancelled cells dropped.
impl<T: Real, I: Index> PrincipalBlocks<T, I> for UpperAdjacency<T, I> {
    type Block = (Box<[I]>, OffDiagonal<T, I>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        (vertices.into(), OffDiagonal::General(self))
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|&v| self.upper_len(v.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &v in vertices {
            for (j, &x) in self.upper(v.index()) {
                indices.push(position[j]);
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        (
            vertices.into(),
            OffDiagonal::General(UpperAdjacency::new(rows)),
        )
    }
}

impl<T: Real, I: Index, C> PrincipalBlocks<T, I> for Dominant<'_, T, C>
where
    C: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>,
{
    type Block = (Box<[I]>, OffDiagonal<T, I>, Grounding<T>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let grounding = surplus_at(self.surplus, &vertices);
        (vertices, off_diagonal, grounding)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let grounding = surplus_at(self.surplus, &vertices);
        (vertices, off_diagonal, grounding)
    }
}

fn surplus_at<T: Real, I: Index>(surplus: Option<&[T]>, vertices: &[I]) -> Grounding<T> {
    surplus.map_or_else(Grounding::ungrounded, |s| Grounding::of(at(s, vertices)))
}

/// `values` in `vertices`' order.
fn at<T: Copy, I: Index>(values: &[T], vertices: &[I]) -> Vec<T> {
    vertices.iter().map(|&v| values[v.index()]).collect()
}

/// The biadjacency `C`, cancelled cells dropped: rows are `A`'s left vertices.
impl<T: Real, I: Index> PrincipalBlocks<T, I> for Rows<T, I> {
    type Block = (Box<[I]>, OffDiagonal<T, I>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (lefts, rights) = vertices.split_at(self.n_rows());
        sides(lefts, rights, self)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let n_left = self.n_rows();
        let (lefts, rights) = vertices.split_at(vertices.partition_point(|&v| v.index() < n_left));
        let mut ptrs = Vec::with_capacity(lefts.len() + 1);
        ptrs.push(0);
        let entries = lefts.iter().map(|&i| self.row_len(i.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &i in lefts {
            for (j, &x) in self.row(i.index()) {
                indices.push(position[n_left + j] - I::from_index(lefts.len()));
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        sides(
            lefts,
            rights,
            Rows::new(ptrs, indices, values, rights.len()),
        )
    }
}

impl<T: Real, I: Index, C> PrincipalBlocks<T, I> for Assembled<T, C>
where
    C: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>,
{
    type Block = Cells<T, I>;

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let diagonal = at(&self.diagonal, &vertices);
        (vertices, off_diagonal, diagonal)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let diagonal = at(&self.diagonal, &vertices);
        (vertices, off_diagonal, diagonal)
    }
}

/// The larger side eliminated first, `C` transposed when it is the right one; a side alone is one uncoupled vertex.
fn sides<T: Real, I: Index>(
    lefts: &[I],
    rights: &[I],
    c: Rows<T, I>,
) -> (Box<[I]>, OffDiagonal<T, I>) {
    if lefts.is_empty() || rights.is_empty() {
        debug_assert_eq!(
            lefts.len() + rights.len(),
            1,
            "a connected side alone is one vertex"
        );
        let empty = Rows::new(vec![0, 0], Vec::new(), Vec::new(), 1);
        let vertices = [lefts, rights].concat().into();
        return (vertices, OffDiagonal::General(UpperAdjacency::new(empty)));
    }
    match lefts.len() >= rights.len() {
        true => ([lefts, rights].concat().into(), OffDiagonal::Bipartite(c)),
        false => {
            let c = c.transpose();
            ([rights, lefts].concat().into(), OffDiagonal::Bipartite(c))
        }
    }
}
