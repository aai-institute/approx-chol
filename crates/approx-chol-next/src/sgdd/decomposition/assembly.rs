//! `A` summed from `V`'s columns, row by row in column order, and cut into its components' blocks.

use num_traits::PrimInt;

use super::partition::{Partition, UnionFind};
use super::{Cells, Partitioned, PrincipalBlocks};
use crate::input::{BipartiteColumns, Column, IntoColumns, SignedGraph};
use crate::sgdd::component::ReduceError;
use crate::storage::{CrossBlock, Rows, UpperAdjacency};
use crate::{in_range, narrow, widen, Real};

/// The signed graph's cells, each vertex's surplus its pairs' cancelled mass.
pub(in crate::sgdd) fn assemble_signed<T: Real, I: PrimInt>(
    graph: SignedGraph<'_, T, I>,
) -> Result<(Partitioned<Assembled<T>>, Vec<T>), ReduceError> {
    let mut surplus = vec![0.0_f64; graph.n()];
    let a = assemble(graph, |i, j, masses: Masses| {
        let s = masses.surplus();
        surplus[i] += s;
        surplus[j] += s;
    })?;
    let surplus: Vec<T> = surplus.into_iter().map(narrow).collect();
    if let Some(vertex) = surplus.iter().position(|&s| !in_range(s)) {
        let vertex = vertex as u32;
        return Err(ReduceError::OutOfRange { vertex });
    }
    Ok((a, surplus))
}

/// `A`'s strict upper triangle, diagonal and components, cells summed by pair in column order.
pub(in crate::sgdd) fn assemble_general<T: Real>(
    v: impl IntoColumns<T>,
) -> Result<Partitioned<Assembled<T>>, ReduceError> {
    assemble::<T, f64>(v, |_, _, _| {})
}

/// As [`assemble_general`], each cell summed as `C` and seen by `pair` with its two ends.
// Out of line, the Laplacian entry measured 3–4% slower: the same work, laid out worse.
#[inline(always)]
fn assemble<T: Real, C: CellSum>(
    v: impl IntoColumns<T>,
    pair: impl FnMut(usize, usize, C),
) -> Result<Partitioned<Assembled<T>>, ReduceError> {
    let n = v.n();
    let (mut counts, mut diagonal) = (vec![0; n], vec![0.0_f64; n]);
    v.for_each_column(
        #[inline(always)]
        |column| {
            counts[column.a.min(column.b) as usize] += 1;
            let (v_a, v_b) = (widen(column.v_a), widen(column.v_b));
            diagonal[column.a as usize] += column.weight.bilinear(v_a, v_a);
            diagonal[column.b as usize] += column.weight.bilinear(v_b, v_b);
        },
    );
    let mut buckets = Buckets::new(&counts);
    v.for_each_column(
        #[inline(always)]
        |column| {
            let (lo, hi) = (column.a.min(column.b), column.a.max(column.b));
            buckets.push(lo as usize, hi, column.cell());
        },
    );
    let (rows, partition) = buckets.sum::<T, C>(n, 0, pair)?;
    let a = Assembled {
        adjacency: UpperAdjacency::new(rows),
        diagonal: diagonal_of(diagonal)?,
    };
    Ok(Partitioned { a, partition })
}

/// `C` and its components.
pub(in crate::sgdd) fn assemble_bipartite<T: Real>(
    v: impl BipartiteColumns<T>,
) -> Result<Partitioned<Biadjacency<T>>, ReduceError> {
    biadjacency(v, |_| {})
}

/// As [`assemble_bipartite`], with `A`'s diagonal summed in the same column pass.
pub(in crate::sgdd) fn assemble_bipartite_gram<T: Real>(
    v: impl BipartiteColumns<T>,
) -> Result<(Partitioned<Biadjacency<T>>, Vec<T>), ReduceError> {
    let mut sums = vec![0.0_f64; v.n()];
    let a = biadjacency(
        v,
        #[inline(always)]
        |column: &Column<T>| {
            let (v_a, v_b) = (widen(column.v_a), widen(column.v_b));
            sums[column.a as usize] += column.weight.bilinear(v_a, v_a);
            sums[column.b as usize] += column.weight.bilinear(v_b, v_b);
        },
    )?;
    Ok((a, diagonal_of(sums)?))
}

/// Each sum as `T` stores it; a nonzero one cast to zero underflowed, so it is out of range.
fn diagonal_of<T: Real>(sums: Vec<f64>) -> Result<Vec<T>, ReduceError> {
    (sums.into_iter().enumerate())
        .map(|(i, sum)| match narrow::<T>(sum) {
            d if sum != 0.0 && d == T::zero() => Err(ReduceError::OutOfRange { vertex: i as u32 }),
            d => Ok(d),
        })
        .collect()
}

/// `C` and its components, `seen` visiting each column once.
fn biadjacency<T: Real>(
    v: impl BipartiteColumns<T>,
    seen: impl FnMut(&Column<T>),
) -> Result<Partitioned<Biadjacency<T>>, ReduceError> {
    let (c, partition) = cross(v, seen)?;
    Ok(Partitioned {
        a: Biadjacency { c },
        partition,
    })
}

/// A general factor-width-2 `A`, assembled.
pub(in crate::sgdd) struct Assembled<T> {
    adjacency: UpperAdjacency<T>,
    diagonal: Vec<T>,
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
pub(in crate::sgdd) struct Biadjacency<T> {
    c: Rows<T>,
}

/// One component of a bipartite assembly.
pub(in crate::sgdd) enum BipartiteComponent<T> {
    /// A vertex no cell couples.
    Lone(u32),
    /// Both sides in local order, the larger eliminated first, and `C` between them.
    Sides(Box<[u32]>, CrossBlock<T>),
}

impl<T: Real> PrincipalBlocks<T> for Assembled<T> {
    type Block = Cells<T>;

    fn whole(self, vertices: &[u32]) -> Self::Block {
        Cells {
            vertices: vertices.into(),
            adjacency: self.adjacency,
            diagonal: self.diagonal,
        }
    }

    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices
            .iter()
            .map(|&v| self.adjacency.upper_len(v as usize))
            .sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &v in vertices {
            for (j, &x) in self.adjacency.upper(v as usize) {
                indices.push(position[j]);
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        Cells {
            vertices: vertices.into(),
            adjacency: UpperAdjacency::new(rows),
            diagonal: vertices
                .iter()
                .map(|&v| self.diagonal[v as usize])
                .collect(),
        }
    }
}

impl<T: Real> PrincipalBlocks<T> for Biadjacency<T> {
    type Block = BipartiteComponent<T>;

    /// Transposed only when the right side is larger.
    fn whole(self, vertices: &[u32]) -> Self::Block {
        if let [v] = vertices {
            return BipartiteComponent::Lone(*v);
        }
        let n_left = self.c.n_rows() as u32;
        let (lefts, rights) = (
            0..n_left,
            n_left..(self.c.n_rows() + self.c.n_cols()) as u32,
        );
        let (vertices, cross) = larger_side_first(lefts.collect(), rights.collect(), self.c);
        BipartiteComponent::Sides(vertices, cross)
    }

    /// Its larger side as rows.
    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        if let [v] = vertices {
            return BipartiteComponent::Lone(*v);
        }
        let n_left = self.c.n_rows();
        let (lefts, rights) =
            vertices.split_at(vertices.partition_point(|&v| (v as usize) < n_left));
        let mut ptrs = Vec::with_capacity(lefts.len() + 1);
        ptrs.push(0);
        let entries = lefts.iter().map(|&i| self.c.row_len(i as usize)).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &i in lefts {
            for (j, &x) in self.c.row(i as usize) {
                indices.push(position[n_left + j] - lefts.len() as u32);
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        let c = Rows::new(ptrs, indices, values, rights.len());
        let (vertices, cross) = larger_side_first(lefts.to_vec(), rights.to_vec(), c);
        BipartiteComponent::Sides(vertices, cross)
    }
}

/// Eliminates the larger side first, transposing `C` when it is the right one.
fn larger_side_first<T: Real>(
    lefts: Vec<u32>,
    rights: Vec<u32>,
    c: Rows<T>,
) -> (Box<[u32]>, CrossBlock<T>) {
    match lefts.len() >= rights.len() {
        true => ([lefts, rights].concat().into(), CrossBlock::new(c)),
        false => (
            [rights, lefts].concat().into(),
            CrossBlock::new(c.transpose()),
        ),
    }
}

/// Row `i`'s nonzero sum as `T` stores it, its ends joined; one cast to zero underflowed, so it is out of range too.
#[inline(always)]
fn admit<T: Real>(sets: &mut UnionFind, i: usize, end: usize, x: f64) -> Result<f64, ReduceError> {
    let value = T::from(x)
        .filter(|&value| value != T::zero() && in_range(value))
        .ok_or(ReduceError::OutOfRange { vertex: i as u32 })?;
    sets.union(i, end);
    Ok(widen(value))
}

/// `C`'s summed cells, in a dense table when small (+54% without it).
fn cross<T: Real>(
    v: impl BipartiteColumns<T>,
    seen: impl FnMut(&Column<T>),
) -> Result<(Rows<T>, Partition), ReduceError> {
    let n_left = v.n_left();
    let n_right = v.n() - n_left;
    let table = n_left.saturating_mul(n_right);
    if table <= DENSE_TABLE_MAX_CELLS {
        return dense_cross(v, seen);
    }
    let counts = left_counts(v, seen);
    let columns: usize = counts.iter().sum();
    // Past the cap, the table still wins whenever the buckets would take more memory.
    match columns.saturating_mul(12) < table.saturating_mul(8) {
        true => sparse_cross(v, &counts),
        false => dense_cross(v, |_| {}),
    }
}

/// within's cap on the dense accumulator, about 40 MB.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

fn dense_cross<T: Real>(
    v: impl BipartiteColumns<T>,
    mut seen: impl FnMut(&Column<T>),
) -> Result<(Rows<T>, Partition), ReduceError> {
    let (n_left, n_right) = (v.n_left(), v.n() - v.n_left());
    let mut table = vec![0.0_f64; n_left * n_right];
    v.for_each_column(
        #[inline(always)]
        |column| {
            table[column.a as usize * n_right + column.b as usize - n_left] += column.cell();
            seen(&column);
        },
    );
    let mut sets = UnionFind::new(n_left + n_right);
    let mut ptrs = Vec::with_capacity(n_left + 1);
    ptrs.push(0);
    let (mut indices, mut cells) = (Vec::new(), Vec::new());
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        for (j, &x) in row.iter().enumerate().filter(|&(_, &x)| x != 0.0) {
            indices.push(j as u32);
            cells.push(admit::<T>(&mut sets, i, n_left + j, x)?);
        }
        ptrs.push(indices.len());
    }
    let c = Rows::new(ptrs, indices, cells, n_right).map(narrow);
    Ok((c, sets.into_partition()))
}

fn left_counts<T: Real>(
    v: impl BipartiteColumns<T>,
    mut seen: impl FnMut(&Column<T>),
) -> Vec<usize> {
    let mut counts = vec![0; v.n_left()];
    v.for_each_column(
        #[inline(always)]
        |column| {
            counts[column.a as usize] += 1;
            seen(&column);
        },
    );
    counts
}

fn sparse_cross<T: Real>(
    v: impl BipartiteColumns<T>,
    counts: &[usize],
) -> Result<(Rows<T>, Partition), ReduceError> {
    let (n_left, n_right) = (v.n_left(), v.n() - v.n_left());
    let mut buckets = Buckets::new(counts);
    v.for_each_column(
        #[inline(always)]
        |column| buckets.push(column.a as usize, column.b - n_left as u32, column.cell()),
    );
    buckets.sum::<T, f64>(n_right, n_left, |_, _, _| {})
}

/// Cells bucketed by row in column order, so each sums the same addends in the same order as a dense table.
struct Buckets {
    starts: Vec<usize>,
    cursor: Vec<usize>,
    columns: Vec<u32>,
    cells: Vec<f64>,
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

impl Buckets {
    fn new(counts: &[usize]) -> Self {
        let starts: Vec<usize> = std::iter::once(0)
            .chain(counts.iter().scan(0, |end, &count| {
                *end += count;
                Some(*end)
            }))
            .collect();
        let entries = starts[counts.len()];
        Self {
            cursor: starts[..counts.len()].to_vec(),
            starts,
            columns: vec![0; entries],
            cells: vec![0.0; entries],
        }
    }

    #[inline(always)]
    fn push(&mut self, row: usize, column: u32, cell: f64) {
        let slot = &mut self.cursor[row];
        self.columns[*slot] = column;
        self.cells[*slot] = cell;
        *slot += 1;
    }

    /// `pair` sees every summed cell with its two ends, a cancelled one too; column `j` is vertex `first_column + j`.
    fn sum<T: Real, C: CellSum>(
        self,
        n_cols: usize,
        first_column: usize,
        mut pair: impl FnMut(usize, usize, C),
    ) -> Result<(Rows<T>, Partition), ReduceError> {
        let Self {
            starts,
            cursor,
            mut columns,
            mut cells,
        } = self;
        drop(cursor);
        let mut sets = UnionFind::new(first_column + n_cols);
        let mut sums = vec![C::default(); n_cols];
        let mut touched: Vec<u32> = Vec::new();
        let mut ptrs = Vec::with_capacity(starts.len());
        ptrs.push(0);
        let mut out = 0;
        for (i, span) in starts.windows(2).enumerate() {
            for p in span[0]..span[1] {
                let j = columns[p];
                // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
                if sums[j as usize].value() == 0.0 {
                    touched.push(j);
                }
                sums[j as usize].add(cells[p]);
            }
            touched.sort_unstable();
            touched.dedup();
            // Written in place: a row's sums never outnumber its own bucket, already read.
            for &j in &touched {
                let cell = std::mem::take(&mut sums[j as usize]);
                pair(i, first_column + j as usize, cell);
                if cell.value() != 0.0 {
                    columns[out] = j;
                    cells[out] = admit::<T>(&mut sets, i, first_column + j as usize, cell.value())?;
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
        let rows = Rows::new(ptrs, columns, cells, n_cols).map(narrow);
        Ok((rows, sets.into_partition()))
    }
}
