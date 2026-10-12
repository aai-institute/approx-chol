use num_traits::PrimInt;

mod components;

use std::borrow::Cow;

use super::{within_series_floor, Component, Grounding, Sgdd};
use crate::input::{BipartiteColumns, Canonical, IntoColumns, Laplacian, SignedGraph};
use crate::storage::{in_range, narrow, widen, CrossBlock, Rows, UpperAdjacency};
use crate::{Real, ReduceError};
use components::UnionFind;
pub(super) use components::{of_csr, of_upper, Partition};

/// An assembled `A` and its connected components, found from the same cells.
pub(super) struct Partitioned<A> {
    a: A,
    partition: Partition,
}

/// `⊕ B_c` over a partition's live components; a dead one's vertices solve to `0`.
pub(super) struct DirectSum<B> {
    partition: Partition,
    blocks: Vec<B>,
    dead: Vec<u32>,
}

impl<B> DirectSum<B> {
    /// Every block at once, for a pass that crosses components.
    pub(super) fn reduce_all<C>(
        self,
        reduce: impl FnOnce(&Partition, Vec<B>) -> Result<Vec<C>, ReduceError>,
    ) -> Result<DirectSum<C>, ReduceError> {
        let blocks = reduce(&self.partition, self.blocks)?;
        Ok(DirectSum {
            partition: self.partition,
            blocks,
            dead: self.dead,
        })
    }
}

impl<T> From<DirectSum<Component<T>>> for Sgdd<T> {
    fn from(sum: DirectSum<Component<T>>) -> Self {
        Self {
            n: sum.partition.n(),
            components: sum.blocks,
            dead: sum.dead.into_boxed_slice(),
        }
    }
}

/// An assembled `A` cut into its components' blocks.
pub(super) trait Assembly<T> {
    type Block;

    /// The whole input as one component, moved where it is owned: its vertices are their own positions.
    fn whole(self, vertices: &[u32]) -> Self::Block;

    /// One component, its columns remapped through `position`.
    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block;
}

impl<A> Partitioned<A> {
    /// Each component's block, `None` when it is dead; a component past `1/ε` vertices of `T` is rejected first, which a dead one, alone, never is.
    pub(super) fn extract<T: Real, B>(
        self,
        mut block: impl FnMut(<A as Assembly<T>>::Block) -> Result<Option<B>, ReduceError>,
    ) -> Result<DirectSum<B>, ReduceError>
    where
        A: Assembly<T>,
    {
        let Self { a, partition } = self;
        let mut a = Some(a);
        let (mut blocks, mut dead) = (Vec::new(), Vec::new());
        for vertices in partition.components() {
            within_series_floor::<T>(vertices)?;
            let cut = match partition.is_whole() {
                true => a.take().expect("one whole component").whole(vertices),
                false => (a.as_ref().expect("A until the whole is taken"))
                    .component(vertices, partition.position()),
            };
            match block(cut)? {
                Some(b) => blocks.push(b),
                None => dead.extend_from_slice(vertices),
            }
        }
        Ok(DirectSum {
            partition,
            blocks,
            dead,
        })
    }
}

/// One component of a bipartite assembly.
pub(super) enum BipartiteComponent<T> {
    /// A vertex no cell couples.
    Lone(u32),
    /// Both sides in local order, the larger eliminated first, and `C` between them.
    Sides(Box<[u32]>, CrossBlock<T>),
}

/// One component's cells as the input gives them: its vertices in local order, strict upper triangle and diagonal.
pub(super) struct Cells<T> {
    pub(super) vertices: Box<[u32]>,
    pub(super) adjacency: UpperAdjacency<T>,
    pub(super) diagonal: Vec<T>,
}

/// A shape's assembled `A`, dominant by shape, with the surplus its cells decide: a Laplacian's ground (none if ungrounded), a signed graph's cancelled mass.
pub(super) struct ShapeDominant<'s, T: Clone> {
    a: Assembled<T>,
    surplus: Option<Cow<'s, [T]>>,
}

/// A graph Laplacian's cells, grounded by `g` when given.
pub(super) fn assemble_laplacian<'g, T: Real, I: PrimInt>(
    graph: Laplacian<'g, T, I>,
) -> Result<Partitioned<ShapeDominant<'g, T>>, ReduceError> {
    let (balanced, ground) = graph.parts();
    let Partitioned { a, partition } = assemble_general(balanced)?;
    let surplus = ground.map(Cow::Borrowed);
    Ok(Partitioned {
        a: ShapeDominant { a, surplus },
        partition,
    })
}

/// The signed graph's cells, each vertex's surplus its pairs' cancelled mass.
pub(super) fn assemble_signed<T: Real, I: PrimInt>(
    graph: SignedGraph<'_, T, I>,
) -> Result<Partitioned<ShapeDominant<'static, T>>, ReduceError> {
    let mut surplus = vec![0.0_f64; graph.n()];
    let Partitioned { a, partition } = assemble(graph, |i, j, masses: Masses| {
        let s = masses.surplus();
        surplus[i] += s;
        surplus[j] += s;
    })?;
    let surplus: Vec<T> = surplus.into_iter().map(narrow).collect();
    if let Some(vertex) = surplus.iter().position(|&s| !in_range(s)) {
        let vertex = vertex as u32;
        return Err(ReduceError::OutOfRange { vertex });
    }
    let surplus = Some(Cow::Owned(surplus));
    Ok(Partitioned {
        a: ShapeDominant { a, surplus },
        partition,
    })
}

impl<T: Real, I: PrimInt> Assembly<T> for Canonical<'_, T, I> {
    type Block = Cells<T>;

    fn whole(self, vertices: &[u32]) -> Self::Block {
        self.component(vertices, vertices)
    }

    /// Upper rows in local order; a zero couples nothing, so it may name another component.
    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let mut diagonal = Vec::with_capacity(vertices.len());
        let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        for &v in vertices {
            let v = v as usize;
            let mut d = T::zero();
            for (j, a) in self.row(v).filter(|&(j, a)| j >= v && a != T::zero()) {
                match j == v {
                    true => d = a,
                    false => {
                        indices.push(position[j]);
                        values.push(a);
                    }
                }
            }
            diagonal.push(d);
            ptrs.push(indices.len());
        }
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        Cells {
            vertices: vertices.into(),
            adjacency: UpperAdjacency::new(rows),
            diagonal,
        }
    }
}

impl<T: Real> Assembly<T> for Assembled<T> {
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
        let (mut indices, mut values) = (Vec::new(), Vec::new());
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

impl<T: Real> Assembly<T> for ShapeDominant<'_, T> {
    type Block = (Cells<T>, Grounding<T>);

    fn whole(self, vertices: &[u32]) -> Self::Block {
        let grounding =
            (self.surplus).map_or_else(Grounding::ungrounded, |s| Grounding::of(s.into_owned()));
        (self.a.whole(vertices), grounding)
    }

    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let gather =
            |s: &Cow<'_, [T]>| Grounding::of(vertices.iter().map(|&v| s[v as usize]).collect());
        let grounding = self
            .surplus
            .as_ref()
            .map_or_else(Grounding::ungrounded, gather);
        (self.a.component(vertices, position), grounding)
    }
}

impl<T: Real> Assembly<T> for Biadjacency<T> {
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
        let (mut indices, mut values) = (Vec::new(), Vec::new());
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

/// `C` and its components, accumulated in column order, in a dense table when it is small (+54% without it), unfiltered: a column that couples nothing adds an exact zero and zero cells drop; a cell leaving `in_range` is `OutOfRange`.
pub(super) fn assemble_bipartite<T: Real>(
    v: impl BipartiteColumns<T>,
) -> Result<Partitioned<Biadjacency<T>>, ReduceError> {
    let n_left = v.n_left();
    let n_right = v.n() - n_left;
    let table = n_left.saturating_mul(n_right);
    let c = if table <= DENSE_TABLE_MAX_CELLS {
        dense_cross(v)
    } else {
        let counts = left_counts(v);
        let columns: usize = counts.iter().sum();
        // Past the cap, the table still wins whenever the buckets would take more memory.
        match columns.saturating_mul(12) < table.saturating_mul(8) {
            true => sparse_cross(v, &counts),
            false => dense_cross(v),
        }
    }?;
    Ok(Partitioned {
        a: Biadjacency { c: c.0 },
        partition: c.1,
    })
}

/// within's cap on the dense accumulator, about 40 MB.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

fn dense_cross<T: Real>(v: impl BipartiteColumns<T>) -> Result<(Rows<T>, Partition), ReduceError> {
    let (n_left, n_right) = (v.n_left(), v.n() - v.n_left());
    let mut table = vec![0.0_f64; n_left * n_right];
    v.for_each_column(
        #[inline(always)]
        |column| table[column.a as usize * n_right + column.b as usize - n_left] += column.cell(),
    );
    let mut rows = CellRows::new(n_left, n_right, n_left);
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        for (j, &x) in row.iter().enumerate() {
            rows.push(j, x)?;
        }
        rows.end_row();
    }
    Ok(rows.finish())
}

fn left_counts<T: Real>(v: impl BipartiteColumns<T>) -> Vec<usize> {
    let mut counts = vec![0; v.n_left()];
    v.for_each_column(
        #[inline(always)]
        |column| counts[column.a as usize] += 1,
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

/// A pair's positive and negative weight summed apart: its cell is `p − q`, and each end's surplus `2·min(p, q)` is zero exactly when nothing cancels.
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

    /// Column `j` is vertex `first_column + j`; `pair` sees every summed cell, a cancelled one too.
    fn sum<T: Real, C: CellSum>(
        self,
        n_cols: usize,
        first_column: usize,
        mut pair: impl FnMut(usize, usize, C),
    ) -> Result<(Rows<T>, Partition), ReduceError> {
        let mut sums = vec![C::default(); n_cols];
        let mut touched: Vec<u32> = Vec::new();
        let mut rows = CellRows::new(self.starts.len() - 1, n_cols, first_column);
        for (i, span) in self.starts.windows(2).enumerate() {
            for (&j, &x) in self.columns[span[0]..span[1]]
                .iter()
                .zip(&self.cells[span[0]..span[1]])
            {
                // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
                if sums[j as usize].value() == 0.0 {
                    touched.push(j);
                }
                sums[j as usize].add(x);
            }
            touched.sort_unstable();
            touched.dedup();
            for &j in &touched {
                let cell = std::mem::take(&mut sums[j as usize]);
                pair(i, first_column + j as usize, cell);
                rows.push(j as usize, cell.value())?;
            }
            touched.clear();
            rows.end_row();
        }
        Ok(rows.finish())
    }
}

/// Rows built in order from `f64` sums: a zero is dropped, any other is cast once, must stay [`in_range`], and joins its ends' components.
struct CellRows<T> {
    ptrs: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<T>,
    n_cols: usize,
    /// Column `j` is vertex `first_column + j`: `0` for `A`'s upper rows, `n_left` for `C`.
    first_column: usize,
    sets: UnionFind,
}

impl<T: Real> CellRows<T> {
    fn new(n_rows: usize, n_cols: usize, first_column: usize) -> Self {
        let mut ptrs = Vec::with_capacity(n_rows + 1);
        ptrs.push(0);
        Self {
            ptrs,
            indices: Vec::new(),
            values: Vec::new(),
            n_cols,
            first_column,
            sets: UnionFind::new(first_column + n_cols),
        }
    }

    /// A nonzero sum cast to zero underflowed, so it is out of range too; the error names the row.
    fn push(&mut self, j: usize, x: f64) -> Result<(), ReduceError> {
        if x == 0.0 {
            return Ok(());
        }
        let vertex = (self.ptrs.len() - 1) as u32;
        let value = T::from(x)
            .filter(|&value| value != T::zero() && in_range(value))
            .ok_or(ReduceError::OutOfRange { vertex })?;
        self.indices.push(j as u32);
        self.values.push(value);
        self.sets.union(vertex as usize, self.first_column + j);
        Ok(())
    }

    fn end_row(&mut self) {
        self.ptrs.push(self.indices.len());
    }

    fn finish(self) -> (Rows<T>, Partition) {
        let rows = Rows::new(self.ptrs, self.indices, self.values, self.n_cols);
        (rows, self.sets.into_partition())
    }
}

/// `A`'s strict upper triangle, diagonal and components, cells `bilinear(v_a, v_b)` summed by pair in column order, zero cells dropped; a value leaving `in_range` is `OutOfRange`.
pub(super) fn assemble_general<T: Real>(
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
        diagonal: diagonal.into_iter().map(narrow).collect(),
    };
    Ok(Partitioned { a, partition })
}

/// A general factor-width-2 `A`, assembled.
pub(super) struct Assembled<T> {
    adjacency: UpperAdjacency<T>,
    diagonal: Vec<T>,
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
pub(super) struct Biadjacency<T> {
    c: Rows<T>,
}

/// Eliminates the larger side, transposing `C` when it is the right one; returns the local order, eliminated first.
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
