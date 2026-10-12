use num_traits::PrimInt;

mod components;

use std::sync::Arc;

use super::reduction::{extract, folding_signs, Congruent, DirectSum, Folded, SymmetricBlock};
use super::{Component, Congruence, Frame, FromCsrError, Grounding, OffDiagonal, Sgdd};
use crate::input::{
    BipartiteGram, BipartiteGraph, BipartiteLaplacian, Canonical, Csr, IntoColumns, Laplacian,
    SignedGraph, UpperCsr,
};
use crate::storage::{
    diagonal_in_range, in_range, narrow, widen, CrossBlock, Rows, UpperAdjacency,
};
use crate::{Real, ReduceError};
pub(super) use components::Partition;
use components::UnionFind;

impl<T: Real> DirectSum<T> {
    /// Checks symmetry and range in the walk that finds the components.
    pub(super) fn from_csr<I: PrimInt>(csr: Csr<'_, T, I>) -> Result<Self, FromCsrError> {
        let csr = Canonical::of(csr)?;
        let partition = components::of_csr(&csr)?;
        Ok(Self::of_rows(&csr, partition)?)
    }

    /// Checks position and range in the walk that finds the components: the triangle is symmetric by storage.
    pub(super) fn from_upper_csr<I: PrimInt>(
        csr: UpperCsr<'_, T, I>,
    ) -> Result<Self, FromCsrError> {
        let csr = Canonical::upper(csr)?;
        let partition = components::of_upper(&csr)?;
        Ok(Self::of_rows(&csr.into_canonical(), partition)?)
    }

    fn of_rows<I: PrimInt>(
        csr: &Canonical<'_, T, I>,
        partition: Partition,
    ) -> Result<Self, ReduceError> {
        Self::extract(partition, |vertices, partition| {
            csr_component(csr, vertices, partition.position())
        })
    }

    pub(super) fn from_general(v: impl IntoColumns<T>) -> Result<Self, ReduceError> {
        let (a, partition) = assemble_general(v)?;
        let mut a = Some(a);
        Self::extract(partition, |vertices, partition| {
            let (adjacency, diagonal) = general_block(&mut a, vertices, partition);
            SymmetricBlock::new(vertices.into(), OffDiagonal::General(adjacency), diagonal)
        })
    }

    /// `[D_l C; Cᵀ D_r]`: each component takes the block elimination and reads its diagonal off the sides.
    pub(super) fn from_bipartite<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
    ) -> Result<Self, ReduceError> {
        let (c, partition) = assemble_bipartite(gram, gram.n_left())?;
        let (left, right) = gram.diagonals();
        let diagonal = |v: u32| match (v as usize).checked_sub(left.len()) {
            None => left[v as usize],
            Some(j) => right[j],
        };
        let mut c = Some(c);
        Self::extract(partition, |vertices, partition| {
            bipartite_component(&mut c, vertices, partition, diagonal)
        })
    }

    /// Each component's `A_c` as the input gives it, signed and unscaled; one with nothing to solve is dead.
    fn extract(
        partition: Partition,
        mut block: impl FnMut(&[u32], &Partition) -> SymmetricBlock<T>,
    ) -> Result<Self, ReduceError> {
        let (blocks, dead) = extract::<T, _>(&partition, |vertices| {
            let a = block(vertices, &partition);
            Ok((!a.is_dead()).then_some(a))
        })?;
        Ok(Self {
            partition,
            blocks,
            dead,
        })
    }
}

impl<T: Real> Sgdd<T> {
    /// A graph Laplacian, grounded by `g` when given: a dominant Z-matrix whose surplus is `g`, so only ranges are checked.
    pub(super) fn from_laplacian<I: PrimInt>(
        graph: Laplacian<'_, T, I>,
    ) -> Result<Self, ReduceError> {
        let (balanced, ground) = graph.parts();
        let (a, partition) = assemble_general(balanced)?;
        let mut a = Some(a);
        let (components, dead) = extract::<T, _>(&partition, |vertices| {
            let (adjacency, diagonal) = general_block(&mut a, vertices, &partition);
            let g = |i: usize| ground.map(|g| g[vertices[i] as usize]);
            // A lone vertex has no edge, so only a ground makes it a component.
            if vertices.len() == 1 && diagonal[0] + g(0).unwrap_or(T::zero()) == T::zero() {
                return Ok(None);
            }
            // `A`'s diagonal `d + g`, summed in `f64`, and each ground like any entry.
            let out_of_range = |i: usize| match g(i) {
                Some(g) => {
                    !in_range(g) || !diagonal_in_range(narrow::<T>(widen(diagonal[i]) + widen(g)))
                }
                None => !diagonal_in_range(diagonal[i]),
            };
            if let Some(i) = (0..vertices.len()).find(|&i| out_of_range(i)) {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let grounding = match ground {
                Some(g) => Grounding::of(vertices.iter().map(|&v| g[v as usize]).collect()),
                None => Grounding::ungrounded(),
            };
            let m = Congruent {
                frame: Frame::new(vertices.into(), Congruence::identity()),
                off_diagonal: OffDiagonal::General(adjacency),
                report: None,
            };
            Folded(m).balanced(grounding).map(Some)
        })?;
        Ok(Self {
            n: partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }

    /// Its diagonal is `|C|`'s row and column sums, so it is balanced and singular, and negating one side folds it.
    pub(super) fn from_bipartite_graph<I: PrimInt>(
        graph: BipartiteGraph<'_, T, I>,
    ) -> Result<Self, ReduceError> {
        Self::folded_bipartite(graph.laplacian(), graph.n_left(), Congruence::negate_from)
    }

    /// Its diagonal is `|C|`'s row and column sums, and `C ≤ 0` is already a Z-matrix.
    pub(super) fn from_bipartite_laplacian<I: PrimInt>(
        graph: BipartiteLaplacian<'_, T, I>,
    ) -> Result<Self, ReduceError> {
        Self::folded_bipartite(graph, graph.n_left(), |_| Congruence::identity())
    }

    /// `m` holds `M`'s columns, a bipartite Laplacian, and `signs` the `F` folding `A` into it; a lone vertex is general, so a bipartite component always has both sides, and only the degrees are checked.
    fn folded_bipartite(
        m: impl IntoColumns<T>,
        n_left: usize,
        signs: impl Fn(usize) -> Congruence<T>,
    ) -> Result<Self, ReduceError> {
        let (c, partition) = assemble_bipartite(m, n_left)?;
        let mut c = Some(c);
        let (components, dead) = extract::<T, _>(&partition, |vertices| {
            if let [_] = vertices {
                return Ok(None);
            }
            let (vertices, cross) = bipartite_block(&mut c, vertices, &partition);
            if let Some(i) = (cross.abs_row_sums().iter()).position(|&d| !diagonal_in_range(d)) {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let congruence = signs(cross.n_eliminated());
            let m = Congruent {
                frame: Frame::new(vertices, congruence),
                off_diagonal: OffDiagonal::Bipartite(Arc::new(cross)),
                report: None,
            };
            Folded(m).balanced(Grounding::ungrounded()).map(Some)
        })?;
        Ok(Self {
            n: partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }
}

impl<T: Real> Sgdd<T> {
    /// Dominant by shape, its surplus each pair's cancelled mass: only ranges are checked, and its signs are searched.
    pub(super) fn from_signed_graph<I: PrimInt>(
        graph: SignedGraph<'_, T, I>,
    ) -> Result<Self, ReduceError> {
        let mut surplus = vec![0.0_f64; graph.n()];
        let (a, partition) = assemble::<T, Masses>(graph, |i, j, masses: Masses| {
            let s = masses.surplus();
            surplus[i] += s;
            surplus[j] += s;
        })?;
        let surplus: Vec<T> = surplus.into_iter().map(narrow).collect();
        if let Some(vertex) = surplus.iter().position(|&s| !in_range(s)) {
            let vertex = vertex as u32;
            return Err(ReduceError::OutOfRange { vertex });
        }
        let mut a = Some(a);
        let (components, dead) = extract::<T, _>(&partition, |vertices| {
            let (adjacency, diagonal) = general_block(&mut a, vertices, &partition);
            if diagonal == [T::zero()] {
                return Ok(None);
            }
            // A zero diagonal beside a cell is its terms underflowing.
            let out_of_range = |&d: &T| d == T::zero() || !diagonal_in_range(d);
            if let Some(i) = diagonal.iter().position(out_of_range) {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let off_diagonal = OffDiagonal::General(adjacency);
            let grounding = Grounding::of(vertices.iter().map(|&v| surplus[v as usize]).collect());
            let component = match folding_signs(&off_diagonal) {
                Some(signs) => Folded(Congruent::new(vertices.into(), off_diagonal, signs)?)
                    .balanced(grounding),
                None => (Congruent::new(vertices.into(), off_diagonal, Congruence::identity())?)
                    .frustrated(grounding),
            };
            component.map(Some)
        })?;
        Ok(Self {
            n: partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }
}

/// Upper rows in local order, columns through `position`; a zero couples nothing, so it may name another component.
fn csr_component<T: Real, I: PrimInt>(
    csr: &Canonical<'_, T, I>,
    vertices: &[u32],
    position: &[u32],
) -> SymmetricBlock<T> {
    let mut diagonal = Vec::with_capacity(vertices.len());
    let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for &v in vertices {
        let v = v as usize;
        let mut d = T::zero();
        for (j, a) in csr.row(v).filter(|&(j, a)| j >= v && a != T::zero()) {
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
    let off_diagonal = OffDiagonal::General(UpperAdjacency::new(rows));
    SymmetricBlock::new(vertices.into(), off_diagonal, diagonal)
}

/// The whole input's rows are moved, not copied: a lone component takes `A` out of the input.
fn general_block<T: Real>(
    a: &mut Option<Assembled<T>>,
    vertices: &[u32],
    partition: &Partition,
) -> (UpperAdjacency<T>, Vec<T>) {
    if partition.is_whole() {
        let Assembled {
            adjacency,
            diagonal,
        } = a.take().expect("one whole component");
        return (adjacency, diagonal);
    }
    let a = a.as_ref().expect("A until the whole is taken");
    let position = partition.position();
    let mut ptrs = Vec::with_capacity(vertices.len() + 1);
    ptrs.push(0);
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for &v in vertices {
        for (j, &x) in a.adjacency.upper(v as usize) {
            indices.push(position[j]);
            values.push(x);
        }
        ptrs.push(indices.len());
    }
    let rows = Rows::new(ptrs, indices, values, vertices.len());
    let diagonal = vertices.iter().map(|&v| a.diagonal[v as usize]).collect();
    (UpperAdjacency::new(rows), diagonal)
}

fn bipartite_component<T: Real>(
    c: &mut Option<Biadjacency<T>>,
    vertices: &[u32],
    partition: &Partition,
    diagonal: impl Fn(u32) -> T,
) -> SymmetricBlock<T> {
    let (vertices, off_diagonal) = match vertices {
        [v] => {
            let empty = Rows::new(vec![0, 0], Vec::new(), Vec::new(), 1);
            (
                Box::from([*v]),
                OffDiagonal::General(UpperAdjacency::new(empty)),
            )
        }
        _ => {
            let (vertices, cross) = bipartite_block(c, vertices, partition);
            (vertices, OffDiagonal::Bipartite(Arc::new(cross)))
        }
    };
    let diagonal = vertices.iter().map(|&v| diagonal(v)).collect();
    SymmetricBlock::new(vertices, off_diagonal, diagonal)
}

/// The whole input's block is moved, not copied: a lone component takes `C` out of the input.
fn bipartite_block<T: Real>(
    c: &mut Option<Biadjacency<T>>,
    vertices: &[u32],
    partition: &Partition,
) -> (Box<[u32]>, CrossBlock<T>) {
    match partition.is_whole() {
        true => c.take().expect("one whole component").into_block(),
        false => {
            (c.as_ref().expect("C until the whole is taken")).block(vertices, partition.position())
        }
    }
}

/// Floats each folded component when `1ᵀM1 ≤ 5ε·tr M`, else grounds it on `s = M·1`, both summed by column as assembled rows cancel them: with `f = F·1` and `p_k = f_a v_a + f_b v_b`, `s_a = Σ_k bilinear_k(f_a v_a, p_k)` clamped at zero, `1ᵀM1 = Σ_k bilinear_k(p_k, p_k)` and `tr M = Σ_k [bilinear_k(f_a v_a, f_a v_a) + bilinear_k(f_b v_b, f_b v_b)]`, `ε` that of `T`; sums run in `f64` in column order over nonzero weights, the certificate's two sides as `ε·term`, an exact scaling under which neither overflows, each end's component and slot are read off the components' own vertices, an end at a dead vertex or in a frustrated component being in none; a column straddling components keeps only its own end's square.
pub(super) fn ground_or_float<T: Real>(
    v: impl IntoColumns<T>,
    components: Vec<Folded<T>>,
) -> Result<Vec<Component<T>>, ReduceError> {
    if components.is_empty() {
        return Ok(Vec::new());
    }
    let mut ends = vec![None; v.n()];
    for (c, component) in components.iter().enumerate() {
        let congruence = component.frame().congruence();
        for (slot, &vertex) in component.frame().vertices().iter().enumerate() {
            ends[vertex as usize] = Some(End {
                component: c,
                slot,
                factor: widen(congruence.factor(slot)),
            });
        }
    }
    let mut surplus: Vec<Vec<f64>> = (components.iter())
        .map(|component| vec![0.0; component.frame().vertices().len()])
        .collect();
    let mut certificates = vec![Certificate::default(); components.len()];
    let epsilon = widen(T::epsilon());
    v.for_each_column(
        #[inline(always)]
        |column| {
            if column.weight.is_zero() {
                return;
            }
            let w = |x: f64, y: f64| column.weight.bilinear(x, y);
            let a = ends[column.a as usize].map(|end| (end, end.factor * widen(column.v_a)));
            let b = ends[column.b as usize].map(|end| (end, end.factor * widen(column.v_b)));
            match (a, b) {
                (Some((a, u_a)), Some((b, u_b))) if a.component == b.component => {
                    let p = u_a + u_b;
                    surplus[a.component][a.slot] += w(u_a, p);
                    surplus[b.component][b.slot] += w(u_b, p);
                    let certificate = &mut certificates[a.component];
                    certificate.ones += w(epsilon * p, p);
                    certificate.trace += w(epsilon * u_a, u_a) + w(epsilon * u_b, u_b);
                }
                (a, b) => {
                    for (end, u) in [a, b].into_iter().flatten() {
                        surplus[end.component][end.slot] += w(u, u);
                        let certificate = &mut certificates[end.component];
                        certificate.ones += w(epsilon * u, u);
                        certificate.trace += w(epsilon * u, u);
                    }
                }
            }
        },
    );
    let decided = components.into_iter().zip(surplus).zip(certificates);
    let decide = |((component, surplus), certificate): ((Folded<T>, Vec<f64>), Certificate)| {
        if certificate.ones <= 5.0 * epsilon * certificate.trace {
            return component.balanced(Grounding::ungrounded());
        }
        let vertices = component.frame().vertices();
        let surplus: Vec<T> = surplus.into_iter().map(|s| narrow(s.max(0.0))).collect();
        if let Some(slot) = surplus.iter().position(|&s| !in_range(s)) {
            let vertex = vertices[slot];
            return Err(ReduceError::OutOfRange { vertex });
        }
        // Unproven, a surplus that vanishes in `T` is never a float.
        let vertex = vertices[0];
        match Grounding::of(surplus) {
            grounding if grounding.is_grounded() => component.balanced(grounding),
            _ => Err(ReduceError::Unscalable { vertex }),
        }
    };
    decided.map(decide).collect()
}

/// Where a column's end sits once reduced, and its `f_i`.
#[derive(Clone, Copy)]
struct End {
    component: usize,
    slot: usize,
    factor: f64,
}

/// `ε·1ᵀM1` and `ε·tr M` of one component, summed by column.
#[derive(Clone, Copy, Default)]
struct Certificate {
    ones: f64,
    trace: f64,
}

/// `C` and its components, accumulated in column order, in a dense table when it is small (+54% without it), unfiltered: a column that couples nothing adds an exact zero and zero cells drop; a cell leaving `in_range` is `OutOfRange`.
fn assemble_bipartite<T: Real>(
    v: impl IntoColumns<T>,
    n_left: usize,
) -> Result<(Biadjacency<T>, Partition), ReduceError> {
    let n_right = v.n() - n_left;
    let table = n_left.saturating_mul(n_right);
    let c = if table <= DENSE_TABLE_MAX_CELLS {
        dense_cross(v, n_left, n_right)
    } else {
        let counts = left_counts(v, n_left);
        let columns: usize = counts.iter().sum();
        // Past the cap, the table still wins whenever the buckets would take more memory.
        match columns.saturating_mul(12) < table.saturating_mul(8) {
            true => sparse_cross(v, n_left, n_right, &counts),
            false => dense_cross(v, n_left, n_right),
        }
    }?;
    Ok((Biadjacency::new(c.0), c.1))
}

/// within's cap on the dense accumulator, about 40 MB.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

fn dense_cross<T: Real>(
    v: impl IntoColumns<T>,
    n_left: usize,
    n_right: usize,
) -> Result<(Rows<T>, Partition), ReduceError> {
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

fn left_counts<T: Real>(v: impl IntoColumns<T>, n_left: usize) -> Vec<usize> {
    let mut counts = vec![0; n_left];
    v.for_each_column(
        #[inline(always)]
        |column| counts[column.a as usize] += 1,
    );
    counts
}

fn sparse_cross<T: Real>(
    v: impl IntoColumns<T>,
    n_left: usize,
    n_right: usize,
    counts: &[usize],
) -> Result<(Rows<T>, Partition), ReduceError> {
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
fn assemble_general<T: Real>(
    v: impl IntoColumns<T>,
) -> Result<(Assembled<T>, Partition), ReduceError> {
    assemble::<T, f64>(v, |_, _, _| {})
}

/// As [`assemble_general`], each cell summed as `C` and seen by `pair` with its two ends.
fn assemble<T: Real, C: CellSum>(
    v: impl IntoColumns<T>,
    pair: impl FnMut(usize, usize, C),
) -> Result<(Assembled<T>, Partition), ReduceError> {
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
    Ok((a, partition))
}

/// A general factor-width-2 `A`, assembled.
struct Assembled<T> {
    adjacency: UpperAdjacency<T>,
    diagonal: Vec<T>,
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
struct Biadjacency<T> {
    c: Rows<T>,
}

impl<T: Real> Biadjacency<T> {
    /// Panics on a stored zero: a cancelled cell couples nothing.
    fn new(c: Rows<T>) -> Self {
        let no_zero = (0..c.n_rows()).all(|i| c.row(i).all(|(_, &x)| x != T::zero()));
        assert!(no_zero, "no stored zero");
        Self { c }
    }

    fn n(&self) -> usize {
        self.c.n_rows() + self.c.n_cols()
    }

    /// The whole input as one component's block, moved, transposed only when the right side is larger.
    fn into_block(self) -> (Box<[u32]>, CrossBlock<T>) {
        let n_left = self.c.n_rows() as u32;
        let (lefts, rights) = (0..n_left, n_left..self.n() as u32);
        larger_side_first(lefts.collect(), rights.collect(), self.c)
    }

    /// One component's block, its larger side as rows, remapped through `position`; returns its local order.
    fn block(&self, vertices: &[u32], position: &[u32]) -> (Box<[u32]>, CrossBlock<T>) {
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
        larger_side_first(lefts.to_vec(), rights.to_vec(), c)
    }
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
