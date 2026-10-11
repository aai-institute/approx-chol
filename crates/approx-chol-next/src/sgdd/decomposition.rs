use num_traits::PrimInt;

mod components;

use std::sync::Arc;

use super::component::ProvenLaplacian;
use super::{
    Component, Congruence, Congruent, FromCsrError, Grounding, NullVectors, OffDiagonal, OnDemand,
    ScalingPolicy, Sgdd, Source, Unreduced,
};
use crate::input::{BipartiteGram, Canonical, Csr, IntoColumns, Laplacian, UpperCsr};
use crate::storage::{in_range, narrow, widen, CrossBlock, Rows, UpperAdjacency};
use crate::{Real, ReduceError};
use components::{Partition, UnionFind};

/// One input's storage: what its type proves decides how each component is given, and how the ones it leaves unproven ground.
trait Stored<T: Real> {
    /// A view's Gram is GDD by type; rows only claim `A`.
    fn source<'n, 'v>(nulls: Option<&'n OnDemand<'v>>) -> Source<'n, 'v> {
        Source::Gram(nulls)
    }

    /// A general factor's null vectors, built by the first component that is not already dominant.
    fn nulls<'p>(&self, _partition: &'p Partition) -> Option<OnDemand<'p>>
    where
        Self: 'p,
    {
        None
    }

    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError>;

    /// Rows ground on their own rows; a factor grounds or floats on `V` once every component is reduced.
    fn ground(&self, congruent: Vec<Congruent<T>>) -> Result<Vec<Component<T>>, ReduceError> {
        congruent.into_iter().map(Congruent::by_rows).collect()
    }
}

/// Rows that only claim `A`: each component searches its signature and scaling, its own rows decide its floating, and no scaling is known to exist.
struct ClaimedRows<'v, T, I>(Canonical<'v, T, I>);

/// A factor-width-2 `V`, so GDD by type: each component searches its signature and scaling, and `V` decides floating once every one is reduced.
struct GeneralFactor<T, V> {
    a: Option<Assembled<T>>,
    v: V,
}

/// A graph Laplacian, grounded by `g` when given: a dominant Z-matrix whose surplus is `g`, so nothing is searched and a component floats exactly when its `g` is zero.
struct LaplacianShape<'v, T> {
    a: Option<Assembled<T>>,
    ground: Option<&'v [T]>,
}

/// `[D_l C; Cᵀ D_r]` of a bipartite Gram, GDD by type: each component searches as a factor's does, takes the block elimination, and reads its diagonal off the sides.
struct BipartiteFactor<'v, T, I> {
    c: Option<Biadjacency<T>>,
    gram: BipartiteGram<'v, T, I>,
}

/// A bipartite graph: its diagonal is `|C|`'s row and column sums, so it is balanced and singular, its fold proves the signature, and every component floats untested.
struct BipartiteShape<T> {
    c: Option<Biadjacency<T>>,
    fold: Fold,
}

/// How a bipartite graph's `C` folds into a Z-matrix.
#[derive(Clone, Copy)]
pub(super) enum Fold {
    /// `C ≥ 0`: negating one side folds it.
    Signless,
    /// `C ≤ 0`: already a Z-matrix.
    Laplacian,
}

impl Fold {
    fn signs<T: Real>(self, n_eliminated: usize) -> Congruence<T> {
        match self {
            Self::Signless => Congruence::negate_from(n_eliminated),
            Self::Laplacian => Congruence::identity(),
        }
    }
}

/// Checks symmetry and range in the walk that finds the components.
pub(super) fn csr<T: Real, I: PrimInt>(
    csr: Csr<'_, T, I>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, FromCsrError> {
    let csr = Canonical::of(csr)?;
    let partition = components::of_csr(&csr)?;
    Ok(reduce(partition, ClaimedRows(csr), policy)?)
}

/// Checks position and range in the walk that finds the components: the triangle is symmetric by storage.
pub(super) fn upper_csr<T: Real, I: PrimInt>(
    csr: UpperCsr<'_, T, I>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, FromCsrError> {
    let csr = Canonical::upper(csr)?;
    let partition = components::of_upper(&csr)?;
    Ok(reduce(
        partition,
        ClaimedRows(csr.into_canonical()),
        policy,
    )?)
}

pub(super) fn gram<T: Real>(
    v: impl IntoColumns<T>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    let (a, partition) = assemble_general(v)?;
    reduce(partition, GeneralFactor { a: Some(a), v }, policy)
}

pub(super) fn bipartite_gram<T: Real, I: PrimInt>(
    gram: BipartiteGram<'_, T, I>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    let (c, partition) = assemble_bipartite(gram, gram.n_left())?;
    reduce(partition, BipartiteFactor { c: Some(c), gram }, policy)
}

/// Nothing is searched, so no policy applies.
pub(super) fn laplacian<T: Real, I: PrimInt>(
    graph: Laplacian<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    let (a, partition) = assemble_general(graph)?;
    let ground = graph.ground();
    let stored = LaplacianShape { a: Some(a), ground };
    reduce(partition, stored, &ScalingPolicy::default())
}

/// Nothing is searched, so no policy applies.
pub(super) fn bipartite_graph<T: Real>(
    graph: impl IntoColumns<T>,
    n_left: usize,
    fold: Fold,
) -> Result<Sgdd<T>, ReduceError> {
    let (c, partition) = assemble_bipartite(graph, n_left)?;
    let stored = BipartiteShape { c: Some(c), fold };
    reduce(partition, stored, &ScalingPolicy::default())
}

/// Dead components are dropped, proven ones are built as their input proves, and the rest reduce, then ground as their input decides.
fn reduce<T: Real, S: Stored<T>>(
    partition: Partition,
    mut stored: S,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    let nulls = stored.nulls(&partition);
    let source = S::source(nulls.as_ref());
    let (mut components, mut congruent, mut dead) = (Vec::new(), Vec::new(), Vec::new());
    for vertices in partition.components() {
        match stored.given(vertices, &partition)? {
            Given::Dead(vertex) => dead.push(vertex),
            Given::Proven(component) => components.push(component),
            Given::Unreduced(a) => congruent.push(a.reduce(policy, source)?),
        }
    }
    components.extend(stored.ground(congruent)?);
    Ok(Sgdd {
        n: partition.n(),
        components,
        dead: dead.into_boxed_slice(),
    })
}

impl<T: Real, I: PrimInt> Stored<T> for ClaimedRows<'_, T, I> {
    fn source<'n, 'v>(_: Option<&'n OnDemand<'v>>) -> Source<'n, 'v> {
        Source::Rows
    }

    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError> {
        let a = csr_component(&self.0, vertices, partition.position());
        Ok(Given::unreduced(a))
    }
}

impl<T: Real, V: IntoColumns<T>> Stored<T> for GeneralFactor<T, V> {
    fn nulls<'p>(&self, partition: &'p Partition) -> Option<OnDemand<'p>>
    where
        Self: 'p,
    {
        let v = self.v;
        Some(OnDemand::new(Box::new(move || {
            NullVectors::of(v, partition.components())
        })))
    }

    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError> {
        let (adjacency, diagonal) = general_block(&mut self.a, vertices, partition);
        let off_diagonal = OffDiagonal::General(adjacency);
        Ok(Given::unreduced(Unreduced::new(
            vertices.into(),
            off_diagonal,
            diagonal,
        )))
    }

    fn ground(&self, congruent: Vec<Congruent<T>>) -> Result<Vec<Component<T>>, ReduceError> {
        ground_or_float(self.v, congruent)
    }
}

impl<T: Real> Stored<T> for LaplacianShape<'_, T> {
    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError> {
        let (adjacency, diagonal) = general_block(&mut self.a, vertices, partition);
        let ground = (self.ground).map(|g| vertices.iter().map(|&v| g[v as usize]).collect());
        let laplacian = ProvenLaplacian::new(vertices.into(), adjacency, diagonal, ground);
        match laplacian.dead() {
            Some(vertex) => Ok(Given::Dead(vertex)),
            None => Ok(Given::Proven(laplacian.into_component()?)),
        }
    }
}

impl<T: Real, I: PrimInt> Stored<T> for BipartiteFactor<'_, T, I> {
    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError> {
        let (left, right) = self.gram.diagonals();
        let diagonal = |v: u32| match (v as usize).checked_sub(left.len()) {
            None => left[v as usize],
            Some(j) => right[j],
        };
        let a = bipartite_component(&mut self.c, vertices, partition, diagonal);
        Ok(Given::unreduced(a))
    }

    fn ground(&self, congruent: Vec<Congruent<T>>) -> Result<Vec<Component<T>>, ReduceError> {
        ground_or_float(self.gram, congruent)
    }
}

/// A lone vertex is general, so a bipartite component always has both sides.
impl<T: Real> Stored<T> for BipartiteShape<T> {
    fn given(&mut self, vertices: &[u32], partition: &Partition) -> Result<Given<T>, ReduceError> {
        match vertices {
            [v] => Ok(Given::Dead(*v)),
            _ => {
                let (vertices, cross) = bipartite_block(&mut self.c, vertices, partition);
                let signs = self.fold.signs(cross.n_eliminated());
                Ok(Given::Proven(Component::folded(vertices, cross, signs)?))
            }
        }
    }
}

/// A component as its input gives it.
enum Given<T> {
    /// A lone vertex with nothing to solve.
    Dead(u32),
    /// Built as its input's type proves: nothing is searched.
    Proven(Component<T>),
    Unreduced(Unreduced<T>),
}

impl<T: Real> Given<T> {
    fn unreduced(a: Unreduced<T>) -> Self {
        match a.is_dead() {
            true => Self::Dead(a.vertices[0]),
            false => Self::Unreduced(a),
        }
    }
}

/// Upper rows in local order, columns through `position`; a zero couples nothing, so it may name another component.
fn csr_component<T: Real, I: PrimInt>(
    csr: &Canonical<'_, T, I>,
    vertices: &[u32],
    position: &[u32],
) -> Unreduced<T> {
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
    Unreduced::new(vertices.into(), off_diagonal, diagonal)
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
) -> Unreduced<T> {
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
    Unreduced::new(vertices, off_diagonal, diagonal)
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

/// Floats each folded component when `1ᵀM1 ≤ 5ε·tr M`, else grounds it on `s = M·1`, both summed by column as assembled rows cancel them: with `f = F·1` and `p_k = f_a v_a + f_b v_b`, `s_a = Σ_k bilinear_k(f_a v_a, p_k)` clamped at zero, `1ᵀM1 = Σ_k bilinear_k(p_k, p_k)` and `tr M = Σ_k [bilinear_k(f_a v_a, f_a v_a) + bilinear_k(f_b v_b, f_b v_b)]`, `ε` that of `T`; sums run in `f64` in column order over nonzero weights, the certificate's two sides as `ε·term`, an exact scaling under which neither overflows, each end's component and slot are read off the components' own vertices, an end at a dead vertex being in none; a column straddling components keeps only its own end's square, a frustrated component is definite and keeps its rows' surplus.
fn ground_or_float<T: Real>(
    v: impl IntoColumns<T>,
    components: Vec<Congruent<T>>,
) -> Result<Vec<Component<T>>, ReduceError> {
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
    let decide = |((component, surplus), certificate): ((Congruent<T>, Vec<f64>), Certificate)| {
        if !component.folds() {
            return component.by_rows();
        }
        if certificate.ones <= 5.0 * epsilon * certificate.trace {
            return component.grounded(Grounding::ungrounded());
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
            grounding if grounding.is_grounded() => component.grounded(grounding),
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
    buckets.sum(n_right, n_left)
}

/// Cells bucketed by row in column order, so each sums the same addends in the same order as a dense table.
struct Buckets {
    starts: Vec<usize>,
    cursor: Vec<usize>,
    columns: Vec<u32>,
    cells: Vec<f64>,
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

    /// Column `j` is vertex `first_column + j`.
    fn sum<T: Real>(
        self,
        n_cols: usize,
        first_column: usize,
    ) -> Result<(Rows<T>, Partition), ReduceError> {
        let mut sums = vec![0.0_f64; n_cols];
        let mut touched: Vec<u32> = Vec::new();
        let mut rows = CellRows::new(self.starts.len() - 1, n_cols, first_column);
        for span in self.starts.windows(2) {
            for (&j, &x) in self.columns[span[0]..span[1]]
                .iter()
                .zip(&self.cells[span[0]..span[1]])
            {
                // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
                if sums[j as usize] == 0.0 {
                    touched.push(j);
                }
                sums[j as usize] += x;
            }
            touched.sort_unstable();
            touched.dedup();
            for &j in &touched {
                rows.push(j as usize, std::mem::take(&mut sums[j as usize]))?;
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
    let (rows, partition) = buckets.sum(n, 0)?;
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
