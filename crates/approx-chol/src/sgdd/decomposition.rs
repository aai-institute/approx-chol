use std::ops::Range;

use num_traits::PrimInt;
use rayon::prelude::*;

mod assembly;
mod partition;

use super::block::ReduceError;
use crate::input::{Canonical, CsrError};
use crate::storage::{tally_abs, OffDiagonal, Rows, UpperAdjacency};
use crate::{Index, Real};
pub(super) use assembly::{
    assemble_incidence, assemble_values, AdjacencyLayout, Biadjacency, StrictUpperTriangle,
};
use partition::Partition;
use partition::UnionFind;

/// An assembled `A` and its connected components, found from the same cells.
pub(super) struct Partitioned<A, I> {
    a: A,
    partition: Partition<I>,
}

/// Unions the pattern in the walk that checks symmetry and range.
pub(super) fn of_csr<T: Real, J: PrimInt, I: Index>(
    csr: Canonical<'_, T, J>,
) -> Result<Partitioned<Canonical<'_, T, J>, I>, CsrError> {
    let mut sets = UnionFind::<I>::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    let partition = sets.into_partition();
    Ok(Partitioned { a: csr, partition })
}

/// Unions the pattern in the walk that checks range: upper storage has no mirror to check.
pub(super) fn of_upper<T: Real, J: PrimInt, I: Index>(
    csr: Canonical<'_, T, J>,
) -> Result<Partitioned<Canonical<'_, T, J>, I>, CsrError> {
    let mut sets = UnionFind::<I>::new(csr.n());
    csr.walk_upper(|i, j, _| sets.union(i, j))?;
    let partition = sets.into_partition();
    Ok(Partitioned { a: csr, partition })
}

/// The pivot of a vertex no entry couples: its diagonal, or its surplus where no diagonal is stored.
pub(super) trait LonePivot<T> {
    fn lone(&self, v: usize) -> T;
}

/// An assembled `A` cut into its components' blocks.
pub(super) trait PrincipalBlocks<T, I> {
    type Block;

    /// How far a component's ascending vertices rotate left so its eliminated side comes first.
    fn rotation(&self, _ascending: &[I]) -> usize {
        0
    }

    /// The whole input as one component, moved where it is owned: its vertices are their own positions.
    fn whole(self, vertices: &[I]) -> Self::Block;

    /// One component, its columns remapped through `position`.
    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block;
}

/// An assembled `A` still to cut, beside its vertices by component.
pub(super) type Arranged<A, I, T> = (Cuts<A, I>, Components<I, T>);

/// An assembled `A` still to cut, and each vertex's position within its ascending component, none when whole.
pub(super) struct Cuts<A, I> {
    a: A,
    position: Option<Vec<I>>,
}

impl<A, I: Index> Partitioned<A, I> {
    /// Each component told apart in the partition's array: a coupled one rotated so its eliminated side leads, a lone or dead one by its pivot.
    pub(super) fn arrange<T: Real>(self) -> Result<Arranged<A, I, T>, ReduceError>
    where
        A: PrincipalBlocks<T, I> + LonePivot<T>,
    {
        let Self { a, partition } = self;
        let Partition {
            mut order,
            ends,
            position,
        } = partition;
        let ranges = (ends.iter()).scan(0, |start, &end| {
            Some(std::mem::replace(start, end.index())..end.index())
        });
        // Past `1/ε` vertices, floor weights in series fall below `MIN_POSITIVE`; a dead one is alone.
        let too_long = |len: usize| T::from(len).is_none_or(|n| n * T::epsilon() >= T::one());
        if let Some(range) = ranges.clone().find(|range| too_long(range.len())) {
            let vertex = order[range.start].index();
            return Err(ReduceError::TooLarge { vertex });
        }
        let (mut coupled, mut lone, mut dead) = (Vec::new(), Vec::new(), Vec::new());
        for range in ranges {
            if range.len() > 1 {
                let rotation = a.rotation(&order[range.clone()]);
                order[range.clone()].rotate_left(rotation);
                coupled.push(range);
                continue;
            }
            let v = order[range.start];
            let vertex = v.index();
            match a.lone(vertex) {
                pivot if pivot == T::zero() => dead.push(v),
                pivot if pivot > T::zero() && super::diagonal_in_range(pivot) => {
                    lone.push((v, pivot))
                }
                pivot if pivot < T::zero() => {
                    return Err(ReduceError::NonpositiveDiagonal { vertex })
                }
                _ => return Err(ReduceError::OutOfRange { vertex }),
            }
        }
        let components = Components {
            numbering: 0..order.len(),
            vertices: order,
            coupled,
            lone,
            dead,
        };
        Ok((Cuts { a, position }, components))
    }

    /// The reduction `block` makes of each coupled component, the assembly freed.
    pub(super) fn reduced<T: Real, B: Send>(
        self,
        block: impl Fn(&[I], <A as PrincipalBlocks<T, I>>::Block) -> Result<B, ReduceError> + Sync,
    ) -> Result<Summands<B, I, T>, ReduceError>
    where
        A: PrincipalBlocks<T, I> + LonePivot<T> + Sync,
    {
        let (cuts, components) = self.arrange()?;
        let blocks = cuts.extract(&components, block)?;
        Ok(Summands { components, blocks })
    }
}

impl<A, I: Index> Cuts<A, I> {
    /// Each coupled component as `block` makes it, in component order, the assembly freed; a connected input moves its assembly in whole.
    pub(super) fn extract<'p, T: Real, B: Send>(
        self,
        components: &'p Components<I, T>,
        block: impl Fn(&'p [I], <A as PrincipalBlocks<T, I>>::Block) -> Result<B, ReduceError> + Sync,
    ) -> Result<Vec<B>, ReduceError>
    where
        A: PrincipalBlocks<T, I> + Sync,
    {
        let Self { a, position } = self;
        let count = components.coupled.len();
        Ok(match (count, position) {
            (0, _) => Vec::new(),
            (_, None) => {
                let whole = components.component(0);
                vec![block(whole, a.whole(whole))?]
            }
            (_, Some(position)) => {
                let cut = |k: usize| {
                    let vertices = components.component(k);
                    block(vertices, a.component(vertices, &position))
                };
                // About `2^PARALLEL_BITS` vertices a task: waking the pool for less costs more than it saves.
                let work = components.coupled.iter().map(|c| c.len() as u64).sum();
                let per_task = crate::per_task(count, work, PARALLEL_BITS);
                let parallel = per_task < count && rayon::current_num_threads() > 1;
                let blocks: Vec<_> = match parallel {
                    true => (0..count)
                        .into_par_iter()
                        .with_min_len(per_task)
                        .map(cut)
                        .collect(),
                    false => (0..count).map(cut).collect(),
                };
                // Kept past extraction, an owned assembly raised the Gram peak 11–22%.
                drop(a);
                drop(position);
                if let Some(&error) = blocks.iter().find_map(|b| b.as_ref().err()) {
                    return Err(error);
                }
                // In place: a block's result is the block's own size.
                blocks.into_iter().filter_map(Result::ok).collect()
            }
        })
    }
}

const PARALLEL_BITS: u32 = 12;

/// `A`'s vertices by component in the partition's array, components by lowest vertex: each coupled component in local order, its eliminated side first; the lone and dead ones also listed with their pivots.
#[derive(Clone, Debug)]
pub(crate) struct Components<I, T> {
    pub(crate) vertices: Vec<I>,
    /// Each coupled component's range of `vertices`, the ranges ascending.
    pub(crate) coupled: Vec<Range<usize>>,
    /// Vertices no entry couples, each with its surplus `σ`, which is its whole pivot.
    pub(crate) lone: Vec<(I, T)>,
    /// One-vertex components whose pivot is zero: an in-place solve zeroes them.
    pub(crate) dead: Vec<I>,
    /// The factored matrix's vertices in `A`'s numbering: all of `A`'s, or a complement's kept side.
    pub(crate) numbering: Range<usize>,
}

impl<I, T> Components<I, T> {
    /// Coupled component `k`'s vertices of `A`, in local order.
    pub(crate) fn component(&self, k: usize) -> &[I] {
        &self.vertices[self.coupled[k].clone()]
    }
}

/// The factored matrix's direct summands: its vertices by component, and each coupled component's block.
#[derive(Clone, Debug)]
pub(crate) struct Summands<B, I, T> {
    pub(crate) components: Components<I, T>,
    /// Block `k` is coupled component `k`'s.
    pub(crate) blocks: Vec<B>,
}

/// One block `A_c` as the input gave it, with its rows' `Σ_j |a_ij|`.
pub(super) struct SymmetricBlock<'p, T, I> {
    /// In `A`'s numbering, in local order.
    pub(super) vertices: &'p [I],
    pub(super) off_diagonal: OffDiagonal<T, I>,
    pub(super) rows: RowSums<T>,
}

impl<'p, T: Real, I: Index> SymmetricBlock<'p, T, I> {
    pub(super) fn new(
        vertices: &'p [I],
        (off_diagonal, rows): (OffDiagonal<T, I>, RowSums<T>),
    ) -> Self {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(
            rows.diagonal.len(),
            off_diagonal.n(),
            "one diagonal per row"
        );
        Self {
            vertices,
            off_diagonal,
            rows,
        }
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
pub(super) struct RowSums<T> {
    pub(super) diagonal: Vec<T>,
    pub(super) abs: Vec<T>,
}

impl<T: Real, J: PrimInt> Canonical<'_, T, J> {
    /// Upper rows in local order, columns remapped through `position`; a zero couples nothing, so it may name another component.
    fn rows<I: Index>(
        &self,
        vertices: &[I],
        position: impl Fn(usize) -> I,
    ) -> (OffDiagonal<T, I>, RowSums<T>) {
        let mut diagonal = Vec::with_capacity(vertices.len());
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|v| self.upper_len(v.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        // `Σ_j |a_ij|` tallied while the row is at hand, saving `abs_row_sums`'s pass.
        let mut abs = vec![0.0; vertices.len()];
        for (i, v) in vertices.iter().enumerate() {
            let (d, upper) = self.upper_row(v.index());
            let row = upper.filter(|&(_, a)| a != T::zero()).map(|(j, a)| {
                let j = position(j);
                indices.push(j);
                values.push(a);
                (j.index(), a)
            });
            tally_abs(&mut abs, i, row);
            diagonal.push(d);
            ptrs.push(indices.len());
        }
        indices.shrink_to_fit();
        values.shrink_to_fit();
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        let off_diagonal = OffDiagonal::General(UpperAdjacency::new(rows));
        let abs = abs.into_iter().map(crate::narrow).collect();
        (off_diagonal, RowSums { diagonal, abs })
    }
}

impl<T: Real, J: PrimInt, I: Index> PrincipalBlocks<T, I> for Canonical<'_, T, J> {
    type Block = (OffDiagonal<T, I>, RowSums<T>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        self.rows(vertices, I::from_index)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        self.rows(vertices, |j| position[j])
    }
}

impl<T: Real, J: PrimInt> LonePivot<T> for Canonical<'_, T, J> {
    fn lone(&self, v: usize) -> T {
        self.row(v)
            .find(|&(j, _)| j == v)
            .map_or(T::zero(), |(_, a)| a)
    }
}
