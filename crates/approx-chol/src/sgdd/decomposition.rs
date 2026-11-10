use std::marker::PhantomData;
use std::ops::Range;

use num_traits::PrimInt;
use rayon::prelude::*;

mod assembly;
mod partition;

use super::block::CoupledBlock;
use crate::input::Canonical;
use crate::storage::{tally_abs, OffDiagonal, Rows, UpperAdjacency};
use crate::threads::task;
use crate::{BuildError, Index, Real, Threads};
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

/// Unions each nonzero above the diagonal: a checked `A` stored whole mirrors them below.
pub(super) fn of_csr<T: Real, J: PrimInt, I: Index>(
    csr: Canonical<'_, T, J>,
) -> Partitioned<Canonical<'_, T, J>, I> {
    let mut sets = UnionFind::<I>::new(csr.n());
    for i in 0..csr.n() {
        for (j, a) in csr.above(i) {
            if a != T::zero() {
                sets.union(i, j);
            }
        }
    }
    let partition = sets.into_partition();
    Partitioned { a: csr, partition }
}

/// The pivot of a vertex no entry couples: its diagonal, or its surplus where no diagonal is stored.
pub(super) trait LonePivot<T> {
    fn lone(&self, v: usize) -> T;
}

/// An assembled `A` cut into its components' blocks.
pub(super) trait PrincipalBlocks<T, I> {
    type Block;

    /// How far a component's ascending vertices rotate left so its eliminated side comes first.
    fn rotation(&self, ascending: &[I]) -> usize;

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
    pub(super) fn arrange<T: Real>(self) -> Result<Arranged<A, I, T>, BuildError>
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
            return Err(BuildError::TooLarge { vertex });
        }
        let (mut coupled, mut lone, mut dead, mut pivots) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut end = 0;
        for range in ranges {
            if range.len() > 1 {
                let rotation = a.rotation(&order[range.clone()]);
                order[range.clone()].rotate_left(rotation);
                let len = range.len();
                order.copy_within(range, end);
                coupled.push(end..end + len);
                end += len;
                continue;
            }
            let v = order[range.start];
            let pivot = a.lone(v.index());
            if pivot == T::zero() {
                dead.push(v);
            } else {
                lone.push(v);
                pivots.push(pivot);
            }
        }
        let numbering = 0..order.len();
        order.truncate(end);
        order.extend(lone.into_iter().chain(dead));
        let components = Components {
            numbering,
            vertices: order,
            coupled,
            pivots,
        };
        Ok((Cuts { a, position }, components))
    }

    /// The reduction `block` makes of each coupled component under the budget its task leaves it, the assembly freed.
    pub(super) fn reduced<T: Real, B: Send>(
        self,
        threads: Threads,
        block: impl Fn(&[I], <A as PrincipalBlocks<T, I>>::Block, Threads) -> Result<B, BuildError>
            + Sync,
    ) -> Result<Summands<B, I, T>, BuildError>
    where
        A: PrincipalBlocks<T, I> + LonePivot<T> + Sync,
    {
        let (cuts, components) = self.arrange()?;
        components.cut(|components| cuts.extract(components, threads, block))
    }
}

impl<A, I: Index> Cuts<A, I> {
    /// Each coupled component as `block` makes it, in component order, the assembly freed; a connected input moves its assembly in whole.
    pub(super) fn extract<'p, T: Real, B: Send>(
        self,
        components: &'p Components<I, T>,
        threads: Threads,
        block: impl Fn(&'p [I], <A as PrincipalBlocks<T, I>>::Block, Threads) -> Result<B, BuildError>
            + Sync,
    ) -> Result<Cut<'p, I, T, B>, BuildError>
    where
        A: PrincipalBlocks<T, I> + Sync,
    {
        let Self { a, position } = self;
        let count = components.coupled.len();
        let blocks = match (count, position) {
            (0, _) => Vec::new(),
            (_, None) => {
                let whole = components.component(0);
                vec![block(whole, a.whole(whole), threads)?]
            }
            (_, Some(position)) => {
                let cut = |k: usize, threads| {
                    let vertices = components.component(k);
                    block(vertices, a.component(vertices, &position), threads)
                };
                // About `2^PARALLEL_BITS` vertices a task: waking the pool for less costs more than it saves.
                let work = components.coupled.iter().map(|c| c.len() as u64).sum();
                let grain = crate::per_task(count, work, PARALLEL_BITS);
                let blocks: Vec<_> = match threads.per_task(count, grain) {
                    Some(per) => {
                        let nested = threads.within_task(count.div_ceil(per));
                        (0..count)
                            .into_par_iter()
                            .with_min_len(per)
                            .map(|k| task(|| cut(k, nested)))
                            .collect()
                    }
                    None => (0..count).map(|k| cut(k, threads)).collect(),
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
        };
        Ok(Cut {
            blocks,
            lent: PhantomData,
        })
    }
}

/// Each coupled component's block, one per component in component order, cut from the components it borrows.
pub(super) struct Cut<'p, I, T, B> {
    blocks: Vec<B>,
    lent: PhantomData<&'p Components<I, T>>,
}

impl<'p, I, T, B> Cut<'p, I, T, B> {
    pub(super) fn blocks(&self) -> &[B] {
        &self.blocks
    }

    /// Each block as `f` makes it from its component's index and itself, in turn, up to the first error.
    pub(super) fn try_map<C>(
        self,
        mut f: impl FnMut(usize, B) -> Result<C, BuildError>,
    ) -> Result<Cut<'p, I, T, C>, BuildError> {
        // Doubling, the folded blocks held up to 2x their size beside the live ones (+12% peak at 99k blocks).
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (k, block) in self.blocks.into_iter().enumerate() {
            blocks.push(f(k, block)?);
        }
        Ok(Cut {
            blocks,
            lent: PhantomData,
        })
    }

    /// As [`try_map`](Self::try_map), into `blocks`' own buffer where `C` fits it.
    pub(super) fn try_map_in_place<C>(
        self,
        mut f: impl FnMut(usize, B) -> Result<C, BuildError>,
    ) -> Result<Cut<'p, I, T, C>, BuildError> {
        let blocks = (self.blocks.into_iter().enumerate())
            .map(|(k, block)| f(k, block))
            .collect::<Result<_, _>>()?;
        Ok(Cut {
            blocks,
            lent: PhantomData,
        })
    }
}

const PARALLEL_BITS: u32 = 12;

/// `A`'s vertices by component in the partition's array, components by lowest vertex: each coupled component in local order, its eliminated side first, then the lone vertices, then the dead ones.
#[derive(Clone, Debug)]
pub(crate) struct Components<I, T, C = Range<usize>> {
    pub(crate) vertices: Vec<I>,
    /// Each coupled component, ascending: its range of `vertices` until it is cut, then its block, which gives the range.
    pub(crate) coupled: Vec<C>,
    /// Each lone vertex's surplus `σ`, which is its whole pivot; dead vertices, whose pivot is zero, follow the lone ones.
    pub(crate) pivots: Vec<T>,
    /// The factored matrix's vertices in `A`'s numbering: all of `A`'s, or a complement's kept side.
    pub(crate) numbering: Range<usize>,
}

/// The factored matrix's direct summands: its vertices by component, and each coupled component's block.
pub(crate) type Summands<B, I, T> = Components<I, T, B>;

/// Each block's kept vertices as a range of `vertices`, each component starting where the one before ends.
pub(crate) fn ranges<B: CoupledBlock>(
    coupled: &[B],
) -> impl ExactSizeIterator<Item = Range<usize>> + Clone + '_ {
    let mut end = 0;
    coupled.iter().map(move |block| {
        let (kept, start) = (block.kept(), end);
        end += kept.end;
        start + kept.start..end
    })
}

impl<I: Index, T> Components<I, T> {
    /// Coupled component `k`'s vertices of `A`, in local order.
    pub(crate) fn component(&self, k: usize) -> &[I] {
        &self.vertices[self.coupled[k].clone()]
    }

    /// Each coupled component's block in place of its range, as `cut` makes them from these components.
    pub(super) fn cut<B>(
        self,
        cut: impl for<'p> FnOnce(&'p Self) -> Result<Cut<'p, I, T, B>, BuildError>,
    ) -> Result<Summands<B, I, T>, BuildError> {
        let blocks = cut(&self)?.blocks;
        Ok(self.with_coupled(blocks))
    }
}

impl<I, T, C> Components<I, T, C> {
    fn with_coupled<D>(self, coupled: Vec<D>) -> Components<I, T, D> {
        let Self {
            vertices,
            pivots,
            numbering,
            ..
        } = self;
        Components {
            vertices,
            coupled,
            pivots,
            numbering,
        }
    }
}

impl<I: Index, T, B: CoupledBlock> Summands<B, I, T> {
    /// Each block as `f` makes it from its kept vertices and itself, in parallel within `threads`.
    pub(super) fn map<C: Send>(
        mut self,
        threads: Threads,
        f: impl Fn(&[I], B) -> C + Sync,
    ) -> Summands<C, I, T>
    where
        B: Send,
    {
        let kept: Vec<_> = ranges(&self.coupled).collect();
        let vertices = &self.vertices;
        let blocks = std::mem::take(&mut self.coupled);
        let map = |(block, kept): (B, std::ops::Range<usize>)| f(&vertices[kept], block);
        let coupled = match threads.per_task(blocks.len(), 1) {
            Some(per) => blocks
                .into_par_iter()
                .zip(kept)
                .with_min_len(per)
                .map(|item| task(|| map(item)))
                .collect(),
            None => blocks.into_iter().zip(kept).map(map).collect(),
        };
        self.with_coupled(coupled)
    }

    /// Drops the lone and dead vertices `keep` rejects, a lone one with its pivot.
    pub(crate) fn retain_isolated(&mut self, keep: impl Fn(I) -> bool) {
        let coupled = ranges(&self.coupled).last().map_or(0, |range| range.end);
        let isolated = self.vertices.split_off(coupled);
        let (lone, dead) = isolated.split_at(self.pivots.len());
        for (&v, pivot) in lone.iter().zip(std::mem::take(&mut self.pivots)) {
            if keep(v) {
                self.vertices.push(v);
                self.pivots.push(pivot);
            }
        }
        self.vertices
            .extend(dead.iter().copied().filter(|&v| keep(v)));
    }
}

/// One block `A_c` as the input gave it.
pub(super) struct SymmetricBlock<'p, T, I> {
    /// In `A`'s numbering, in local order.
    pub(super) vertices: &'p [I],
    pub(super) entries: Entries<T, I>,
}

/// A block's entries off its diagonal, with its rows' diagonal and `Σ_j |a_ij|`.
pub(super) struct Entries<T, I> {
    pub(super) off_diagonal: OffDiagonal<T, I>,
    pub(super) rows: RowSums<T>,
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
pub(super) struct RowSums<T> {
    pub(super) diagonal: Vec<T>,
    pub(super) abs: Vec<T>,
}

impl<T: Real, J: PrimInt> Canonical<'_, T, J> {
    /// Upper rows in local order, columns remapped through `position`; a zero couples nothing, so it may name another component.
    fn rows<I: Index>(&self, vertices: &[I], position: impl Fn(usize) -> I) -> Entries<T, I> {
        let mut diagonal = Vec::with_capacity(vertices.len());
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|v| self.above(v.index()).len()).sum();
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
        let rows = RowSums { diagonal, abs };
        Entries { off_diagonal, rows }
    }
}

impl<T: Real, J: PrimInt, I: Index> PrincipalBlocks<T, I> for Canonical<'_, T, J> {
    type Block = Entries<T, I>;

    fn rotation(&self, _: &[I]) -> usize {
        0
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::CsrRef;

    /// The size screen reads only the partition, before `A`: a component of `1/ε` vertices after a lone vertex 0 is too large at its lowest vertex.
    #[test]
    fn a_component_of_one_over_epsilon_vertices_is_too_large_at_its_lowest_vertex() {
        let n = (1u32 << 23) + 1;
        let csr = CsrRef::new(1, &[0u32, 1], &[0], &[1.0f32]).expect("a valid input");
        let partitioned = Partitioned {
            a: Canonical::of(csr.0),
            partition: Partition {
                order: (0..n).collect(),
                ends: vec![1, n],
                position: None,
            },
        };
        let error = partitioned.arrange::<f32>().err();
        assert_eq!(error, Some(BuildError::TooLarge { vertex: 1 }));
    }
}
