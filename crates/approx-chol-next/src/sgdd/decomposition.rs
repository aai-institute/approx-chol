use std::borrow::Cow;
use std::sync::Arc;

use num_traits::PrimInt;
use rayon::prelude::*;

mod assembly;
mod partition;

use super::block::{Grounding, ReduceError};
use crate::input::{Canonical, CsrError, UpperCanonical};
use crate::storage::{OffDiagonal, Rows, UpperAdjacency};
use crate::{Index, Real};
pub(super) use assembly::{
    assemble_bipartite, assemble_bipartite_gram, assemble_general, assemble_laplacian,
    assemble_signed,
};
pub(super) use partition::Partition;
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
    csr: UpperCanonical<'_, T, J>,
) -> Result<Partitioned<Canonical<'_, T, J>, I>, CsrError> {
    let mut sets = UnionFind::<I>::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    let partition = sets.into_partition();
    Ok(Partitioned {
        a: csr.into_canonical(),
        partition,
    })
}

/// `I` holds `0..count` below its `MAX`, which stays free as a sentinel.
pub(super) fn indexable<I: Index>(count: usize) -> Result<(), ReduceError> {
    match num_traits::cast::<usize, I>(count) {
        Some(_) => Ok(()),
        None => Err(ReduceError::IndexOverflow { count }),
    }
}

/// An assembled `A` cut into its components' blocks.
pub(super) trait PrincipalBlocks<T, I> {
    type Block;

    /// The whole input as one component, moved where it is owned: its vertices are their own positions.
    fn whole(self, vertices: &[I]) -> Self::Block;

    /// One component, its columns remapped through `position`.
    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block;
}

impl<A, I: Index> Partitioned<A, I> {
    /// The partition, each live component's block and the dead vertices, the assembly freed; a connected input moves its assembly in whole.
    pub(super) fn extract<T: Real, B: Send>(
        self,
        block: impl Fn(<A as PrincipalBlocks<T, I>>::Block) -> Result<Option<B>, ReduceError> + Sync,
    ) -> Result<Components<B, I>, ReduceError>
    where
        A: PrincipalBlocks<T, I> + Sync,
    {
        let Self { a, partition } = self;
        // Past `1/ε` vertices, floor weights in series fall below `MIN_POSITIVE`; a dead one is alone.
        let too_long =
            |vertices: &[I]| T::from(vertices.len()).is_none_or(|n| n * T::epsilon() >= T::one());
        if let Some(vertices) = partition.components().find(|&vertices| too_long(vertices)) {
            let vertex = vertices[0].index();
            return Err(ReduceError::TooLarge { vertex });
        }
        if partition.is_whole() {
            let vertices = partition.components().next().expect("one component");
            let (live, dead) = match block(a.whole(vertices))? {
                Some(b) => (vec![b], Vec::new()),
                None => (Vec::new(), vertices.to_vec()),
            };
            return Ok((partition, live, dead));
        }
        let components: Vec<&[I]> = partition.components().collect();
        let cut = |vertices: &&[I]| block(a.component(vertices, partition.position()));
        // About `PARALLEL_VERTICES` vertices a task: waking the pool for less costs more than it saves.
        let per_task = (PARALLEL_VERTICES * components.len()).div_ceil(partition.n());
        let blocks: Vec<_> = match per_task < components.len() && rayon::current_num_threads() > 1 {
            true => (components.par_iter())
                .with_min_len(per_task)
                .map(cut)
                .collect(),
            false => components.iter().map(cut).collect(),
        };
        // Kept past extraction, an owned assembly raised the Gram peak 11–22%.
        drop(a);
        if let Some(&error) = blocks.iter().find_map(|b| b.as_ref().err()) {
            return Err(error);
        }
        let dead = (components.iter().zip(&blocks))
            .filter(|(_, b)| matches!(b, Ok(None)))
            .flat_map(|(vertices, _)| vertices.iter().copied())
            .collect();
        // In place: a block's result is the block's own size.
        let live = blocks
            .into_iter()
            .filter_map(|b| b.ok().flatten())
            .collect();
        Ok((partition, live, dead))
    }
}

const PARALLEL_VERTICES: usize = 1 << 12;

/// The partition, each live component's block, and the dead vertices.
pub(super) type Components<B, I> = (Partition<I>, Vec<B>, Vec<I>);

/// A general `A` assembled with its diagonal, and its components.
pub(super) type General<T, I> = Partitioned<Assembled<T, UpperAdjacency<T, I>>, I>;

/// A bipartite Gram assembled as `C` and the diagonal, and its components.
pub(super) type BipartiteGram<T, I> = Partitioned<Assembled<T, Rows<T, I>>, I>;

/// One component's vertices in local order, off-diagonal and diagonal.
pub(super) type Cells<T, I> = (Box<[I]>, OffDiagonal<T, I>, Vec<T>);

impl<T: Real, J: PrimInt, I: Index> PrincipalBlocks<T, I> for Canonical<'_, T, J> {
    type Block = Cells<T, I>;

    fn whole(self, vertices: &[I]) -> Self::Block {
        self.component(vertices, vertices)
    }

    /// Upper rows in local order; a zero couples nothing, so it may name another component.
    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let mut diagonal = Vec::with_capacity(vertices.len());
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|&v| self.upper_len(v.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &v in vertices {
            let v = v.index();
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
        indices.shrink_to_fit();
        values.shrink_to_fit();
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        let off_diagonal = OffDiagonal::General(UpperAdjacency::new(rows));
        (vertices.into(), off_diagonal, diagonal)
    }
}

/// A factor-width-2 `A`, assembled: general, or bipartite as `C`.
pub(super) struct Assembled<T, C> {
    adjacency: C,
    diagonal: Vec<T>,
}

/// `A = L(C) + diag(σ)`, dominant by shape: no diagonal is stored, each pivot is `Σ_j |c_ij| + σ_i`.
pub(super) struct Dominant<'a, T: Clone, I> {
    adjacency: UpperAdjacency<T, I>,
    surplus: Option<Cow<'a, [T]>>,
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

impl<T: Real, I: Index> PrincipalBlocks<T, I> for Assembled<T, UpperAdjacency<T, I>> {
    type Block = Cells<T, I>;

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        (vertices, off_diagonal, self.diagonal)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let diagonal = vertices.iter().map(|&v| self.diagonal[v.index()]).collect();
        (vertices, off_diagonal, diagonal)
    }
}

impl<T: Real, I: Index> PrincipalBlocks<T, I> for Dominant<'_, T, I> {
    type Block = (Box<[I]>, OffDiagonal<T, I>, Grounding<T>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let grounding =
            (self.surplus).map_or_else(Grounding::ungrounded, |s| Grounding::of(s.into_owned()));
        (vertices, off_diagonal, grounding)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let at = |s: &Cow<'_, [T]>| Grounding::of(vertices.iter().map(|&v| s[v.index()]).collect());
        let grounding = self.surplus.as_ref().map_or_else(Grounding::ungrounded, at);
        (vertices, off_diagonal, grounding)
    }
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

impl<T: Real, I: Index> PrincipalBlocks<T, I> for Assembled<T, Rows<T, I>> {
    type Block = Cells<T, I>;

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let diagonal = vertices.iter().map(|&v| self.diagonal[v.index()]).collect();
        (vertices, off_diagonal, diagonal)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let diagonal = vertices.iter().map(|&v| self.diagonal[v.index()]).collect();
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
        true => (
            [lefts, rights].concat().into(),
            OffDiagonal::Bipartite(Arc::new(c)),
        ),
        false => {
            let c = c.transpose();
            (
                [rights, lefts].concat().into(),
                OffDiagonal::Bipartite(Arc::new(c)),
            )
        }
    }
}
