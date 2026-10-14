use std::sync::Arc;

use num_traits::PrimInt;
use rayon::prelude::*;

mod assembly;
mod partition;

use super::block::ReduceError;
use crate::input::{Canonical, CsrError, UpperCanonical};
use crate::storage::{OffDiagonal, Rows, UpperAdjacency};
use crate::Real;
pub(super) use assembly::{
    assemble_bipartite, assemble_bipartite_gram, assemble_general, assemble_signed,
};
pub(super) use partition::Partition;
use partition::UnionFind;

/// An assembled `A` and its connected components, found from the same cells.
pub(super) struct Partitioned<A> {
    a: A,
    partition: Partition,
}

/// Unions the pattern in the walk that checks symmetry and range.
pub(super) fn of_csr<T: Real, I: PrimInt>(
    csr: Canonical<'_, T, I>,
) -> Result<Partitioned<Canonical<'_, T, I>>, CsrError> {
    let mut sets = UnionFind::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    let partition = sets.into_partition();
    Ok(Partitioned { a: csr, partition })
}

/// Unions the pattern in the walk that checks range: upper storage has no mirror to check.
pub(super) fn of_upper<T: Real, I: PrimInt>(
    csr: UpperCanonical<'_, T, I>,
) -> Result<Partitioned<Canonical<'_, T, I>>, CsrError> {
    let mut sets = UnionFind::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    let partition = sets.into_partition();
    Ok(Partitioned {
        a: csr.into_canonical(),
        partition,
    })
}

/// An assembled `A` cut into its components' blocks.
pub(super) trait PrincipalBlocks<T> {
    type Block;

    /// The whole input as one component, moved where it is owned: its vertices are their own positions.
    fn whole(self, vertices: &[u32]) -> Self::Block;

    /// One component, its columns remapped through `position`.
    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block;
}

impl<A> Partitioned<A> {
    /// The partition, each live component's block and the dead vertices, the assembly freed; a connected input moves its assembly in whole.
    pub(super) fn extract<T: Real, B: Send>(
        self,
        block: impl Fn(<A as PrincipalBlocks<T>>::Block) -> Result<Option<B>, ReduceError> + Sync,
    ) -> Result<(Partition, Vec<B>, Vec<u32>), ReduceError>
    where
        A: PrincipalBlocks<T> + Sync,
    {
        let Self { a, partition } = self;
        // Past `1/ε` vertices, floor weights in series fall below `MIN_POSITIVE`; a dead one is alone.
        let too_long =
            |vertices: &[u32]| T::from(vertices.len()).is_none_or(|n| n * T::epsilon() >= T::one());
        if let Some(vertices) = partition.components().find(|&vertices| too_long(vertices)) {
            let vertex = vertices[0];
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
        let components: Vec<&[u32]> = partition.components().collect();
        let cut = |vertices: &&[u32]| block(a.component(vertices, partition.position()));
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

/// One component's vertices in local order, off-diagonal and diagonal.
pub(super) type Cells<T> = (Box<[u32]>, OffDiagonal<T>, Vec<T>);

impl<T: Real, I: PrimInt> PrincipalBlocks<T> for Canonical<'_, T, I> {
    type Block = Cells<T>;

    fn whole(self, vertices: &[u32]) -> Self::Block {
        self.component(vertices, vertices)
    }

    /// Upper rows in local order; a zero couples nothing, so it may name another component.
    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let mut diagonal = Vec::with_capacity(vertices.len());
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|&v| self.upper_len(v as usize)).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
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
        indices.shrink_to_fit();
        values.shrink_to_fit();
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        let off_diagonal = OffDiagonal::General(UpperAdjacency::new(rows));
        (vertices.into(), off_diagonal, diagonal)
    }
}

/// A factor-width-2 `A`, assembled: general, or bipartite as `C`.
pub(super) struct Assembled<T, C = UpperAdjacency<T>> {
    adjacency: C,
    diagonal: Vec<T>,
}

impl<T: Real> PrincipalBlocks<T> for Assembled<T> {
    type Block = Cells<T>;

    fn whole(self, vertices: &[u32]) -> Self::Block {
        let off_diagonal = OffDiagonal::General(self.adjacency);
        (vertices.into(), off_diagonal, self.diagonal)
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
        let diagonal = vertices
            .iter()
            .map(|&v| self.diagonal[v as usize])
            .collect();
        let off_diagonal = OffDiagonal::General(UpperAdjacency::new(rows));
        (vertices.into(), off_diagonal, diagonal)
    }
}

/// The biadjacency `C`, cancelled cells dropped: rows are `A`'s left vertices.
impl<T: Real> PrincipalBlocks<T> for Rows<T> {
    type Block = (Box<[u32]>, OffDiagonal<T>);

    fn whole(self, vertices: &[u32]) -> Self::Block {
        let (lefts, rights) = vertices.split_at(self.n_rows());
        sides(lefts, rights, self)
    }

    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let n_left = self.n_rows();
        let (lefts, rights) =
            vertices.split_at(vertices.partition_point(|&v| (v as usize) < n_left));
        let mut ptrs = Vec::with_capacity(lefts.len() + 1);
        ptrs.push(0);
        let entries = lefts.iter().map(|&i| self.row_len(i as usize)).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &i in lefts {
            for (j, &x) in self.row(i as usize) {
                indices.push(position[n_left + j] - lefts.len() as u32);
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

impl<T: Real> PrincipalBlocks<T> for Assembled<T, Rows<T>> {
    type Block = Cells<T>;

    fn whole(self, vertices: &[u32]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let diagonal = vertices
            .iter()
            .map(|&v| self.diagonal[v as usize])
            .collect();
        (vertices, off_diagonal, diagonal)
    }

    fn component(&self, vertices: &[u32], position: &[u32]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let diagonal = vertices
            .iter()
            .map(|&v| self.diagonal[v as usize])
            .collect();
        (vertices, off_diagonal, diagonal)
    }
}

/// The larger side eliminated first, `C` transposed when it is the right one; a side alone is one uncoupled vertex.
fn sides<T: Real>(lefts: &[u32], rights: &[u32], c: Rows<T>) -> (Box<[u32]>, OffDiagonal<T>) {
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
