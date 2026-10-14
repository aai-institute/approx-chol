use num_traits::PrimInt;

mod assembly;
mod partition;

use super::block::ReduceError;
use crate::input::{Canonical, CsrError, UpperCanonical};
use crate::storage::{Rows, UpperAdjacency};
use crate::Real;
pub(super) use assembly::{
    assemble_bipartite, assemble_bipartite_gram, assemble_general, assemble_signed,
    BipartiteComponent,
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
    pub(super) fn extract<T: Real, B>(
        self,
        mut block: impl FnMut(<A as PrincipalBlocks<T>>::Block) -> Result<Option<B>, ReduceError>,
    ) -> Result<(Partition, Vec<B>, Vec<u32>), ReduceError>
    where
        A: PrincipalBlocks<T>,
    {
        let Self { a, partition } = self;
        // Past `1/ε` vertices, floor weights in series fall below `MIN_POSITIVE`; a dead one is alone.
        let too_long =
            |vertices: &[u32]| T::from(vertices.len()).is_none_or(|n| n * T::epsilon() >= T::one());
        if let Some(vertices) = partition.components().find(|&vertices| too_long(vertices)) {
            let vertex = vertices[0];
            return Err(ReduceError::Unscalable { vertex });
        }
        let (mut live, mut dead) = (Vec::new(), Vec::new());
        let mut sort = |vertices: &[u32], cut| -> Result<(), ReduceError> {
            match block(cut)? {
                Some(b) => live.push(b),
                None => dead.extend_from_slice(vertices),
            }
            Ok(())
        };
        if partition.is_whole() {
            let vertices = partition.components().next().expect("one component");
            sort(vertices, a.whole(vertices))?;
        } else {
            for vertices in partition.components() {
                sort(vertices, a.component(vertices, partition.position()))?;
            }
            // Kept past extraction, an owned assembly raised the Gram peak 11–22%.
            drop(a);
        }
        Ok((partition, live, dead))
    }
}

/// One component's vertices in local order, strict upper triangle and diagonal.
pub(super) type Cells<T> = (Box<[u32]>, UpperAdjacency<T>, Vec<T>);

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
        (vertices.into(), UpperAdjacency::new(rows), diagonal)
    }
}
