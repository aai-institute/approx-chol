//! [`Sddm`] to elimination graph.

mod sets;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::types::Real;
use crate::{Laplacian, Sddm};
use sets::DisjointSets;

/// Counts both triangles' degrees and unions each edge's endpoints in one walk.
fn walk<T>(laplacian: &Laplacian<T>) -> (Vec<u32>, DisjointSets) {
    let m = laplacian.n();
    let mut sets = DisjointSets::new(m);
    let mut degrees = vec![0u32; m];
    for row in 0..m {
        let (neighbors, _) = laplacian.row(row);
        degrees[row] += neighbors.len() as u32;
        let mut root = sets.find(row as u32);
        for &col in neighbors {
            degrees[col as usize] += 1;
            root = sets.union_resolved(root, col);
        }
    }
    (degrees, sets)
}

/// Kept whole so a block routed to the dense arm never gets an adjacency list built.
pub(crate) struct Ingestion<T> {
    sddm: Sddm<T>,
    /// Both triangles' count per vertex, so adjacency lists never regrow.
    degrees: Vec<u32>,
    layout: Option<BlockLayout>,
}

impl<T: Real> Ingestion<T> {
    pub(crate) fn of(sddm: Sddm<T>) -> Self {
        let (degrees, mut sets) = walk(sddm.laplacian());
        Self {
            layout: sets.layout(),
            sddm,
            degrees,
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.sddm.n()
    }

    fn surplus_of(&self) -> Option<&[T]> {
        match &self.sddm {
            Sddm::Laplacian(_) => None,
            Sddm::Grounded(grounded) => Some(grounded.surplus()),
        }
    }

    /// `None` when connected. Taken so the caller can walk blocks while asking for each.
    pub(crate) fn take_layout(&mut self) -> Option<BlockLayout> {
        self.layout.take()
    }

    /// Whether any of the block's vertices carries surplus; a [`Grounded`](crate::Grounded)
    /// has some, so the connected case needs no scan.
    pub(crate) fn is_grounded(&self, block: &BlockVertices<'_>) -> bool {
        match (self.surplus_of(), block) {
            (None, _) => false,
            (Some(_), BlockVertices::Whole(_)) => true,
            (Some(surplus), BlockVertices::Part { vertices, .. }) => vertices
                .iter()
                .any(|&vertex| surplus[vertex as usize] > T::zero()),
        }
    }

    /// The surplus on the block row `local`'s diagonal.
    pub(crate) fn surplus(&self, block: &BlockVertices<'_>, local: usize) -> T {
        self.surplus_of()
            .map_or_else(T::zero, |surplus| surplus[block.global(local)])
    }

    /// The edge weights above the block row's diagonal, by local column.
    pub(crate) fn upper_row(
        &self,
        block: &BlockVertices<'_>,
        local: usize,
        mut entry: impl FnMut(usize, T),
    ) {
        let (neighbors, weights) = self.sddm.laplacian().row(block.global(local));
        for (&col, &weight) in neighbors.iter().zip(weights) {
            entry(block.local(col as usize), weight);
        }
    }

    /// Builds the block's adjacency, which only the approximate arm needs. A grounded
    /// block gets its ground appended as the last vertex.
    pub(crate) fn block_graph<C: EdgeCount>(
        &self,
        block: &BlockVertices<'_>,
        grounded: bool,
    ) -> AdjListGraph<C, T> {
        let laplacian = self.sddm.laplacian();
        let k = block.len();
        let surplus = self.surplus_of().filter(|_| grounded);
        let grounds = |global: usize| surplus.is_some_and(|surplus| surplus[global] > T::zero());
        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(k + usize::from(grounded));
        adj.extend((0..k).map(|local| {
            let global = block.global(local);
            Vec::with_capacity(self.degrees[global] as usize + usize::from(grounds(global)))
        }));

        // Measured: an in-loop discriminant test spills `local_of` and reloads per edge.
        match block {
            BlockVertices::Whole(_) => {
                for local in 0..k {
                    let (neighbors, weights) = laplacian.row(local);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, col as usize, weight);
                    }
                }
            }
            BlockVertices::Part { vertices, local_of } => {
                // Narrowed so the bound lives in a register.
                let local_of = &local_of[..laplacian.n()];
                for (local, &global) in vertices.iter().enumerate() {
                    let (neighbors, weights) = laplacian.row(global as usize);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, local_of[col as usize] as usize, weight);
                    }
                }
            }
        }

        if let Some(surplus) = surplus {
            let degree = (0..k).filter(|&local| grounds(block.global(local))).count();
            adj.push(Vec::with_capacity(degree));
            for local in 0..k {
                let s = surplus[block.global(local)];
                if s > T::zero() {
                    add_edge_pair(&mut adj, local, k, s);
                }
            }
        }

        AdjListGraph::from_adjacency(adj)
    }
}

#[cfg(test)]
mod tests;
