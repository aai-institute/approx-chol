//! [`Sddm`] to elimination graph.

mod sets;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::sddm::{Grounded, Sddm};
use crate::types::Real;
use sets::DisjointSets;

/// Kept whole so a block routed to the dense arm never gets an adjacency list built.
pub(crate) struct Ingestion<T> {
    sddm: Sddm<T>,
    /// One per real vertex, and the ground vertex's last when grounded.
    diagonal: Vec<T>,
    /// Each vertex's edge count, its ground edge included.
    degrees: Vec<u32>,
    layout: Option<BlockLayout>,
}

impl<T: Real> Ingestion<T> {
    /// One walk sums the diagonal, counts edges and unions the components.
    pub(crate) fn of(sddm: Sddm<T>) -> Self {
        let m = sddm.n();
        let mut sets = DisjointSets::new(m);
        // Room for the ground's count.
        let mut degrees = Vec::with_capacity(m + 1);
        degrees.resize(m, 0u32);
        // The row whose root is resolved, and that root.
        let mut resolved = (u32::MAX, 0u32);
        let diagonal = sddm.diagonal(|row, col| {
            let row = row as u32;
            degrees[row as usize] += 1;
            degrees[col as usize] += 1;
            if resolved.0 != row {
                resolved = (row, sets.find(row));
            }
            resolved.1 = sets.union_resolved(resolved.1, col);
        });
        if let Sddm::Grounded(grounded) = &sddm {
            // The ground is absent from the input, so the rows it closes are unioned through it here.
            let mut root = sets.push();
            let mut degree = 0u32;
            for (row, &s) in grounded.surplus().iter().enumerate() {
                if s > T::zero() {
                    root = sets.union_resolved(root, row as u32);
                    degrees[row] += 1;
                    degree += 1;
                }
            }
            degrees.push(degree);
        }
        debug_assert!(diagonal.iter().all(|d| d.is_finite()));
        Self {
            layout: sets.layout(),
            sddm,
            diagonal,
            degrees,
        }
    }

    /// The ground vertex outranks every real one, so it can only be a block's last.
    fn ground_of(&self, block: &BlockVertices<'_>) -> Option<&Grounded<T>> {
        match &self.sddm {
            Sddm::Grounded(grounded) if block.last() == self.sddm.n() as u32 => Some(grounded),
            _ => None,
        }
    }

    /// Vertices the factorization covers, the ground one included.
    pub(crate) fn n(&self) -> usize {
        self.diagonal.len()
    }

    pub(crate) fn carries_ground(&self, block: &BlockVertices<'_>) -> bool {
        self.ground_of(block).is_some()
    }

    /// `None` when connected. Taken because its order becomes the factor's permutation.
    pub(crate) fn take_layout(&mut self) -> Option<BlockLayout> {
        self.layout.take()
    }

    /// The diagonal entry the block's row `local` carries.
    pub(crate) fn block_diagonal(&self, block: &BlockVertices<'_>, local: usize) -> T {
        self.diagonal[block.global(local)]
    }

    /// The off-diagonal entries `-w` above the block row's diagonal.
    pub(crate) fn upper_row(
        &self,
        block: &BlockVertices<'_>,
        local: usize,
        mut entry: impl FnMut(usize, T),
    ) {
        let row = block.global(local);
        let laplacian = self.sddm.laplacian();
        if row >= laplacian.n() {
            return;
        }
        let (neighbors, weights) = laplacian.row(row);
        for (&col, &weight) in neighbors.iter().zip(weights) {
            entry(block.local(col as usize), -weight);
        }
    }

    /// Builds the block's adjacency, which only the approximate arm needs.
    pub(crate) fn block_graph<C: EdgeCount>(
        &self,
        block: &BlockVertices<'_>,
    ) -> AdjListGraph<C, T> {
        let laplacian = self.sddm.laplacian();
        let rows = laplacian.n();
        let n = block.len();
        // One slot of slack takes the first fill edge: exact capacity measured +3% on degree-4 grids.
        let mut adj: Vec<Vec<Edge<T, C>>> = (0..n)
            .map(|local| Vec::with_capacity(self.degrees[block.global(local)] as usize + 1))
            .collect();

        // Measured: an in-loop discriminant test spills `local_of` and reloads per edge.
        match block {
            BlockVertices::Whole(_) => {
                for local in 0..n.min(rows) {
                    let (neighbors, weights) = laplacian.row(local);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, col as usize, weight);
                    }
                }
            }
            BlockVertices::Part { vertices, local_of } => {
                // Narrowed so the bound lives in a register.
                let local_of = &local_of[..rows];
                for (local, &global) in vertices.iter().enumerate() {
                    let global = global as usize;
                    if global >= rows {
                        continue;
                    }
                    let (neighbors, weights) = laplacian.row(global);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, local_of[col as usize] as usize, weight);
                    }
                }
            }
        }

        if let Some(grounded) = self.ground_of(block) {
            let ground = n - 1;
            for (row, &surplus) in grounded.surplus().iter().enumerate() {
                if surplus > T::zero() {
                    add_edge_pair(&mut adj, block.local(row), ground, surplus);
                }
            }
        }

        AdjListGraph::from_adjacency(adj)
    }
}

#[cfg(test)]
mod tests;
