//! [`Sddm`] to elimination graph.

mod sets;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::types::Real;
use crate::{Error, Grounded, Laplacian, Sddm};
use sets::DisjointSets;

/// Adds every edge weight to both endpoints' `diagonal` entries, counts both triangles'
/// degrees, and unions each edge's endpoints.
fn walk<T: Real>(
    laplacian: &Laplacian<T>,
    mut diagonal: Vec<T>,
) -> Result<(Vec<T>, Vec<u32>, DisjointSets), Error> {
    let m = laplacian.n();
    let mut sets = DisjointSets::new(m);
    let mut degrees = vec![0u32; m];
    for row in 0..m {
        let (neighbors, weights) = laplacian.row(row);
        degrees[row] += neighbors.len() as u32;
        let mut root = sets.find(row as u32);
        for (&col, &weight) in neighbors.iter().zip(weights) {
            diagonal[row] = diagonal[row] + weight;
            diagonal[col as usize] = diagonal[col as usize] + weight;
            degrees[col as usize] += 1;
            root = sets.union_resolved(root, col);
        }
    }
    if let Some(row) = diagonal.iter().position(|d| !d.is_finite()) {
        return Err(Error::NonFiniteRow { row });
    }
    Ok((diagonal, degrees, sets))
}

/// Kept whole so a block routed to the dense arm never gets an adjacency list built.
pub(crate) struct Ingestion<T> {
    sddm: Sddm<T>,
    /// The ground vertex's last; `take_block_diagonal` empties it.
    diagonal: Vec<T>,
    /// Both triangles' count per vertex, so adjacency lists never regrow.
    degrees: Vec<u32>,
    layout: Option<BlockLayout>,
}

impl<T: Real> Ingestion<T> {
    /// One walk derives the diagonal and unions the components.
    pub(crate) fn of(sddm: Sddm<T>) -> Result<Self, Error> {
        let start = match &sddm {
            Sddm::Laplacian(laplacian) => vec![T::zero(); laplacian.n()],
            Sddm::Grounded(grounded) => grounded.surplus().to_vec(),
        };
        let (mut diagonal, mut degrees, mut sets) = walk(sddm.laplacian(), start)?;
        if let Sddm::Grounded(grounded) = &sddm {
            let surplus = grounded.surplus();
            diagonal.push(surplus.iter().fold(T::zero(), |sum, &s| sum + s));
            // Absent from the input, so the rows it closes are unioned through it here —
            // connectivity read from the edges alone would hand each component back separately.
            let mut root = sets.push();
            let mut degree = 0u32;
            for (row, &s) in surplus.iter().enumerate() {
                if s > T::zero() {
                    root = sets.union_resolved(root, row as u32);
                    degree += 1;
                }
            }
            degrees.push(degree);
        }
        Ok(Self {
            layout: sets.layout(),
            sddm,
            diagonal,
            degrees,
        })
    }

    fn grounded(&self) -> Option<&Grounded<T>> {
        match &self.sddm {
            Sddm::Laplacian(_) => None,
            Sddm::Grounded(grounded) => Some(grounded),
        }
    }

    /// Vertices the factorization covers, the ground one included.
    pub(crate) fn n(&self) -> usize {
        self.sddm.n() + usize::from(self.grounded().is_some())
    }

    /// The ground vertex outranks every real one, so it can only be a block's last.
    pub(crate) fn carries_ground(&self, block: &BlockVertices<'_>) -> bool {
        self.grounded()
            .is_some_and(|grounded| block.last() == grounded.n() as u32)
    }

    /// `None` when connected. Taken so the caller can walk blocks while asking for each.
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

    /// The whole graph is moved out: it is one block, so nothing reads it again.
    pub(crate) fn take_block_diagonal(&mut self, block: &BlockVertices<'_>) -> Vec<T> {
        match block {
            BlockVertices::Whole(_) => core::mem::take(&mut self.diagonal),
            BlockVertices::Part { vertices, .. } => vertices
                .iter()
                .map(|&vertex| self.diagonal[vertex as usize])
                .collect(),
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
        let mut adj: Vec<Vec<Edge<T, C>>> = (0..n)
            .map(|local| Vec::with_capacity(self.degrees[block.global(local)] as usize))
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

        if let Some(grounded) = self.grounded().filter(|_| self.carries_ground(block)) {
            for (row, &surplus) in grounded.surplus().iter().enumerate() {
                if surplus > T::zero() {
                    add_edge_pair(&mut adj, block.local(row), n - 1, surplus);
                }
            }
        }

        AdjListGraph::from_adjacency(adj)
    }
}

#[cfg(test)]
mod tests;
