//! CSR to elimination graph: canonicalize, pair each off-diagonal with its mirror,
//! and close the row deficits with a Gremban ground vertex. One module per phase, in
//! the order the pipeline runs them.

mod canonical;
mod sets;
mod validate;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::types::Real;
use crate::{CsrRef, Error};
use canonical::Canonical;
use validate::{validate, Grounding, Ingested};

/// The ingested input, kept whole so each block takes only what its backend needs. A
/// block routed to the dense arm reads its triangle straight from these arrays; only
/// one that will actually be eliminated on gets an adjacency list built for it.
pub(crate) struct Ingestion<'a, T> {
    canonical: Canonical<'a, T>,
    /// Emptied by [`take_block_diagonal`](Ingestion::take_block_diagonal) for the whole
    /// graph, which is why `n` is its own field rather than this vector's length.
    diagonal: Vec<T>,
    n: usize,
    grounding: Grounding<T>,
    layout: Option<BlockLayout>,
}

impl<'a, T: Real> Ingestion<'a, T> {
    pub(crate) fn of(csr: CsrRef<'a, T, u32>) -> Result<Self, Error> {
        let canonical = Canonical::of(csr)?;
        let Ingested {
            diagonal,
            grounding,
            layout,
        } = validate(&canonical)?;
        Ok(Self {
            canonical,
            n: diagonal.len(),
            diagonal,
            grounding,
            layout,
        })
    }

    /// Vertices the factorization covers, the ground one included.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Whether this block holds the Gremban ground vertex, which decides how it is
    /// anchored. Ingestion appends that vertex above every real one and a block lists
    /// its vertices ascending, so it can only ever be a block's last.
    pub(crate) fn carries_ground(&self, block: &BlockVertices<'_>) -> bool {
        match &self.grounding {
            Grounding::Floating => false,
            // The bound checked in `ground` is what makes the cast lossless.
            Grounding::Grounded { surpluses, .. } => block.last() == surpluses.len() as u32,
        }
    }

    /// `None` when the graph is connected, which is the one block case. Taken rather
    /// than borrowed so the caller can walk the blocks while asking for each one.
    pub(crate) fn take_layout(&mut self) -> Option<BlockLayout> {
        self.layout.take()
    }

    /// The diagonal entry the block's row `local` carries.
    pub(crate) fn block_diagonal(&self, block: &BlockVertices<'_>, local: usize) -> T {
        self.diagonal[block.global(local)]
    }

    /// Hands `entry` each strictly-upper off-diagonal of the block's row `local`, in
    /// the block's own numbering.
    ///
    /// The upper mirror is the authoritative one: `validate` tolerates mirrors that
    /// differ by a few ulps, and the approximate route symmetrizes on this same value,
    /// so reading the stored lower one instead would make the two routes disagree about
    /// a matrix they were both handed.
    ///
    /// A block's last vertex has no CSR row when it is the ground vertex, and no upper
    /// entries when it is not, so it yields nothing either way rather than indexing past
    /// the row pointers.
    pub(crate) fn upper_row(
        &self,
        block: &BlockVertices<'_>,
        local: usize,
        mut entry: impl FnMut(usize, T),
    ) {
        let (row_ptrs, col_indices, values) = self.canonical.arrays();
        let row = block.global(local);
        if row + 1 >= row_ptrs.len() {
            return;
        }
        let (from, to) = (row_ptrs[row] as usize, row_ptrs[row + 1] as usize);
        for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
            if col as usize > row && value != T::zero() {
                entry(block.local(col as usize), value);
            }
        }
    }

    /// The block's diagonal in local order. The whole graph is moved out rather than
    /// copied: it is one block, so nothing reads it again.
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
        let (row_ptrs, col_indices, values) = self.canonical.arrays();
        let rows = row_ptrs.len() - 1;
        let n = block.len();
        // Every grounded row is unioned through the ground vertex, so they all share one
        // block and its degree is the whole count wherever it lands.
        let ground_degree = match self.grounding {
            Grounding::Floating => 0,
            Grounding::Grounded { degree, .. } => degree,
        };

        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(n);
        for local in 0..n {
            let global = block.global(local);
            let degree = if global < rows {
                (row_ptrs[global + 1] - row_ptrs[global]) as usize
            } else {
                ground_degree
            };
            adj.push(Vec::with_capacity(degree));
        }

        // The arm is resolved once around the edge loop, not tested inside it: measured
        // on this loop, the per-edge discriminant test kept `local_of`'s pointer and
        // length spilled to the stack and reloaded for every edge added.
        match block {
            BlockVertices::Whole(_) => {
                for local in 0..n {
                    if local >= rows {
                        continue;
                    }
                    let (from, to) = (row_ptrs[local] as usize, row_ptrs[local + 1] as usize);
                    for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
                        if col as usize > local && value != T::zero() {
                            add_edge_pair(&mut adj, local, col as usize, -value);
                        }
                    }
                }
            }
            BlockVertices::Part { vertices, local_of } => {
                // Columns are already bounded by the matrix dimension, so narrowing the
                // slice lets the bound live in a register rather than being reloaded.
                let local_of = &local_of[..rows];
                for (local, &global) in vertices.iter().enumerate() {
                    let global = global as usize;
                    if global >= rows {
                        continue;
                    }
                    let (from, to) = (row_ptrs[global] as usize, row_ptrs[global + 1] as usize);
                    for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
                        if col as usize > global && value != T::zero() {
                            add_edge_pair(&mut adj, local, local_of[col as usize] as usize, -value);
                        }
                    }
                }
            }
        }

        if let Grounding::Grounded { surpluses, .. } = &self.grounding {
            if self.carries_ground(block) {
                let ground = n - 1;
                for (row, &surplus) in surpluses.iter().enumerate() {
                    // The clamp in `ground` left every surplus non-negative.
                    if surplus > T::zero() {
                        add_edge_pair(&mut adj, block.local(row), ground, surplus);
                    }
                }
            }
        }

        AdjListGraph::from_adjacency(adj)
    }
}

#[cfg(test)]
mod tests;
