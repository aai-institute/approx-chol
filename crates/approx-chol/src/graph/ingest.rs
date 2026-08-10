//! CSR to elimination graph, one module per phase in pipeline order.

mod canonical;
mod sets;
mod validate;

use super::adjacency::{AdjListGraph, EdgeBuffer, SequentialFill};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::types::Real;
use crate::{CsrRef, Error};
use canonical::Canonical;
use validate::{validate, Grounding, Ingested};

/// Kept whole so a block routed to the dense arm never gets an adjacency list built.
pub(crate) struct Ingestion<'a, T> {
    canonical: Canonical<'a, T>,
    /// `take_block_diagonal` empties this, so `n` cannot be its length.
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

    /// The ground vertex outranks every real one, so it can only be a block's last.
    pub(crate) fn carries_ground(&self, block: &BlockVertices<'_>) -> bool {
        match &self.grounding {
            Grounding::Floating => false,
            Grounding::Grounded { surpluses, .. } => block.last() == surpluses.len() as u32,
        }
    }

    /// `None` when connected. Taken so the caller can walk blocks while asking for each.
    pub(crate) fn take_layout(&mut self) -> Option<BlockLayout> {
        self.layout.take()
    }

    /// The diagonal entry the block's row `local` carries.
    pub(crate) fn block_diagonal(&self, block: &BlockVertices<'_>, local: usize) -> T {
        self.diagonal[block.global(local)]
    }

    /// Upper, not the stored lower: mirrors may differ by ulps and the approximate
    /// route symmetrizes on this one.
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
        // Empty unless this block carries the ground vertex, which is where its edges are.
        let surpluses = match &self.grounding {
            Grounding::Grounded { surpluses, .. } if self.carries_ground(block) => &surpluses[..],
            _ => &[][..],
        };
        // Measured: an in-loop discriminant test spills `local_of` and reloads per edge.
        let edges = match block {
            BlockVertices::Whole(n) => self.fill_block(Contiguous(*n), surpluses),
            BlockVertices::Part { vertices, local_of } => self.fill_block(
                Relabelled {
                    vertices,
                    // Narrowed so the bound lives in a register.
                    local_of: &local_of[..self.canonical.arrays().0.len() - 1],
                },
                surpluses,
            ),
        };
        AdjListGraph::from_edges(edges)
    }

    /// Vertex order, edges ascending: what [`SequentialFill`] needs. The lower half takes
    /// its weight from the mirror written for it, which `validate` pins only within 8 ulps.
    fn fill_block<C: EdgeCount, R: Relabel>(
        &self,
        relabel: R,
        surpluses: &[T],
    ) -> EdgeBuffer<T, C> {
        let (row_ptrs, col_indices, values) = self.canonical.arrays();
        let rows = row_ptrs.len() - 1;
        let n = relabel.vertices();
        // Every grounded row shares one block, so its degree is the whole count.
        let ground_degree = match self.grounding {
            Grounding::Floating => 0,
            Grounding::Grounded { degree, .. } => degree,
        };
        let mut fill = SequentialFill::new(n, relabel.entries(row_ptrs) + ground_degree);

        for local in 0..n {
            let global = relabel.global(local);
            // The one vertex the CSR does not carry, and every edge to it is written.
            if global >= rows {
                fill.open(ground_degree);
                for (row, &surplus) in surpluses.iter().enumerate() {
                    if surplus > T::zero() {
                        fill.lower(relabel.local(row as u32));
                    }
                }
                continue;
            }

            let (from, to) = (row_ptrs[global] as usize, row_ptrs[global + 1] as usize);
            fill.open(to - from);
            let mut at = from;
            while at < to && (col_indices[at] as usize) < global {
                fill.lower(relabel.local(col_indices[at]));
                at += 1;
            }
            while at < to {
                let col = col_indices[at] as usize;
                if col > global && values[at] != T::zero() {
                    fill.upper(relabel.local(col_indices[at]), -values[at]);
                }
                at += 1;
            }
            // The clamp in `ground` left every surplus non-negative.
            if surpluses.get(global).is_some_and(|&s| s > T::zero()) {
                fill.upper(n - 1, surpluses[global]);
            }
        }
        fill.finish()
    }
}

/// Monomorphized per block shape, so the block's own discriminant is tested once for
/// the whole fill rather than once per edge.
trait Relabel {
    fn vertices(&self) -> usize;
    fn global(&self, local: usize) -> usize;
    fn local(&self, global: u32) -> usize;
    /// How many stored entries the block's rows cover, which sizes the buffer.
    fn entries(&self, row_ptrs: &[u32]) -> usize;
}

/// The connected case, which never materializes `0..n`.
struct Contiguous(usize);

impl Relabel for Contiguous {
    fn vertices(&self) -> usize {
        self.0
    }

    #[inline]
    fn global(&self, local: usize) -> usize {
        local
    }

    #[inline]
    fn local(&self, global: u32) -> usize {
        global as usize
    }

    fn entries(&self, row_ptrs: &[u32]) -> usize {
        row_ptrs[row_ptrs.len() - 1] as usize
    }
}

struct Relabelled<'v> {
    vertices: &'v [u32],
    local_of: &'v [u32],
}

impl Relabel for Relabelled<'_> {
    fn vertices(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    fn global(&self, local: usize) -> usize {
        self.vertices[local] as usize
    }

    #[inline]
    fn local(&self, global: u32) -> usize {
        self.local_of[global as usize] as usize
    }

    fn entries(&self, row_ptrs: &[u32]) -> usize {
        self.vertices
            .iter()
            .filter(|&&global| (global as usize) < row_ptrs.len() - 1)
            .map(|&global| (row_ptrs[global as usize + 1] - row_ptrs[global as usize]) as usize)
            .sum()
    }
}

#[cfg(test)]
mod tests;
