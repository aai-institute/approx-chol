//! Input to elimination graph, one module per phase in pipeline order.

mod canonical;
mod sets;
mod validate;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::types::Real;
use crate::{CsrError, CsrRef, Error, IndexKind, Laplacian, Sddm};
use num_traits::PrimInt;
use sets::DisjointSets;

/// The CSR path alone canonicalizes, checks mirrors, and judges which surplus is noise.
/// Canonical input is read in place, in the caller's own index type.
pub(crate) fn sddm_from_csr<T: Real, I: PrimInt>(csr: CsrRef<'_, T, I>) -> Result<Sddm<T>, Error> {
    if csr.col_indices().len() > u32::MAX as usize {
        return Err(Error::InvalidCsr(CsrError::IndexExceedsIndexType {
            kind: IndexKind::RowPtr,
        }));
    }
    let terms = canonical::terms(csr.row_ptrs());
    if canonical::is_canonical(csr.row_ptrs(), csr.col_indices()) {
        return validate::sddm_of(csr.row_ptrs(), csr.col_indices(), csr.values(), terms);
    }
    // Before rewriting, so the position stays the caller's own.
    if let Some(position) = csr.values().iter().position(|value| !value.is_finite()) {
        return Err(Error::NonFiniteValue { position });
    }
    let narrowed = csr.narrow_indices()?;
    let rewritten = canonical::rewrite(narrowed.with_values(csr.values()))?;
    validate::sddm_of(
        &rewritten.row_ptrs,
        &rewritten.col_indices,
        &rewritten.values,
        terms,
    )
}

/// All rows balancing means a bare Laplacian, with no ground vertex to attach.
enum Grounding<T> {
    Floating,
    Grounded {
        /// Zero where the vertex has no edge to ground.
        surpluses: Vec<T>,
        /// How many of those are positive, which is the ground vertex's degree.
        degree: usize,
    },
}

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
    laplacian: Laplacian<T>,
    /// `take_block_diagonal` empties this, so `n` cannot be its length.
    diagonal: Vec<T>,
    /// Both triangles' count per real vertex, so adjacency lists never regrow.
    degrees: Vec<u32>,
    n: usize,
    grounding: Grounding<T>,
    layout: Option<BlockLayout>,
}

impl<T: Real> Ingestion<T> {
    /// One walk derives the diagonal and unions the components.
    pub(crate) fn of(sddm: Sddm<T>) -> Result<Self, Error> {
        match sddm {
            Sddm::Laplacian(laplacian) => {
                let zeros = vec![T::zero(); laplacian.n()];
                let (diagonal, degrees, mut sets) = walk(&laplacian, zeros)?;
                Ok(Self {
                    laplacian,
                    n: diagonal.len(),
                    diagonal,
                    degrees,
                    grounding: Grounding::Floating,
                    layout: sets.layout(),
                })
            }
            Sddm::Grounded(grounded) => {
                let (laplacian, surplus) = grounded.into_parts();
                let m = laplacian.n();
                if m >= u32::MAX as usize {
                    return Err(Error::InvalidCsr(
                        CsrError::MatrixDimensionExceedsIndexType {
                            n: m.saturating_add(1),
                        },
                    ));
                }
                let (mut diagonal, degrees, mut sets) = walk(&laplacian, surplus.clone())?;
                diagonal.push(surplus.iter().fold(T::zero(), |sum, &s| sum + s));
                // Absent from the input, so the rows it closes are unioned through it here —
                // connectivity read from the edges alone would hand each component back separately.
                let mut root = sets.push();
                let mut degree = 0;
                for (row, &s) in surplus.iter().enumerate() {
                    if s > T::zero() {
                        root = sets.union_resolved(root, row as u32);
                        degree += 1;
                    }
                }
                Ok(Self {
                    laplacian,
                    n: diagonal.len(),
                    diagonal,
                    degrees,
                    grounding: Grounding::Grounded {
                        surpluses: surplus,
                        degree,
                    },
                    layout: sets.layout(),
                })
            }
        }
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

    /// The off-diagonal entries `-w` above the block row's diagonal.
    pub(crate) fn upper_row(
        &self,
        block: &BlockVertices<'_>,
        local: usize,
        mut entry: impl FnMut(usize, T),
    ) {
        let row = block.global(local);
        if row >= self.laplacian.n() {
            return;
        }
        let (neighbors, weights) = self.laplacian.row(row);
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
        let rows = self.laplacian.n();
        let n = block.len();
        // Every grounded row shares one block, so its degree is the whole count.
        let ground_degree = match self.grounding {
            Grounding::Floating => 0,
            Grounding::Grounded { degree, .. } => degree,
        };

        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(n);
        for local in 0..n {
            let global = block.global(local);
            let degree = if global < rows {
                self.degrees[global] as usize
            } else {
                ground_degree
            };
            adj.push(Vec::with_capacity(degree));
        }

        // Measured: an in-loop discriminant test spills `local_of` and reloads per edge.
        match block {
            BlockVertices::Whole(_) => {
                for local in 0..n.min(rows) {
                    let (neighbors, weights) = self.laplacian.row(local);
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
                    let (neighbors, weights) = self.laplacian.row(global);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, local_of[col as usize] as usize, weight);
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
