//! The [`Sddm`] split into components, each grounded by its own surplus or floating.

mod sets;

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::{BlockLayout, BlockVertices};
use super::multiplicity::EdgeCount;
use crate::sddm::{Laplacian, Sddm};
use crate::types::Real;
use sets::DisjointSets;

/// `None` when connected. Its order becomes the factor's permutation; surplus never joins components.
pub(crate) fn components<T: Real>(sddm: &Sddm<T>) -> Option<BlockLayout> {
    let laplacian = sddm.laplacian();
    let mut sets = DisjointSets::new(laplacian.n());
    for row in 0..laplacian.n() {
        let (neighbors, _) = laplacian.row(row);
        if neighbors.is_empty() {
            continue;
        }
        let mut root = sets.find(row as u32);
        for &col in neighbors {
            root = sets.union_resolved(root, col);
        }
    }
    sets.layout()
}

/// A floating component pins its last vertex; a grounded one appends a ground slot after its vertices.
pub(crate) struct Component<'a, T> {
    sddm: &'a Sddm<T>,
    vertices: BlockVertices<'a>,
    /// `Some` exactly when a vertex here has positive surplus.
    surplus: Option<&'a [T]>,
}

impl<'a, T: Real> Component<'a, T> {
    pub(crate) fn new(sddm: &'a Sddm<T>, vertices: BlockVertices<'a>) -> Self {
        let surplus = match sddm {
            Sddm::Laplacian(_) => None,
            Sddm::Grounded(grounded) => {
                let surplus = grounded.surplus();
                let holds_surplus = match &vertices {
                    BlockVertices::Whole(_) => true,
                    BlockVertices::Part { vertices, .. } => vertices
                        .iter()
                        .any(|&vertex| surplus[vertex as usize] > T::zero()),
                };
                holds_surplus.then_some(surplus)
            }
        };
        Self {
            sddm,
            vertices,
            surplus,
        }
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.surplus.is_some()
    }

    /// Every vertex but a floating component's pinned one.
    pub(crate) fn eliminated(&self) -> usize {
        let n = self.vertices.len();
        if self.is_grounded() {
            n
        } else {
            n.checked_sub(1)
                .expect("a component has at least one vertex")
        }
    }

    /// Names the component by what it holds rather than by how many precede it.
    pub(crate) fn first(&self) -> u64 {
        self.vertices.first()
    }

    pub(crate) fn global(&self, local: usize) -> usize {
        self.vertices.global(local)
    }

    /// The input's entries among the eliminated vertices, in component numbering.
    #[inline]
    pub(crate) fn entries(&self, mut entry: impl FnMut(usize, usize, T)) {
        let rows = self.eliminated();
        // Matched once, outside the walk, as `for_each_edge` is.
        match &self.vertices {
            BlockVertices::Whole(_) => self.sddm.entries(0..rows, |row, col, value| {
                if col < rows {
                    entry(row, col, value);
                }
            }),
            BlockVertices::Part { vertices, local_of } => self.sddm.entries(
                vertices[..rows].iter().map(|&global| global as usize),
                |row, col, value| {
                    let col = local_of[col] as usize;
                    if col < rows {
                        entry(local_of[row] as usize, col, value);
                    }
                },
            ),
        }
    }

    /// Builds the component's adjacency, which only the approximate arm needs.
    pub(crate) fn graph<C: EdgeCount>(&self) -> AdjListGraph<C, T> {
        let laplacian = self.sddm.laplacian();
        let vertices = self.vertices.len();
        let ground = vertices;

        // A row's own edges are its length and its ground edge; only edges from earlier rows need counting.
        let mut ground_degree = 0u32;
        let mut degrees: Vec<u32> = Vec::with_capacity(vertices + 1);
        degrees.extend((0..vertices).map(|local| {
            let global = self.vertices.global(local);
            let grounded = self
                .surplus
                .is_some_and(|surplus| surplus[global] > T::zero());
            ground_degree += u32::from(grounded);
            laplacian.row(global).0.len() as u32 + u32::from(grounded)
        }));
        if self.is_grounded() {
            degrees.push(ground_degree);
        }
        for_each_edge(laplacian, &self.vertices, |_, col, _| degrees[col] += 1);
        // One slot of slack takes the first fill edge: exact capacity measured +3% on degree-4 grids.
        let mut adj: Vec<Vec<Edge<T, C>>> = degrees
            .iter()
            .map(|&degree| Vec::with_capacity(degree as usize + 1))
            .collect();

        for_each_edge(laplacian, &self.vertices, |local, col, weight| {
            add_edge_pair(&mut adj, local, col, weight);
        });
        if let Some(surplus) = self.surplus {
            for local in 0..vertices {
                let s = surplus[self.vertices.global(local)];
                if s > T::zero() {
                    add_edge_pair(&mut adj, local, ground, s);
                }
            }
        }
        AdjListGraph::from_adjacency(adj)
    }
}

/// Every edge among the component's vertices, in its numbering.
#[inline(always)]
fn for_each_edge<T: Copy>(
    laplacian: &Laplacian<T>,
    block: &BlockVertices<'_>,
    mut edge: impl FnMut(usize, usize, T),
) {
    // Measured: an in-loop discriminant test spills `local_of` and reloads per edge.
    match block {
        BlockVertices::Whole(n) => {
            for local in 0..*n {
                let (neighbors, weights) = laplacian.row(local);
                for (&col, &weight) in neighbors.iter().zip(weights) {
                    edge(local, col as usize, weight);
                }
            }
        }
        BlockVertices::Part { vertices, local_of } => {
            // Narrowed so the bound lives in a register.
            let local_of = &local_of[..laplacian.n()];
            for (local, &global) in vertices.iter().enumerate() {
                let (neighbors, weights) = laplacian.row(global as usize);
                for (&col, &weight) in neighbors.iter().zip(weights) {
                    edge(local, local_of[col as usize] as usize, weight);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
