//! [`Sddm`] to its connected components, each a classified view in its own numbering.

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

/// The input's connected components; the only source of [`Component`] views.
pub(crate) struct Components<'a, T> {
    sddm: &'a Sddm<T>,
    /// Both triangles' count per vertex, so adjacency lists never regrow.
    degrees: Vec<u32>,
    /// `None` when connected.
    layout: Option<BlockLayout>,
}

impl<'a, T: Real> Components<'a, T> {
    pub(crate) fn of(sddm: &'a Sddm<T>) -> Self {
        let (degrees, mut sets) = walk(sddm.laplacian());
        Self {
            sddm,
            degrees,
            layout: sets.layout(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.layout.as_ref().map_or(1, BlockLayout::block_count)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Component<'_, T>> + '_ {
        let whole = self
            .layout
            .is_none()
            .then(|| BlockVertices::Whole(self.sddm.n()));
        let parts = self.layout.iter().flat_map(BlockLayout::blocks);
        whole
            .into_iter()
            .chain(parts)
            .map(|vertices| self.component(vertices))
    }

    /// The one place a component's variant is decided; only split grounded input scans.
    fn component<'s>(&'s self, vertices: BlockVertices<'s>) -> Component<'s, T> {
        let view = View {
            laplacian: self.sddm.laplacian(),
            degrees: &self.degrees,
            vertices,
        };
        let Sddm::Grounded(grounded) = self.sddm else {
            return Component::Laplacian(view);
        };
        let surplus = grounded.surplus();
        let holds_surplus = match &view.vertices {
            BlockVertices::Whole(_) => true,
            BlockVertices::Part { vertices, .. } => vertices
                .iter()
                .any(|&vertex| surplus[vertex as usize] > T::zero()),
        };
        if holds_surplus {
            Component::Grounded { view, surplus }
        } else {
            Component::Laplacian(view)
        }
    }

    /// Component-contiguous input order; `None` when connected.
    pub(crate) fn into_order(self) -> Option<Vec<u32>> {
        self.layout.map(BlockLayout::into_order)
    }
}

/// One component of the input, borrowed in place; the variant is its gauge.
pub(crate) enum Component<'a, T> {
    Laplacian(View<'a, T>),
    /// `surplus` is the input's, indexed by input vertex.
    Grounded {
        view: View<'a, T>,
        surplus: &'a [T],
    },
}

/// The input's rows seen in a component's own numbering.
pub(crate) struct View<'a, T> {
    laplacian: &'a Laplacian<T>,
    degrees: &'a [u32],
    vertices: BlockVertices<'a>,
}

impl<T> View<'_, T> {
    pub(crate) fn vertices(&self) -> &BlockVertices<'_> {
        &self.vertices
    }

    /// The edge weights above the row's diagonal, by local column.
    pub(crate) fn upper_row(&self, local: usize, mut entry: impl FnMut(usize, T))
    where
        T: Copy,
    {
        let (neighbors, weights) = self.laplacian.row(self.vertices.global(local));
        for (&col, &weight) in neighbors.iter().zip(weights) {
            entry(self.vertices.local(col as usize), weight);
        }
    }
}

impl<T: Real> Component<'_, T> {
    pub(crate) fn view(&self) -> &View<'_, T> {
        match self {
            Self::Laplacian(view) | Self::Grounded { view, .. } => view,
        }
    }

    /// Every slot but one: a floating component's free vertex, or a grounded one's ground.
    pub(crate) fn eliminated(&self) -> usize {
        match self {
            Self::Laplacian(view) => view.vertices.len() - 1,
            Self::Grounded { view, .. } => view.vertices.len(),
        }
    }

    /// A grounded component's ground is appended as the last vertex.
    pub(crate) fn graph<C: EdgeCount>(&self) -> AdjListGraph<C, T> {
        let view = self.view();
        let laplacian = view.laplacian;
        let k = view.vertices.len();
        let surplus = match self {
            Self::Laplacian(_) => None,
            Self::Grounded { surplus, .. } => Some(*surplus),
        };
        let grounds = |global: usize| surplus.is_some_and(|surplus| surplus[global] > T::zero());
        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(k + usize::from(surplus.is_some()));
        adj.extend((0..k).map(|local| {
            let global = view.vertices.global(local);
            Vec::with_capacity(view.degrees[global] as usize + usize::from(grounds(global)))
        }));

        // Measured: an in-loop discriminant test spills the map and reloads per edge.
        match &view.vertices {
            BlockVertices::Whole(_) => {
                for local in 0..k {
                    let (neighbors, weights) = laplacian.row(local);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        add_edge_pair(&mut adj, local, col as usize, weight);
                    }
                }
            }
            BlockVertices::Part {
                vertices,
                position,
                start,
            } => {
                // Narrowed so the bound lives in a register.
                let position = &position[..laplacian.n()];
                for (local, &global) in vertices.iter().enumerate() {
                    let (neighbors, weights) = laplacian.row(global as usize);
                    for (&col, &weight) in neighbors.iter().zip(weights) {
                        let col = (position[col as usize] - start) as usize;
                        add_edge_pair(&mut adj, local, col, weight);
                    }
                }
            }
        }

        if let Some(surplus) = surplus {
            let degree = (0..k)
                .filter(|&local| grounds(view.vertices.global(local)))
                .count();
            adj.push(Vec::with_capacity(degree));
            for local in 0..k {
                let s = surplus[view.vertices.global(local)];
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
