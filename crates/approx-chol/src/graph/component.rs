//! One connected block of the SDDM input, grounded by its own surplus or floating.

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::blocks::BlockVertices;
use super::multiplicity::EdgeCount;
use crate::sddm::Laplacian;
use crate::types::Real;

/// How a component's free slot is fixed: by its own ground, or by pinning its last vertex.
pub(crate) enum Gauge<'a, T> {
    Floating,
    /// The input's surplus, indexed by global vertex.
    Grounded(&'a [T]),
}

/// A floating component pins its last vertex; a grounded one appends a ground slot after its vertices.
pub(crate) struct Component<'a, T> {
    laplacian: &'a Laplacian<T>,
    vertices: BlockVertices<'a>,
    gauge: Gauge<'a, T>,
}

impl<'a, T: Real> Component<'a, T> {
    /// `surplus` is `Some` only when some component holds surplus, so a whole input with it is grounded.
    pub(crate) fn new(
        laplacian: &'a Laplacian<T>,
        vertices: BlockVertices<'a>,
        surplus: Option<&'a [T]>,
    ) -> Self {
        let holds_surplus = |surplus: &[T]| match &vertices {
            BlockVertices::Whole(_) => true,
            BlockVertices::Part { vertices, .. } => vertices
                .iter()
                .any(|&vertex| surplus[vertex as usize] > T::zero()),
        };
        let gauge = match surplus {
            Some(surplus) if holds_surplus(surplus) => Gauge::Grounded(surplus),
            _ => Gauge::Floating,
        };
        Self {
            laplacian,
            vertices,
            gauge,
        }
    }

    pub(crate) fn is_grounded(&self) -> bool {
        matches!(self.gauge, Gauge::Grounded(_))
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
            BlockVertices::Whole(_) => self.input_entries(0..rows, |row, col, value| {
                if col < rows {
                    entry(row, col, value);
                }
            }),
            BlockVertices::Part { vertices, local_of } => self.input_entries(
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

    /// Each row's upper entries, then its surplus; a diagonal arrives as summands, in storage order.
    #[inline]
    fn input_entries(
        &self,
        rows: impl Iterator<Item = usize> + Clone,
        mut entry: impl FnMut(usize, usize, T),
    ) {
        for row in rows.clone() {
            let (neighbors, weights) = self.laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                let col = col as usize;
                entry(row, col, -weight);
                entry(row, row, weight);
                entry(col, col, weight);
            }
        }
        if let Gauge::Grounded(surplus) = self.gauge {
            for row in rows {
                entry(row, row, surplus[row]);
            }
        }
    }

    /// Builds the component's adjacency, which only the approximate arm needs.
    pub(crate) fn graph<C: EdgeCount>(&self) -> AdjListGraph<C, T> {
        let laplacian = self.laplacian;
        let surplus = match self.gauge {
            Gauge::Grounded(surplus) => Some(surplus),
            Gauge::Floating => None,
        };
        let vertices = self.vertices.len();
        let ground = vertices;

        // A row's own edges are its length and its ground edge; only edges from earlier rows need counting.
        let mut ground_degree = 0u32;
        let mut degrees: Vec<u32> = Vec::with_capacity(vertices + 1);
        degrees.extend((0..vertices).map(|local| {
            let global = self.vertices.global(local);
            let grounded = surplus.is_some_and(|surplus| surplus[global] > T::zero());
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
        if let Some(surplus) = surplus {
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
