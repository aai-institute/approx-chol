//! A block assembled in memory, beside [`Ingestion`](super::Ingestion), for callers that never held a CSR.

use super::adjacency::{add_edge_pair, AdjListGraph, Edge};
use super::multiplicity::EdgeCount;
use crate::types::Real;

/// One connected block: its diagonal and strict upper triangle, both block-local.
pub(crate) struct CoreRows<T> {
    diagonal: Vec<T>,
    upper_ptr: Vec<usize>,
    upper: Vec<(u32, T)>,
    degree: Vec<u32>,
}

impl<T: Real> CoreRows<T> {
    /// A Laplacian from `(lo, hi, weight)` edges sorted by `(lo, hi)`, each pair once.
    pub(crate) fn from_edges(n: usize, edges: &[(u32, u32, T)]) -> Self {
        let mut diagonal = vec![T::zero(); n];
        let mut upper_ptr = vec![0usize; n + 1];
        for &(lo, hi, weight) in edges {
            debug_assert!(lo < hi);
            diagonal[lo as usize] = diagonal[lo as usize] + weight;
            diagonal[hi as usize] = diagonal[hi as usize] + weight;
            upper_ptr[lo as usize + 1] += 1;
        }
        for row in 0..n {
            upper_ptr[row + 1] += upper_ptr[row];
        }
        let upper = edges.iter().map(|&(_, hi, weight)| (hi, -weight)).collect();
        Self::assembled(diagonal, upper_ptr, upper)
    }

    /// Full symmetric rows with ascending columns; only the diagonal and the strict upper part are read.
    pub(crate) fn from_full_rows(indptr: &[u32], indices: &[u32], data: &[T]) -> Self {
        let n = indptr.len() - 1;
        let mut diagonal = vec![T::zero(); n];
        let mut upper_ptr = Vec::with_capacity(n + 1);
        let mut upper = Vec::new();
        upper_ptr.push(0);
        for row in 0..n {
            let span = indptr[row] as usize..indptr[row + 1] as usize;
            for (&col, &value) in indices[span.clone()].iter().zip(&data[span]) {
                if col as usize == row {
                    diagonal[row] = value;
                } else if col as usize > row {
                    upper.push((col, value));
                }
            }
            upper_ptr.push(upper.len());
        }
        Self::assembled(diagonal, upper_ptr, upper)
    }

    fn assembled(diagonal: Vec<T>, upper_ptr: Vec<usize>, upper: Vec<(u32, T)>) -> Self {
        let mut degree = vec![0u32; diagonal.len()];
        for row in 0..diagonal.len() {
            for &(col, _) in &upper[upper_ptr[row]..upper_ptr[row + 1]] {
                degree[row] += 1;
                degree[col as usize] += 1;
            }
        }
        Self {
            diagonal,
            upper_ptr,
            upper,
            degree,
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.diagonal.len()
    }

    pub(crate) fn diagonal(&self, row: usize) -> T {
        self.diagonal[row]
    }

    pub(crate) fn upper_row(&self, row: usize, mut entry: impl FnMut(usize, T)) {
        for &(col, value) in &self.upper[self.upper_ptr[row]..self.upper_ptr[row + 1]] {
            if value != T::zero() {
                entry(col as usize, value);
            }
        }
    }

    /// Same insertion order as [`Ingestion::block_graph`](super::Ingestion::block_graph) on the equivalent CSR.
    pub(crate) fn graph<C: EdgeCount>(&self) -> AdjListGraph<C, T> {
        let mut adj: Vec<Vec<Edge<T, C>>> = (self.degree.iter())
            .map(|&degree| Vec::with_capacity(degree as usize))
            .collect();
        for row in 0..self.n() {
            self.upper_row(row, |col, value| add_edge_pair(&mut adj, row, col, -value));
        }
        AdjListGraph::from_adjacency(adj)
    }
}
