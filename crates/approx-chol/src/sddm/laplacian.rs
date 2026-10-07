use super::floor;
use crate::types::Real;

/// `L(G)` stored as `G`'s strict upper adjacency with positive weights; the diagonal is implied.
pub(crate) struct Laplacian<T> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
}

impl<T> Laplacian<T> {
    pub(crate) fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    #[inline]
    pub(crate) fn row(&self, i: usize) -> (&[u32], &[T]) {
        let (from, to) = (self.row_ptrs[i] as usize, self.row_ptrs[i + 1] as usize);
        (&self.neighbors[from..to], &self.weights[from..to])
    }
}

impl<T: Real> Laplacian<T> {
    fn is_upper_adjacency(&self) -> bool {
        let n = self.n();
        self.row_ptrs.first() == Some(&0)
            && self.row_ptrs.last().map(|&end| end as usize) == Some(self.neighbors.len())
            && self.weights.len() == self.neighbors.len()
            && (0..n).all(|row| {
                let (neighbors, weights) = self.row(row);
                neighbors.first().is_none_or(|&col| col as usize > row)
                    && neighbors.windows(2).all(|pair| pair[0] < pair[1])
                    && neighbors.last().is_none_or(|&col| (col as usize) < n)
                    && weights.iter().all(|&w| w.is_finite() && w >= floor())
            })
    }
}

/// The only way to store a [`Laplacian`]: rows in order, each weighted degree summed as its edge lands.
pub(super) struct UpperRows<T> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
    degrees: Vec<T>,
}

impl<T: Real> UpperRows<T> {
    pub(super) fn with_capacity(n: usize, edges: usize) -> Self {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        row_ptrs.push(0);
        Self {
            row_ptrs,
            neighbors: Vec::with_capacity(edges),
            weights: Vec::with_capacity(edges),
            degrees: vec![T::zero(); n],
        }
    }

    /// An entry right of the diagonal of the row being stored, columns ascending.
    #[inline]
    pub(super) fn push(&mut self, row: usize, col: usize, weight: T) {
        self.neighbors.push(col as u32);
        self.weights.push(weight);
        self.degrees[row] = self.degrees[row] + weight;
        self.degrees[col] = self.degrees[col] + weight;
    }

    pub(super) fn end_row(&mut self) {
        self.row_ptrs.push(self.neighbors.len() as u32);
    }

    pub(super) fn finish(self) -> Summed<T> {
        let laplacian = Laplacian {
            row_ptrs: self.row_ptrs,
            neighbors: self.neighbors,
            weights: self.weights,
        };
        debug_assert!(laplacian.is_upper_adjacency());
        Summed {
            laplacian,
            degrees: self.degrees,
        }
    }
}

/// A [`Laplacian`] with its weighted degrees, summed in storage order: the order every diagonal is summed in.
pub(super) struct Summed<T> {
    laplacian: Laplacian<T>,
    degrees: Vec<T>,
}

impl<T: Real> Summed<T> {
    /// The first vertex whose diagonal, its degree then its surplus, is not finite.
    pub(super) fn first_non_finite_diagonal(&self, surplus: &[T]) -> Option<usize> {
        self.degrees
            .iter()
            .zip(surplus)
            .position(|(&degree, &s)| !(degree + s).is_finite())
    }

    pub(super) fn into_laplacian(self) -> Laplacian<T> {
        self.laplacian
    }
}
