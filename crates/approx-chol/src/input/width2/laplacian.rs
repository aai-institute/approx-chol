//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::edges::Weights;
use super::gram::{Gram, Incidence, VertexPartition};
use super::ViewError;
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32>(Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n`, unit weight; lengths and ends are checked.
    pub fn new(n: usize, a: &'a [I], b: &'a [I]) -> Result<Self, ViewError> {
        Gram::incidence(n, VertexPartition::Unpartitioned, a, b, Incidence::Oriented).map(Self)
    }

    /// `L` under `weights`, one per edge; `W = I` without it.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        self.0.with_weights(weights).map(Self)
    }

    /// `L + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn with_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.with_diagonal(diagonal).map(Self)
    }
}

impl<'a, T, I> From<Laplacian<'a, T, I>> for Gram<'a, T, I> {
    fn from(laplacian: Laplacian<'a, T, I>) -> Self {
        laplacian.0
    }
}
