//! A signed graph by its edges: each column of `V` is `e_a ± e_b`.

use num_traits::PrimInt;

use super::edges::Weights;
use super::gram::{Gram, Incidence, VertexPartition};
use super::ViewError;
use crate::Real;

/// `A = Σ_k w_k (e_a ± e_b)(e_a ± e_b)ᵀ + diag(g)`: `+` where `positive[k]`, so edge `k` adds `+w_k` to `a_ab`, else `−w_k`.
#[derive(Clone, Copy, Debug)]
pub struct SignedGraph<'a, T, I = u32>(Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> SignedGraph<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n`, unit weight; lengths and ends are checked.
    pub fn new(n: usize, a: &'a [I], b: &'a [I], positive: &'a [bool]) -> Result<Self, ViewError> {
        let incidence = Incidence::Signed(positive);
        Gram::incidence(n, VertexPartition::Unpartitioned, a, b, incidence).map(Self)
    }

    /// `A` under `weights`, one per edge; `W = I` without it.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        self.0.with_weights(weights).map(Self)
    }

    /// `A + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn with_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.with_diagonal(diagonal).map(Self)
    }
}

impl<'a, T, I> From<SignedGraph<'a, T, I>> for Gram<'a, T, I> {
    fn from(graph: SignedGraph<'a, T, I>) -> Self {
        graph.0
    }
}
