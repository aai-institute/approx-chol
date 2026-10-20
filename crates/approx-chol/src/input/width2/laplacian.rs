//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::{General, Gram, MinusOne, One};
use super::{check_ends, IntoColumns, ViewError};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32>(Gram<'a, T, I, General, One, MinusOne>);

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n` under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(a, b, weights)?;
        check_ends(n, a, b)?;
        Ok(Self(Gram::new(n, General, edges, One, MinusOne, None)))
    }

    /// `L + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn with_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.with_diagonal(diagonal).map(Self)
    }

    pub(crate) fn n(&self) -> usize {
        self.0.n()
    }

    pub(crate) fn gram(self) -> Gram<'a, T, I, General, One, MinusOne> {
        self.0
    }
}
