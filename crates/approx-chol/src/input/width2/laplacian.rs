//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::{General, Gram, MinusOne, One};
use super::{check_diagonal, check_ends, ViewError};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32> {
    n: usize,
    edges: Edges<'a, T, I>,
    ground: Option<&'a [T]>,
}

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
        Ok(Self {
            n,
            edges,
            ground: None,
        })
    }

    /// `L + diag(g)`, one finite nonnegative `g_i` per vertex.
    pub fn with_ground(self, ground: &'a [T]) -> Result<Self, ViewError> {
        check_diagonal(self.n, ground)?;
        Ok(Self {
            ground: Some(ground),
            ..self
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn gram(self) -> Gram<'a, T, I, General, One, MinusOne> {
        Gram::new(self.n, General, self.edges, One, MinusOne, self.ground)
    }
}
