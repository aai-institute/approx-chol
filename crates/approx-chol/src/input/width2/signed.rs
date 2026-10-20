//! A signed graph by its edges: each column of `V` is `e_a ± e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::{General, Gram, Incidences, One};
use super::{check_ends, length, IntoColumns, ViewError};
use crate::Real;

/// `A = Σ_k w_k (e_a ± e_b)(e_a ± e_b)ᵀ + diag(g)`: `+` where `positive[k]`, so edge `k` adds `+w_k` to `a_ab`, else `−w_k`.
#[derive(Clone, Copy, Debug)]
pub struct SignedGraph<'a, T, I = u32>(Gram<'a, T, I, General, One, Incidences<'a>>);

impl<'a, T: Real, I: PrimInt> SignedGraph<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n` under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        positive: &'a [bool],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        length(a.len(), positive.len())?;
        let edges = Edges::new(a, b, weights)?;
        check_ends(n, a, b)?;
        let head = Incidences(positive);
        Ok(Self(Gram::new(n, General, edges, One, head, None)))
    }

    /// `A + diag(g)`, one finite nonnegative `g_i` per vertex.
    pub fn with_ground(self, ground: &'a [T]) -> Result<Self, ViewError> {
        self.0.with_diagonal(ground).map(Self)
    }

    pub(crate) fn n(&self) -> usize {
        self.0.n()
    }

    pub(crate) fn gram(self) -> Gram<'a, T, I, General, One, Incidences<'a>> {
        self.0
    }
}
