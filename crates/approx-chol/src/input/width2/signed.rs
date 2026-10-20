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
    /// Edge `k` joins `a[k] ≠ b[k] < n`, unit weight; lengths and ends are checked.
    pub fn new(n: usize, a: &'a [I], b: &'a [I], positive: &'a [bool]) -> Result<Self, ViewError> {
        length(a.len(), positive.len())?;
        let edges = Edges::new(a, b)?;
        check_ends(n, a, b)?;
        let head = Incidences(positive);
        Ok(Self(Gram::new(n, General, edges, One, head, None)))
    }

    /// `A` under `weights`, one per edge; `W = I` without it.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        self.0.with_weights(weights).map(Self)
    }

    /// `A + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn plus_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.plus_diagonal(diagonal).map(Self)
    }

    pub(crate) fn n(&self) -> usize {
        self.0.n()
    }

    pub(crate) fn gram(self) -> Gram<'a, T, I, General, One, Incidences<'a>> {
        self.0
    }
}
