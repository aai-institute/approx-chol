//! A signed graph by its edges: each column of `V` is `e_a ∓ e_b`, a factor-width-2 Gram whose signs are searched and whose `V` decides its floating.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{check_ends, Column, IntoColumns, ViewError};
use crate::Real;

/// `A = Σ_k w_k (e_a − σ_k e_b)(e_a − σ_k e_b)ᵀ`, `σ_k = −1` where `positive[k]`, so edge `k` adds `+w_k` to `a_ab`, else `−w_k`.
#[derive(Clone, Copy, Debug)]
pub struct SignedGraph<'a, T, I = u32> {
    n: usize,
    edges: Edges<'a, T, I>,
    positive: &'a [bool],
}

impl<'a, T: Real, I: PrimInt> SignedGraph<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k]`, both below `n`, under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        positive: &'a [bool],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        if positive.len() != a.len() {
            return Err(ViewError::LengthMismatch {
                expected: a.len(),
                found: positive.len(),
            });
        }
        let edges = Edges::new(a, b, weights)?;
        check_ends(n, a, b)?;
        Ok(Self { n, edges, positive })
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for SignedGraph<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        let positive = self.positive;
        let sign = |k: usize| match positive[k] {
            true => T::one(),
            false => -T::one(),
        };
        self.edges.for_each_column(0, sign, visit)
    }
}
