//! A signed graph by its edges: each column of `V` is `e_a ± e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{check_ends, length, Column, IntoColumns, ViewError};
use crate::{Index, Real};

/// `A = Σ_k w_k (e_a ± e_b)(e_a ± e_b)ᵀ`: `+` where `positive[k]`, so edge `k` adds `+w_k` to `a_ab`, else `−w_k`.
#[derive(Clone, Copy, Debug)]
pub struct SignedGraph<'a, T, I = u32> {
    n: usize,
    edges: Edges<'a, T, I>,
    positive: &'a [bool],
}

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
        Ok(Self { n, edges, positive })
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for SignedGraph<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let positive = self.positive;
        let sign = |k: usize| match positive[k] {
            true => T::one(),
            false => -T::one(),
        };
        self.edges.for_each_column(0, |_| T::one(), sign, visit)
    }
}
