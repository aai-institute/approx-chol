//! The factor-width-2 inputs: `A = V W Vᵀ + diag(s)`, column `k` of `V` joining `a[k]` and `b[k]`.

pub(super) mod bipartite;
pub(super) mod error;
pub(super) mod laplacian;

use num_traits::PrimInt;

use self::error::{check_ends, length, ViewError};
use super::repr::gram::{Gram, Layout, Values, Weights};
use crate::storage::Real;

/// `V Vᵀ` on `n` vertices, column `k` of `V` being `x[k] e_a[k] + y[k] e_b[k]`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32> {
    n: usize,
    a: &'a [I],
    b: &'a [I],
    x: &'a [T],
    y: &'a [T],
}

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `a[k] ≠ b[k] < n`; `a`, `x`, `b`, `y` one entry per column.
    pub fn new(
        n: usize,
        (a, x): (&'a [I], &'a [T]),
        (b, y): (&'a [I], &'a [T]),
    ) -> Result<Self, ViewError> {
        length(a.len(), x.len())?;
        length(a.len(), y.len())?;
        check_ends(n, a, b)?;
        Ok(Self { n, a, b, x, y })
    }

    pub(super) fn gram(self) -> Gram<'a, T, I> {
        let Self { n, a, b, x, y } = self;
        let values = Values::Valued {
            x: Some(x),
            y: Some(y),
        };
        Gram::new((a, b), Layout::Whole { n }, values, Weights::Unit)
    }
}
