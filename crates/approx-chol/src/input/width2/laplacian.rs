//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::error::{check, check_ends, length, nonnegative, ViewError};
use super::weights::Weights;
use super::{Columns, FactorWidth2};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32>(pub(crate) FactorWidth2<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n` under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let weights = weights.of_columns(a.len())?;
        check_ends(n, a, b)?;
        let columns = Columns::Incidence { n, surplus: None };
        Ok(Self(FactorWidth2::of(a, b, columns, weights)))
    }

    /// `L + diag(g)`, one finite nonnegative `g_i` per vertex.
    pub fn grounded(self, ground: &'a [T]) -> Result<Self, ViewError> {
        let v = self.0;
        let n = v.n();
        length(n, ground.len())?;
        check(ground, nonnegative, |vertex| ViewError::InvalidGround {
            vertex,
        })?;
        Ok(Self(FactorWidth2 {
            columns: Columns::Incidence {
                n,
                surplus: Some(ground),
            },
            ..v
        }))
    }
}
