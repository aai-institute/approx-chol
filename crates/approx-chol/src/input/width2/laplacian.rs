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
    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight 1; repeated edges sum.
    pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, ViewError> {
        check_ends(n, u, v)?;
        Ok(Self(FactorWidth2 {
            a: u,
            b: v,
            weights: Weights::Unit,
            columns: Columns::Incidence { n, surplus: None },
        }))
    }

    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight `w[k]`, finite and `≥ 0`; repeated edges sum.
    pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T]) -> Result<Self, ViewError> {
        let weights = Weights::Of(w).checked(u.len())?;
        let Self(laplacian) = Self::new(n, u, v)?;
        Ok(Self(FactorWidth2 {
            weights,
            ..laplacian
        }))
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
