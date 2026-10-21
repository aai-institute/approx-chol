//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::error::{check, check_ends, length, nonnegative, ViewError};
use super::{Columns, FactorWidth2};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32>(pub(crate) FactorWidth2<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight 1; repeated edges sum.
    pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, ViewError> {
        check_ends(n, u, v)?;
        Ok(Self::of(n, u, v, None))
    }

    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight `w[k]`, finite and `≥ 0`; repeated edges sum.
    pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T]) -> Result<Self, ViewError> {
        length(u.len(), w.len())?;
        check(w, nonnegative, |entry| ViewError::InvalidWeight { entry })?;
        check_ends(n, u, v)?;
        Ok(Self::of(n, u, v, Some(w)))
    }

    fn of(n: usize, u: &'a [I], v: &'a [I], w: Option<&'a [T]>) -> Self {
        Self(FactorWidth2 {
            a: u,
            b: v,
            columns: Columns::Incidence {
                n,
                w,
                surplus: None,
            },
        })
    }

    /// `L + diag(g)`, one finite nonnegative `g_i` per vertex.
    pub fn grounded(self, ground: &'a [T]) -> Result<Self, ViewError> {
        let Self(mut v) = self;
        length(v.n(), ground.len())?;
        check(ground, nonnegative, |vertex| ViewError::InvalidGround {
            vertex,
        })?;
        if let Columns::Incidence { surplus, .. } = &mut v.columns {
            *surplus = Some(ground);
        }
        Ok(Self(v))
    }
}
