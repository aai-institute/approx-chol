//! A graph Laplacian by its edges, each column of `V` being `e_a − e_b`, and an SDDM over one.

use num_traits::PrimInt;

use super::error::{check, check_ends, length, nonnegative, ViewError};
use super::{Columns, FactorWidth2};
use crate::Real;

/// The Laplacian of an undirected graph on `n` vertices.
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
}

/// `L + diag(s)`: a Laplacian plus each vertex's surplus over diagonal dominance.
#[derive(Clone, Copy, Debug)]
pub struct Sddm<'a, T, I = u32>(pub(crate) FactorWidth2<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Sddm<'a, T, I> {
    /// `s[i]` finite and `≥ 0`, one per vertex, not all zero (that is a [`Laplacian`]).
    pub fn new(laplacian: Laplacian<'a, T, I>, s: &'a [T]) -> Result<Self, ViewError> {
        let Laplacian(mut v) = laplacian;
        length(v.n(), s.len())?;
        check(s, nonnegative, |vertex| ViewError::InvalidSurplus {
            vertex,
        })?;
        if s.iter().all(|&s| s == T::zero()) {
            return Err(ViewError::ZeroSurplus);
        }
        if let Columns::Incidence { surplus, .. } = &mut v.columns {
            *surplus = Some(s);
        }
        Ok(Self(v))
    }
}
