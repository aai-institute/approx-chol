//! A graph Laplacian by its edges, each column of `V` being `e_a − e_b`, and an SDDM over one.

use num_traits::PrimInt;

use super::error::{check, check_ends, length, nonnegative, ViewError};
use crate::input::repr::gram::{Gram, Layout, Values, Weights};
use crate::storage::Real;

/// The Laplacian of an undirected graph on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32> {
    n: usize,
    u: &'a [I],
    v: &'a [I],
    w: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight 1; repeated edges sum.
    pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, ViewError> {
        check_ends(n, u, v)?;
        Ok(Self { n, u, v, w: None })
    }

    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight `w[k]`, finite and `≥ 0`; repeated edges sum.
    pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T]) -> Result<Self, ViewError> {
        length(u.len(), w.len())?;
        check(w, nonnegative, |entry| ViewError::InvalidWeight { entry })?;
        check_ends(n, u, v)?;
        Ok(Self {
            n,
            u,
            v,
            w: Some(w),
        })
    }

    pub(in crate::input) fn gram(self) -> Gram<'a, T, I> {
        self.plus(None)
    }

    /// `L + diag(s)` as the core's `V W Vᵀ` and surplus.
    fn plus(self, surplus: Option<&'a [T]>) -> Gram<'a, T, I> {
        let Self { n, u, v, w } = self;
        let weights = w.map_or(Weights::Unit, Weights::Of);
        Gram::new(
            (u, v),
            Layout::Whole { n },
            Values::Incidence { surplus },
            weights,
        )
    }
}

/// `L + diag(s)`: a Laplacian plus each vertex's surplus over diagonal dominance.
#[derive(Clone, Copy, Debug)]
pub struct Sddm<'a, T, I = u32> {
    laplacian: Laplacian<'a, T, I>,
    surplus: &'a [T],
}

impl<'a, T: Real, I: PrimInt> Sddm<'a, T, I> {
    /// `s[i]` finite and `≥ 0`, one per vertex, not all zero (that is a [`Laplacian`]).
    pub fn new(laplacian: Laplacian<'a, T, I>, surplus: &'a [T]) -> Result<Self, ViewError> {
        length(laplacian.n, surplus.len())?;
        check(surplus, nonnegative, |vertex| ViewError::InvalidSurplus {
            vertex,
        })?;
        match surplus.iter().all(|&s| s == T::zero()) {
            true => Err(ViewError::ZeroSurplus),
            false => Ok(Self { laplacian, surplus }),
        }
    }

    pub(in crate::input) fn gram(self) -> Gram<'a, T, I> {
        self.laplacian.plus(Some(self.surplus))
    }
}
