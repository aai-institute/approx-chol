//! A graph Laplacian by its edges, each column of `V` being `e_a − e_b`, and an SDDM over one.

use num_traits::PrimInt;

use super::{Columns, FactorWidth2};
use crate::input::error::{check_ends, length, nonnegative, InputError};
use crate::Real;

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
    pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, InputError> {
        check_ends(n, ("u", u), ("v", v), |edge| InputError::SelfLoop { edge })?;
        Ok(Self { n, u, v, w: None })
    }

    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight `w[k]`, [in range](InputError::OutOfRange) and `≥ 0`; repeated edges sum.
    pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T]) -> Result<Self, InputError> {
        length("w", u.len(), w.len())?;
        nonnegative("w", w)?;
        check_ends(n, ("u", u), ("v", v), |edge| InputError::SelfLoop { edge })?;
        Ok(Self {
            n,
            u,
            v,
            w: Some(w),
        })
    }

    pub(crate) fn lower(self) -> FactorWidth2<'a, T, I> {
        self.plus(None)
    }

    /// `L + diag(s)` as the core's `V Vᵀ` and surplus.
    fn plus(self, surplus: Option<&'a [T]>) -> FactorWidth2<'a, T, I> {
        let Self { n, u, v, w } = self;
        FactorWidth2 {
            a: u,
            b: v,
            columns: Columns::Incidence { n, w, surplus },
        }
    }
}

/// `L + diag(s)`: a Laplacian plus each vertex's surplus over diagonal dominance. An `s_i` within `16ε` of its row's diagonal is rounding: a component grounds on its `s_i` past that, or, with none past, on all of them unless together they stay within `5ε` of its trace, where it floats.
#[derive(Clone, Copy, Debug)]
pub struct Sddm<'a, T, I = u32> {
    laplacian: Laplacian<'a, T, I>,
    /// `None` when every `s_i` is zero: the matrix is its Laplacian.
    surplus: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Sddm<'a, T, I> {
    /// `s[i]` [in range](InputError::OutOfRange) and `≥ 0`, one per vertex. Zeros are no ground: a coupled component whose `s_i` are all zero floats, as [`Component::floats`](crate::Component::floats) reports, and a lone vertex with `s_i = 0` solves to 0.
    pub fn new(laplacian: Laplacian<'a, T, I>, surplus: &'a [T]) -> Result<Self, InputError> {
        length("surplus", laplacian.n, surplus.len())?;
        nonnegative("surplus", surplus)?;
        let surplus = Some(surplus).filter(|s| s.iter().any(|&s| s != T::zero()));
        Ok(Self { laplacian, surplus })
    }

    pub(crate) fn lower(self) -> FactorWidth2<'a, T, I> {
        self.laplacian.plus(self.surplus)
    }
}
