//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{check_ends, Column, IntoColumns, ViewError};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g`: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32> {
    n: usize,
    edges: Edges<'a, T, I>,
    ground: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k] < n` under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(a, b, weights)?;
        check_ends(n, a, b)?;
        Ok(Self {
            n,
            edges,
            ground: None,
        })
    }

    /// `L + diag(g)`, one finite nonnegative `g_i` per vertex.
    pub fn grounded(self, ground: &'a [T]) -> Result<Self, ViewError> {
        if ground.len() != self.n {
            return Err(ViewError::LengthMismatch {
                expected: self.n,
                found: ground.len(),
            });
        }
        let valid = |g: &T| g.is_finite() && *g >= T::zero();
        if let Some(vertex) = ground.iter().position(|g| !valid(g)) {
            return Err(ViewError::InvalidGround { vertex });
        }
        Ok(Self {
            ground: Some(ground),
            ..self
        })
    }

    /// `g`, which is no column of `V`.
    pub(crate) fn ground(&self) -> Option<&'a [T]> {
        self.ground
    }
}

/// The columns of `L` alone: the ground is read through [`Laplacian::ground`].
impl<T: Real, I: PrimInt> IntoColumns<T> for Laplacian<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        self.edges
            .for_each_column(0, |_| T::one(), |_| -T::one(), visit)
    }
}
