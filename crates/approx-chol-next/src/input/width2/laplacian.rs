//! A graph Laplacian by its edges: each column of `V` is `e_a − e_b`, so `A` is a dominant Z-matrix, balanced but for its ground, by shape.

use num_traits::PrimInt;

use super::bipartite::Weights;
use super::{check_ends, index, Column, IntoColumns, ViewError, Weight};
use crate::Real;

/// Weighted multigraph on `n` vertices, grounded by `g` when given: `A = Σ_k w_k (e_a − e_b)(e_a − e_b)ᵀ + diag(g)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32> {
    n: usize,
    a: &'a [I],
    b: &'a [I],
    weights: Weights<'a, T>,
    ground: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `a[k] ≠ b[k]`, both below `n`, under `weights`; lengths, ends and weights are checked.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        if a.len() != b.len() {
            return Err(ViewError::LengthMismatch {
                expected: a.len(),
                found: b.len(),
            });
        }
        weights.check(a.len())?;
        check_ends(n, a, b)?;
        Ok(Self {
            n,
            a,
            b,
            weights,
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

    pub(crate) fn ground(&self) -> Option<&'a [T]> {
        self.ground
    }

    #[inline(always)]
    fn columns(self, weight: impl Fn(usize) -> Weight<T>, mut visit: impl FnMut(Column<T>)) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: index(*a) as u32,
                b: index(*b) as u32,
                v_a: T::one(),
                v_b: -T::one(),
                weight: weight(k),
            });
        }
    }
}

/// The ground is not a column: the shape grounds each component on it directly.
impl<T: Real, I: PrimInt> IntoColumns<T> for Laplacian<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        match self.weights {
            Weights::Unit => self.columns(|_| Weight::Unit, visit),
            Weights::Of(w) => self.columns(|k| Weight::Of(w[k]), visit),
            Weights::SqrtOf(s) => self.columns(|k| Weight::SqrtOf(s[k]), visit),
        }
    }
}
