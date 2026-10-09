use crate::Real;

/// `D` in `A_c = D M D`: signs and dominance scales, applied on both sides of the solve.
#[derive(Clone, Debug)]
pub(crate) enum Diagonal<T> {
    Identity,
    /// A bipartite Laplacian's signature: negate every vertex from this one on.
    NegateFrom(usize),
    Scaled(Scales<T>),
}

/// Positive and finite, so `D⁻¹` exists; only the reduction makes one.
#[derive(Clone, Debug)]
pub(crate) struct Scales<T>(Box<[T]>);

impl<T: Real> Scales<T> {
    /// `None` unless every factor is positive and finite.
    pub(super) fn new(factors: Vec<T>) -> Option<Self> {
        todo!()
    }
}

impl<T: Real> Diagonal<T> {
    pub(crate) fn apply(&self, x: &mut [T]) {
        todo!()
    }

    /// `A`'s null vector when `M`'s is `1`: `D⁻¹·1`.
    pub(crate) fn null_vector(&self) -> NullVector<'_, T> {
        match self {
            Self::Identity | Self::NegateFrom(_) => NullVector::Ones,
            Self::Scaled(Scales(factors)) => NullVector::Reciprocal(factors),
        }
    }
}

/// Direction `A⁺` projects out; sign flips keep plain mean-zero exact.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NullVector<'a, T> {
    Ones,
    /// `1 / factors`, max-normalized: the raw `zᵀz` overflows at a 1e200 scale range.
    Reciprocal(&'a [T]),
}

impl<T: Real> NullVector<'_, T> {
    pub(crate) fn project(self, x: &mut [T]) {
        todo!()
    }
}
