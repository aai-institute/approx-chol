use crate::Real;

/// `D` in `A_c = D M D`: signs and scales, finite and nonzero, and the null vector `D⁻¹·1` it implies.
#[derive(Clone, Debug)]
pub(crate) struct Congruence<T>(Kind<T>);

#[derive(Clone, Debug)]
enum Kind<T> {
    Identity,
    /// A bipartite Laplacian's signature: every vertex from this one on is negated.
    NegateFrom(usize),
    Factors(Box<[T]>),
}

impl<T: Real> Congruence<T> {
    pub(super) fn identity() -> Self {
        Self(Kind::Identity)
    }

    pub(super) fn negate_from(first: usize) -> Self {
        Self(Kind::NegateFrom(first))
    }

    /// `None` unless every factor is finite and nonzero.
    pub(super) fn factors(factors: Vec<T>) -> Option<Self> {
        todo!()
    }

    /// Its own inverse up to scale, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        todo!()
    }

    /// Projects out `D⁻¹·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_null(&self, x: &mut [T]) {
        todo!()
    }
}
