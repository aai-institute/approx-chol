use crate::Real;

/// `F` in `M = F A_c F`: signs and scales, finite and nonzero; `A_c`'s null vector is then `F·1`.
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

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match &self.0 {
            Kind::Identity => true,
            Kind::NegateFrom(first) => *first <= n,
            Kind::Factors(factors) => factors.len() == n,
        }
    }

    /// `None` unless every factor is finite and nonzero.
    pub(super) fn factors(factors: Vec<T>) -> Option<Self> {
        todo!()
    }

    /// `x ← F x`; `A_c⁺ = F M⁺ F` on `A_c`'s range, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        todo!()
    }

    /// Projects out `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_null(&self, x: &mut [T]) {
        todo!()
    }
}
