use super::OffDiagonal;
use crate::Real;

/// `F` in `M = F A_c F`: signs and scales, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[derive(Clone, Debug)]
pub(crate) struct Congruence<T>(Kind<T>);

#[derive(Clone, Debug)]
enum Kind<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    Factors(Box<[T]>),
}

impl<T: Real> Congruence<T> {
    pub(super) fn identity() -> Self {
        Self(Kind::Identity)
    }

    /// A positive bipartite block's signature, unmaterialized: every vertex from `first` on is negated.
    pub(super) fn negate_from(first: usize) -> Self {
        Self(Kind::NegateFrom(first))
    }

    /// `true` where the vertex is negated.
    pub(super) fn negated(signs: &[bool]) -> Self {
        todo!()
    }

    /// `S R` for these signs and positive scales `R`.
    pub(super) fn scaled(self, scales: &[T]) -> Self {
        todo!()
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match &self.0 {
            Kind::Identity => true,
            Kind::NegateFrom(first) => *first <= n,
            Kind::Factors(factors) => factors.len() == n,
        }
    }

    /// `F X F`: `M`'s off-diagonal from `A_c`'s; `None` where an entry leaves [`in_range`](crate::storage::in_range).
    pub(super) fn transform(&self, off_diagonal: OffDiagonal<T>) -> Option<OffDiagonal<T>> {
        todo!()
    }

    /// `x ← F x`; with `P` projecting out `F·1` when the part floats, else `I`, `A_c⁺ = P F M⁺ F P`, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        todo!()
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_out_null(&self, x: &mut [T]) {
        todo!()
    }
}
