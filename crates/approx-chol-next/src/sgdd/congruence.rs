use super::OffDiagonal;
use crate::storage::CrossBlock;
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

    /// Folds a positive cross block: the kept side negated.
    pub(super) fn negate_kept(cross: &CrossBlock<T>) -> Self {
        Self(Kind::NegateFrom(cross.n_eliminated()))
    }

    /// `true` where the vertex is negated.
    pub(super) fn signs(negated: &[bool]) -> Self {
        todo!()
    }

    /// `S R` for these signs `S` and positive scales `R`; `None` if a product leaves the finite range.
    pub(super) fn then_scale(self, scales: &[T]) -> Option<Self> {
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

    /// `F X F`: `M`'s off-diagonal from `A_c`'s.
    pub(super) fn congruent(&self, off_diagonal: OffDiagonal<T>) -> OffDiagonal<T> {
        todo!()
    }

    /// `x ← F x`; `A_c⁺ = F M⁺ F` on `A_c`'s range, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        todo!()
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_out_null(&self, x: &mut [T]) {
        todo!()
    }
}
