use super::scaling::NotScalable;
use super::signature::Signature;
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
    /// `S R`; unscaled signs stay unmaterialized, so an SDDM input keeps `F = I`.
    pub(super) fn new(signature: Signature, scales: Option<Box<[T]>>) -> Self {
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

    /// `F X F`: `M`'s off-diagonal from `A_c`'s, where an entry can leave the finite range.
    pub(super) fn transform(
        &self,
        off_diagonal: OffDiagonal<T>,
    ) -> Result<OffDiagonal<T>, NotScalable> {
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
