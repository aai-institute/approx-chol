//! How an elimination reads its weights: an SDDM's Laplacian holds magnitudes alone, a signed `M` stores `−a_ij`, its sign bit the entry's.

use crate::{Index, Real};

/// What a weight's sign bit means to a star's merge, sampling and fill.
pub(super) trait Sign<T: Real, I: Index>: Copy {
    /// The weight a pivot's degree, its CDF and its order read.
    fn magnitude(self, w: T) -> T;

    /// Two duplicates as one, and the magnitude their sum cancels.
    fn merge(self, a: T, b: T) -> (T, T);

    /// The fill edge `a — b` sampled at `magnitude`, `hi` its larger end.
    fn fill(self, a: T, b: T, magnitude: T, hi: I) -> T;

    /// The vertex a merge's cancelled magnitude grounds both its ends to; none where nothing cancels.
    fn ground(self) -> Option<I>;

    /// Whether a decoded factor's share `c` is one this sign's elimination stores.
    #[cfg(feature = "serde")]
    fn admits(c: T) -> bool;
}

/// `a + b`, and the magnitude their sum cancels, `|a| + |b| − |a + b|`: `2 min(|a|, |b|)` where their signs differ.
pub(super) fn cancel<T: Real>(a: T, b: T) -> (T, T) {
    let cancelled = match (a < T::zero()) != (b < T::zero()) {
        true => (T::one() + T::one()) * a.abs().min(b.abs()),
        false => T::zero(),
    };
    (a + b, cancelled)
}

/// A Laplacian's weights: every one a magnitude, so nothing cancels.
#[derive(Clone, Copy)]
pub(super) struct Unsigned;

impl<T: Real, I: Index> Sign<T, I> for Unsigned {
    fn magnitude(self, w: T) -> T {
        w
    }

    fn merge(self, a: T, b: T) -> (T, T) {
        (a + b, T::zero())
    }

    fn fill(self, _: T, _: T, magnitude: T, _: I) -> T {
        magnitude
    }

    fn ground(self) -> Option<I> {
        None
    }

    #[cfg(feature = "serde")]
    fn admits(c: T) -> bool {
        c >= T::zero()
    }
}

/// A signed `M`'s weights, grounded at `ground`, the vertex past its slots.
#[derive(Clone, Copy)]
pub(super) struct Signed<I> {
    pub(super) ground: I,
}

impl<T: Real, I: Index> Sign<T, I> for Signed<I> {
    fn magnitude(self, w: T) -> T {
        w.abs()
    }

    fn merge(self, a: T, b: T) -> (T, T) {
        cancel(a, b)
    }

    /// `w_a w_b / d`, its sign `σ_a σ_b`; an edge to the ground is diagonal mass, a magnitude.
    fn fill(self, a: T, b: T, magnitude: T, hi: I) -> T {
        match hi == self.ground || (a < T::zero()) == (b < T::zero()) {
            true => magnitude,
            false => -magnitude,
        }
    }

    fn ground(self) -> Option<I> {
        Some(self.ground)
    }

    /// A share of either sign; `fits` bounds its magnitude.
    #[cfg(feature = "serde")]
    fn admits(_: T) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opposite signs cancel the smaller magnitude twice over; like signs, or a zero, cancel nothing.
    #[test]
    fn cancel_drops_twice_the_smaller_opposite_magnitude() {
        assert_eq!(cancel(2.0, -0.5), (1.5, 1.0));
        assert_eq!(cancel(-0.5, 2.0), (1.5, 1.0));
        assert_eq!(cancel(-1.0, 1.0), (0.0, 2.0));
        assert_eq!(cancel(1.0, 2.0), (3.0, 0.0));
        assert_eq!(cancel(-1.0, -2.0), (-3.0, 0.0));
        assert_eq!(cancel(0.0, -2.0), (-2.0, 0.0));
    }

    /// Fill takes its ends' signs' product, but an edge to the ground stays a magnitude.
    #[test]
    fn signed_fill_is_signed_by_its_ends_but_grounds_as_a_magnitude() {
        let sign = Signed { ground: 9u32 };
        assert_eq!(sign.fill(2.0, 3.0, 0.5, 4), 0.5);
        assert_eq!(sign.fill(-2.0, -3.0, 0.5, 4), 0.5);
        assert_eq!(sign.fill(-2.0, 3.0, 0.5, 4), -0.5);
        assert_eq!(sign.fill(-2.0, 3.0, 0.5, 9), 0.5);
        assert_eq!(Sign::<f64, u32>::merge(sign, 1.0, 1.0), (2.0, 0.0));
        assert_eq!(Sign::<f64, u32>::magnitude(sign, -2.5), 2.5);
    }
}
