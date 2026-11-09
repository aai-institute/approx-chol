//! How an elimination reads its weights: an SDDM's Laplacian holds magnitudes alone.

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
}
