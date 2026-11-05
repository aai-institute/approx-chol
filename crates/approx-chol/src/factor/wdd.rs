//! The matrix factored, typed by the sign of its off-diagonal.

use std::marker::PhantomData;

use crate::storage::{Adjacency, Balance, Grounded, Grounding, Index, Real, WddCsr};

/// An edge's sign: `Unsigned` is zero-sized and compiles to the Laplacian elimination, `Signed` keeps positive entries.
pub(super) trait Sign: Copy + Send + Sync + 'static {
    const SIGNED: bool;

    /// The sign of a fill edge between two neighbors of an eliminated vertex.
    fn fill(self, other: Self) -> Self;

    /// Two entries of one edge as one: like signs add, opposite ones cancel.
    fn merge<T: Real>(a: Entry<Self, T>, b: Entry<Self, T>) -> Merged<Self, T>;
}

/// An entry's sign and its magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Entry<S, T> {
    pub(super) sign: S,
    pub(super) weight: T,
}

/// A merge's entry and what cancelled: `2·min`, surplus at both ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Merged<S, T> {
    pub(super) kept: Entry<S, T>,
    pub(super) cancelled: T,
}

/// A Z-matrix's: every off-diagonal entry nonpositive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Unsigned;

/// A frustrated matrix's: whether the entry is positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Signed(pub(super) bool);

impl Sign for Unsigned {
    const SIGNED: bool = false;

    fn fill(self, _: Self) -> Self {
        Self
    }

    fn merge<T: Real>(a: Entry<Self, T>, b: Entry<Self, T>) -> Merged<Self, T> {
        Merged {
            kept: Entry {
                sign: Self,
                weight: a.weight + b.weight,
            },
            cancelled: T::zero(),
        }
    }
}

impl Sign for Signed {
    const SIGNED: bool = true;

    /// `−a_vi a_vj / d` is positive exactly when the two entries differ in sign.
    fn fill(self, other: Self) -> Self {
        Self(self.0 != other.0)
    }

    fn merge<T: Real>(a: Entry<Self, T>, b: Entry<Self, T>) -> Merged<Self, T> {
        if a.sign == b.sign {
            let weight = a.weight + b.weight;
            let kept = Entry { weight, ..a };
            return Merged {
                kept,
                cancelled: T::zero(),
            };
        }
        let (larger, smaller) = match a.weight >= b.weight {
            true => (a, b),
            false => (b, a),
        };
        Merged {
            kept: Entry {
                sign: larger.sign,
                weight: larger.weight - smaller.weight,
            },
            cancelled: smaller.weight + smaller.weight,
        }
    }
}

/// A WDD matrix with its grounding, its sign a type.
pub(super) struct Wdd<T, I, S> {
    matrix: Grounded<T, WddCsr<T, I>>,
    sign: PhantomData<S>,
}

impl<T: Real, I: Index, S: Sign> Wdd<T, I, S> {
    /// Panics unless nonempty, with one surplus per row.
    fn of(matrix: Grounded<T, WddCsr<T, I>>) -> Self {
        let n = matrix.adjacency.n();
        assert!(n > 0, "M has a vertex");
        assert!(matrix.grounding.fits(n), "one surplus per row");
        Self {
            matrix,
            sign: PhantomData,
        }
    }

    pub(super) fn n(&self) -> usize {
        self.matrix.adjacency.n()
    }

    pub(super) fn adjacency(&self) -> &WddCsr<T, I> {
        &self.matrix.adjacency
    }

    pub(super) fn grounding(&self) -> &Grounding<T> {
        &self.matrix.grounding
    }

    /// Singular exactly when balanced and ungrounded: its null vector is `1`, the fold's `F·1` in `A`.
    pub(super) fn floats(&self) -> bool {
        !S::SIGNED && !self.matrix.grounding.is_grounded()
    }
}

impl<T: Real, I: Index> Wdd<T, I, Unsigned> {
    pub(super) fn z_matrix(matrix: Grounded<T, WddCsr<T, I>>) -> Self {
        debug_assert!(matrix.adjacency.nonpositive(), "a Z-matrix");
        Self::of(matrix)
    }
}

/// A balanced matrix is a Z-matrix, a frustrated one signed.
pub(super) enum AnyWdd<T, I> {
    Unsigned(Wdd<T, I, Unsigned>),
    Signed(Wdd<T, I, Signed>),
}

impl<T: Real, I: Index> AnyWdd<T, I> {
    pub(super) fn new(matrix: Grounded<T, WddCsr<T, I>>, balance: Balance) -> Self {
        match balance {
            Balance::Balanced => Self::Unsigned(Wdd::z_matrix(matrix)),
            Balance::Frustrated => Self::Signed(Wdd::of(matrix)),
        }
    }

    pub(super) fn n(&self) -> usize {
        match self {
            Self::Unsigned(wdd) => wdd.n(),
            Self::Signed(wdd) => wdd.n(),
        }
    }

    pub(super) fn floats(&self) -> bool {
        match self {
            Self::Unsigned(wdd) => wdd.floats(),
            Self::Signed(wdd) => wdd.floats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(positive: bool, weight: f64) -> Entry<Signed, f64> {
        Entry {
            sign: Signed(positive),
            weight,
        }
    }

    #[test]
    fn like_signs_add_and_opposite_ones_cancel_twice_the_smaller() {
        let added = Signed::merge(entry(true, 2.0), entry(true, 3.0));
        assert_eq!(added.kept, entry(true, 5.0));
        assert_eq!(added.cancelled, 0.0);
        let cancelled = Signed::merge(entry(true, 2.0), entry(false, 3.0));
        assert_eq!(cancelled.kept, entry(false, 1.0));
        assert_eq!(cancelled.cancelled, 4.0);
    }

    #[test]
    fn fill_is_positive_exactly_between_unlike_signs() {
        assert_eq!(Signed(true).fill(Signed(false)), Signed(true));
        assert_eq!(Signed(false).fill(Signed(false)), Signed(false));
        assert_eq!(Signed(true).fill(Signed(true)), Signed(false));
    }
}
