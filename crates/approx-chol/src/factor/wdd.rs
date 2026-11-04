//! The matrix factored, typed by the sign of its off-diagonal.

use std::marker::PhantomData;

use crate::storage::{Adjacency, Balance, Grounded, Grounding, Index, Real, WddCsr};

/// An edge's sign: `Unsigned` compiles to the Laplacian kernels, `Signed` keeps positive entries.
pub(super) trait Sign {
    const SIGNED: bool;
}

/// A Z-matrix's: every off-diagonal entry nonpositive.
pub(super) struct Unsigned;

/// A frustrated matrix's: off-diagonals of either sign.
pub(super) struct Signed;

impl Sign for Unsigned {
    const SIGNED: bool = false;
}

impl Sign for Signed {
    const SIGNED: bool = true;
}

/// A WDD matrix with its grounding, its sign a type.
pub(super) struct Wdd<T, I, S> {
    matrix: Grounded<T, WddCsr<T, I>>,
    sign: PhantomData<S>,
}

impl<T: Real, I: Index, S: Sign> Wdd<T, I, S> {
    fn of(matrix: Grounded<T, WddCsr<T, I>>) -> Self {
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

/// A balanced matrix is a Z-matrix, a frustrated one signed.
pub(super) enum AnyWdd<T, I> {
    Unsigned(Wdd<T, I, Unsigned>),
    Signed(Wdd<T, I, Signed>),
}

impl<T: Real, I: Index> AnyWdd<T, I> {
    /// Panics unless nonempty, with one surplus per row.
    pub(super) fn new(matrix: Grounded<T, WddCsr<T, I>>, balance: Balance) -> Self {
        let n = matrix.adjacency.n();
        assert!(n > 0, "M has a vertex");
        assert!(matrix.grounding.fits(n), "one surplus per row");
        match balance {
            Balance::Balanced => {
                debug_assert!(matrix.adjacency.nonpositive(), "a balanced M is a Z-matrix");
                Self::Unsigned(Wdd::of(matrix))
            }
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
