//! The kernel's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use std::borrow::Cow;

use crate::storage::UpperAdjacency;
use crate::Real;

/// `L + diag(s)`, borrowed from its part or owned when a cover or a complement built it.
#[derive(Clone, Debug)]
pub(crate) struct Sddm<'a, T: Real> {
    adjacency: Cow<'a, UpperAdjacency<T>>,
    grounding: Cow<'a, Grounding<T>>,
    sheets: Sheets,
}

impl<'a, T: Real> Sddm<'a, T> {
    /// Panics unless nonempty, with one surplus per row and the rows split evenly into sheets.
    pub(crate) fn new(
        adjacency: Cow<'a, UpperAdjacency<T>>,
        grounding: Cow<'a, Grounding<T>>,
        sheets: Sheets,
    ) -> Self {
        let n = adjacency.n();
        assert!(n > 0, "an SDDM has a vertex");
        assert!(grounding.fits(n), "one surplus per row");
        assert_eq!(n % sheets.count(), 0, "every sheet holds the same vertices");
        Self {
            adjacency,
            grounding,
            sheets,
        }
    }

    /// The Gremban cover of signed `M`: a Laplacian on two sheets that acts as `M` on `[z; −z]`.
    pub(crate) fn cover(signed: &UpperAdjacency<T>, grounding: &Grounding<T>) -> Sddm<'static, T> {
        todo!()
    }

    pub(crate) fn adjacency(&self) -> &UpperAdjacency<T> {
        &self.adjacency
    }

    pub(crate) fn grounding(&self) -> &Grounding<T> {
        &self.grounding
    }

    pub(crate) fn sheets(&self) -> Sheets {
        self.sheets
    }

    pub(crate) fn rank(&self) -> usize {
        self.grounding.rank(self.adjacency.n())
    }

    /// The vertex a row covers; sheets are stacked, so a cover's second sheet repeats the first.
    pub(crate) fn base_vertex(&self, row: usize) -> usize {
        row % (self.adjacency.n() / self.sheets.count())
    }
}

/// `M`'s surplus `s = diag(M) − Σ|offdiag|`, or none: a floating `M` is a Laplacian and singular.
#[derive(Clone, Debug)]
pub(crate) enum Grounding<T> {
    Grounded(Surplus<T>),
    /// A proven alias anchors one vertex instead, whatever surplus rounding left.
    Floating,
}

impl<T: Real> Grounding<T> {
    /// All-zero surplus floats; panics on a negative or non-finite entry.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        todo!()
    }

    pub(crate) fn surplus(&self) -> Option<&[T]> {
        match self {
            Self::Grounded(surplus) => Some(&surplus.0),
            Self::Floating => None,
        }
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.surplus().is_none_or(|s| s.len() == n)
    }

    /// Pivots a Cholesky over `n` rows takes: a floating one leaves one free.
    pub(crate) fn rank(&self, n: usize) -> usize {
        match self {
            Self::Grounded(_) => n,
            Self::Floating => n - 1,
        }
    }
}

/// Finite and nonnegative with a positive total; built only by [`Grounding::of`].
#[derive(Clone, Debug)]
pub(crate) struct Surplus<T>(Box<[T]>);

/// How many copies of each vertex a matrix holds: two for a Gremban cover, stacked by sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sheets {
    One,
    Two,
}

impl Sheets {
    pub(crate) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }

    /// `[x; −x]` into `rows`: the antisymmetric lift a cover solves on.
    pub(crate) fn lift<T: Real>(self, x: &[T], rows: &mut [T]) {
        todo!()
    }

    /// Half the sheets' difference back into `x`, undoing [`lift`](Self::lift).
    pub(crate) fn descend<T: Real>(self, rows: &[T], x: &mut [T]) {
        todo!()
    }
}
