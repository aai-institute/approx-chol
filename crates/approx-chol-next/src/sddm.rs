//! The kernel's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use std::borrow::Cow;

use crate::storage::UpperAdjacency;
use crate::Real;

/// `L + diag(s)`, `L`'s off-diagonal nonpositive; borrowed from its part, or owned when derived.
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

    pub(crate) fn n(&self) -> usize {
        self.adjacency.n()
    }

    pub(crate) fn grounded(&self) -> bool {
        self.grounding.is_grounded()
    }

    pub(crate) fn sheets(&self) -> Sheets {
        self.sheets
    }

    pub(crate) fn rank(&self) -> usize {
        rank(self.n(), self.grounded())
    }

    /// The vertex a row covers; sheets are stacked, so a cover's second sheet repeats the first.
    pub(crate) fn base_vertex(&self, row: usize) -> usize {
        row % (self.n() / self.sheets.count())
    }

    /// Positive Laplacian weights, each edge once, the ground vertex `n` joined by each surplus.
    pub(crate) fn grounded_edges(&self) -> impl Iterator<Item = (u32, u32, T)> + '_ {
        std::iter::from_fn(|| -> Option<(u32, u32, T)> { todo!() })
    }
}

/// Pivots a Cholesky over `rows` takes: without a ground one row stays free.
pub(crate) fn rank(rows: usize, grounded: bool) -> usize {
    if grounded {
        rows
    } else {
        rows - 1
    }
}

/// What closes `L` into an SDDM: a surplus with a positive total, or none, leaving it singular.
#[derive(Clone, Debug)]
pub(crate) struct Grounding<T>(Option<Surplus<T>>);

impl<T: Real> Grounding<T> {
    /// Floats exactly when nothing grounds: an all-zero surplus.
    pub(crate) fn of(surplus: Surplus<T>) -> Self {
        Self(surplus.grounds().then_some(surplus))
    }

    /// A proven alias: one vertex anchors instead, whatever surplus rounding left.
    pub(crate) fn floating() -> Self {
        Self(None)
    }

    pub(crate) fn surplus(&self) -> Option<&Surplus<T>> {
        self.0.as_ref()
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.0.is_some()
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.0.as_ref().is_none_or(|surplus| surplus.0.len() == n)
    }
}

/// `s = diag(M) − Σ|offdiag|`, finite and nonnegative.
#[derive(Clone, Debug)]
pub(crate) struct Surplus<T>(Box<[T]>);

impl<T: Real> Surplus<T> {
    /// Panics on a negative or non-finite entry.
    pub(crate) fn new(values: Vec<T>) -> Self {
        todo!()
    }

    pub(crate) fn values(&self) -> &[T] {
        &self.0
    }

    /// Whether any entry is positive.
    pub(crate) fn grounds(&self) -> bool {
        self.0.iter().any(|&s| s > T::zero())
    }
}

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
