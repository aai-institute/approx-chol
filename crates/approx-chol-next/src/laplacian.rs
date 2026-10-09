//! The kernel's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use crate::storage::UpperAdjacency;
use crate::Real;

/// `L + diag(s)` in general storage; built only through [`new`](Self::new), which checks it.
#[derive(Clone, Debug)]
pub(crate) struct Sddm<T> {
    adjacency: UpperAdjacency<T>,
    ground: Ground<T>,
    sheets: Sheets,
}

impl<T: Real> Sddm<T> {
    /// Panics unless nonempty, with one surplus per vertex and the vertices split evenly into sheets.
    pub(crate) fn new(adjacency: UpperAdjacency<T>, ground: Ground<T>, sheets: Sheets) -> Self {
        let n = adjacency.n();
        assert!(n > 0, "an SDDM has a vertex");
        assert!(ground.fits(n), "one surplus per vertex");
        assert_eq!(n % sheets.count(), 0, "every sheet holds the same vertices");
        Self {
            adjacency,
            ground,
            sheets,
        }
    }

    pub(crate) fn adjacency(&self) -> &UpperAdjacency<T> {
        &self.adjacency
    }

    pub(crate) fn ground(&self) -> &Ground<T> {
        &self.ground
    }

    pub(crate) fn sheets(&self) -> Sheets {
        self.sheets
    }

    pub(crate) fn eliminated(&self) -> usize {
        self.ground.eliminated(self.adjacency.n())
    }

    /// The vertex a row copies; sheets are stacked, so a cover's second sheet repeats the first.
    pub(crate) fn copied(&self, row: usize) -> usize {
        row % (self.adjacency.n() / self.sheets.count())
    }
}

/// `M`'s surplus `s = diag(M) − Σ|offdiag|`; without one `M` is a Laplacian and singular.
#[derive(Clone, Debug)]
pub(crate) struct Ground<T>(Option<Surplus<T>>);

impl<T: Real> Ground<T> {
    /// All-zero surplus floats; panics on a negative or non-finite entry.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        todo!()
    }

    /// A proven alias anchors one vertex instead, whatever surplus rounding left.
    pub(crate) fn floating() -> Self {
        Self(None)
    }

    pub(crate) fn surplus(&self) -> Option<&[T]> {
        self.0.as_ref().map(|surplus| &*surplus.0)
    }

    /// Whether it grounds exactly `n` vertices.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.surplus().is_none_or(|s| s.len() == n)
    }

    /// Vertices a Cholesky over `n` solves for: a floating one leaves one free.
    pub(crate) fn eliminated(&self, n: usize) -> usize {
        match self.0 {
            Some(_) => n,
            None => n - 1,
        }
    }
}

/// Finite and nonnegative with a positive total.
#[derive(Clone, Debug)]
struct Surplus<T>(Box<[T]>);

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

    /// Half the sheets' difference back into `x`.
    pub(crate) fn fold<T: Real>(self, rows: &[T], x: &mut [T]) {
        todo!()
    }
}
