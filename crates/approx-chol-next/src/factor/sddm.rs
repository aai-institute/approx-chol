//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use std::borrow::Cow;

use super::clique_tree::Fill;
use crate::sgdd::{Balance, Grounding};
use crate::storage::UpperAdjacency;
use crate::Real;

/// `L + diag(s)`, `L`'s off-diagonal nonpositive; borrowed from its component, or owned when derived.
#[derive(Clone, Debug)]
pub(super) struct Sddm<'a, T: Real> {
    adjacency: Cow<'a, UpperAdjacency<T>>,
    grounding: Cow<'a, Grounding<T>>,
    sheets: Sheets,
}

impl<'a, T: Real> Sddm<'a, T> {
    /// Panics unless nonempty, with one surplus per row and the rows split evenly into sheets.
    pub(super) fn new(
        adjacency: Cow<'a, UpperAdjacency<T>>,
        grounding: Cow<'a, Grounding<T>>,
        sheets: Sheets,
    ) -> Self {
        let n = adjacency.n();
        assert!(n > 0, "an SDDM has a vertex");
        assert!(grounding.fits(n), "one surplus per row");
        assert_eq!(n % sheets.count(), 0, "every sheet holds the same vertices");
        debug_assert!(adjacency.nonpositive(), "L's off-diagonal is nonpositive");
        Self {
            adjacency,
            grounding,
            sheets,
        }
    }

    /// The Gremban cover of signed `M`: a Laplacian on two sheets that acts as `M` on `[z; −z]`.
    pub(super) fn cover(signed: &UpperAdjacency<T>, grounding: &Grounding<T>) -> Sddm<'static, T> {
        todo!()
    }

    pub(super) fn n(&self) -> usize {
        self.adjacency.n()
    }

    pub(super) fn grounded(&self) -> bool {
        self.grounding.is_grounded()
    }

    pub(super) fn sheets(&self) -> Sheets {
        self.sheets
    }

    /// Positive Laplacian weights, each edge once, the ground vertex `n` joined by each surplus.
    pub(super) fn grounded_edges(&self) -> impl Iterator<Item = (u32, u32, T)> + '_ {
        let n = self.n();
        let laplacian = (0..n).flat_map(move |i| {
            let row = self.adjacency.upper(i);
            row.map(move |(j, &a)| (i as u32, j as u32, -a))
        });
        let surplus = self.grounding.surplus().unwrap_or_default();
        let ground = surplus.iter().enumerate().filter(|(_, &s)| s > T::zero());
        laplacian.chain(ground.map(move |(k, &s)| (k as u32, n as u32, s)))
    }
}

/// Counting-sorts chunks unflattened, merges each row's duplicates, stores `−w`; an edge to `n` is `lo`'s ground weight.
pub(super) fn from_fill<T: Real>(n: usize, chunks: Vec<Fill<T>>) -> (UpperAdjacency<T>, Vec<T>) {
    todo!()
}

/// How many copies of each vertex a matrix holds: two for a Gremban cover, stacked by sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Sheets {
    One,
    Two,
}

impl Sheets {
    /// A frustrated `M` factors on a cover's two sheets.
    pub(super) fn of<T: Real>(balance: &Balance<T>) -> Self {
        match balance {
            Balance::Balanced(_) => Self::One,
            Balance::Frustrated(_) => Self::Two,
        }
    }

    pub(super) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }

    /// `[x; −x]` into `rows`: the antisymmetric lift a cover solves on.
    pub(super) fn lift<T: Real>(self, x: &[T], rows: &mut [T]) {
        todo!()
    }

    /// Half the sheets' difference back into `x`, undoing [`lift`](Self::lift).
    pub(super) fn descend<T: Real>(self, rows: &[T], x: &mut [T]) {
        todo!()
    }
}
