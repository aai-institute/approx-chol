//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use std::borrow::Cow;

use super::clique_tree::Fill;
use crate::sgdd::{Balance, Grounding};
use crate::storage::{Rows, UpperAdjacency};
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
    let mut starts = vec![0; n + 1];
    // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
    for chunk in &chunks {
        for &(lo, _, _) in chunk {
            starts[lo as usize + 1] += 1;
        }
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut next = starts[..n].to_vec();
    let mut by_row = vec![(0, T::zero()); starts[n]];
    for chunk in chunks {
        for (lo, hi, w) in chunk {
            by_row[next[lo as usize]] = (hi as usize, w);
            next[lo as usize] += 1;
        }
    }
    let mut rows = RowSums::new(n);
    for span in starts.windows(2) {
        for &(hi, w) in &by_row[span[0]..span[1]] {
            rows.add(hi, w);
        }
        rows.end_row();
    }
    rows.finish()
}

/// `S` row by row, each column's addends summed in arrival order and stored as `−w` by column; column `n` is the row's ground weight.
pub(super) struct RowSums<T> {
    sums: Vec<T>,
    touched: Vec<u32>,
    ptrs: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<T>,
    ground: Vec<T>,
}

impl<T: Real> RowSums<T> {
    pub(super) fn new(n: usize) -> Self {
        Self {
            sums: vec![T::zero(); n + 1],
            touched: Vec::new(),
            ptrs: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
            ground: Vec::new(),
        }
    }

    /// A zero addend is dropped, as a stored zero couples nothing.
    pub(super) fn add(&mut self, column: usize, w: T) {
        if w == T::zero() {
            return;
        }
        if self.sums[column] == T::zero() {
            self.touched.push(column as u32);
        }
        self.sums[column] = self.sums[column] + w;
    }

    pub(super) fn end_row(&mut self) {
        let n = self.sums.len() - 1;
        self.touched.sort_unstable();
        let ground = match self.touched.last() {
            Some(&j) if j as usize == n => {
                self.touched.pop();
                std::mem::replace(&mut self.sums[n], T::zero())
            }
            _ => T::zero(),
        };
        self.ground.push(ground);
        for &j in &self.touched {
            self.indices.push(j);
            self.values
                .push(-std::mem::replace(&mut self.sums[j as usize], T::zero()));
        }
        self.touched.clear();
        self.ptrs.push(self.indices.len());
    }

    pub(super) fn finish(self) -> (UpperAdjacency<T>, Vec<T>) {
        let n = self.ground.len();
        let rows = Rows::new(self.ptrs, self.indices, self.values, n);
        (UpperAdjacency::new(rows), self.ground)
    }
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
