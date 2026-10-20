//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use rayon::prelude::*;

use crate::sgdd::{within_band, Grounding};
use crate::storage::{Rows, UpperAdjacency};
use crate::{widen, Index, Real};

/// `L + diag(s)`, `L`'s off-diagonal nonpositive.
#[derive(Clone, Debug)]
pub(super) struct Sddm<T, I> {
    adjacency: UpperAdjacency<T, I>,
    grounding: Grounding<T>,
    sheets: Sheets,
}

impl<T: Real, I: Index> Sddm<T, I> {
    /// Panics unless nonempty, with one surplus per row and the rows split evenly into sheets.
    pub(super) fn new(
        adjacency: UpperAdjacency<T, I>,
        grounding: Grounding<T>,
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

    /// The Gremban cover of signed `M`: a Laplacian on two sheets that acts as `M` on `[z; −z]`; a surplus within the band joins `i`'s copies by `s_i/2` instead of grounding them, so the sheets' `1/s` common mode cannot swamp `z`.
    pub(super) fn cover(signed: UpperAdjacency<T, I>, grounding: Grounding<T>) -> Self {
        let n = signed.n();
        let twisted = grounding.surplus().is_some_and(|s| {
            let total = crate::compensated_sum(s.iter().map(|&s| widen(s)));
            within_band::<T>(total, total + signed.abs_total())
        });
        let twists = if twisted { n } else { 0 };
        let mut rows = SddmRows::with_capacity(2 * n, 2 * signed.nnz() + twists);
        let half = T::one() / (T::one() + T::one());
        for i in 0..n {
            let surplus = grounding.edge(i);
            let twist = surplus.filter(|_| twisted).map(|s| s * half);
            if let Some(t) = twist {
                rows.add(cover_vertex(i, 1), t);
            }
            for (j, &a) in signed.upper(i) {
                rows.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
            }
            if let Some(s) = surplus.filter(|_| !twisted) {
                rows.add(2 * n, s);
            }
            rows.end_row();
            rows.mirror_row(usize::from(twist.is_some()));
        }
        let (adjacency, ground) = rows.finish();
        let grounding = Grounding::of(ground);
        Self::new(adjacency, grounding, Sheets::Two)
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
    pub(super) fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
        let n = self.n();
        for i in 0..n {
            for (j, &a) in self.adjacency.upper(i) {
                edge(I::from_index(i), I::from_index(j), -a);
            }
        }
        for k in 0..n {
            if let Some(s) = self.grounding.edge(k) {
                edge(I::from_index(k), I::from_index(n), s);
            }
        }
    }
}

/// An SDDM row by row, each column's addends summed in arrival order and stored as `−w` by column; column `n` is the row's ground weight.
pub(super) struct SddmRows<T, I> {
    sums: Vec<T>,
    touched: Vec<I>,
    ptrs: Vec<usize>,
    indices: Vec<I>,
    values: Vec<T>,
    ground: Vec<T>,
}

impl<T: Real, I: Index> SddmRows<T, I> {
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

    pub(super) fn add(&mut self, column: usize, w: T) {
        self.row().add(column, w);
    }

    /// `new` with room for exactly `n` rows of `entries` in all, when the caller knows them.
    fn with_capacity(n: usize, entries: usize) -> Self {
        let mut rows = Self::new(n);
        rows.ptrs.reserve_exact(n);
        rows.ground.reserve_exact(n);
        rows.indices.reserve_exact(entries);
        rows.values.reserve_exact(entries);
        rows
    }

    fn row(&mut self) -> Row<'_, T, I> {
        Row {
            sums: &mut self.sums,
            touched: &mut self.touched,
        }
    }

    pub(super) fn end_row(&mut self) {
        let n = self.sums.len() - 1;
        self.touched.sort_unstable();
        let ground = match self.touched.last() {
            Some(&j) if j.index() == n => {
                self.touched.pop();
                std::mem::replace(&mut self.sums[n], T::zero())
            }
            _ => T::zero(),
        };
        self.ground.push(ground);
        for &j in &self.touched {
            let sum = std::mem::replace(&mut self.sums[j.index()], T::zero());
            // A stored zero couples nothing, and a column pushed twice reads zero the second time.
            if sum != T::zero() {
                self.indices.push(j);
                self.values.push(-sum);
            }
        }
        self.touched.clear();
        self.ptrs.push(self.indices.len());
    }

    /// Repeats the row just ended on the other sheet past its first `twists` entries, each column's sheet bit flipped.
    fn mirror_row(&mut self, twists: usize) {
        let (start, end) = (self.ptrs[self.ptrs.len() - 2] + twists, self.indices.len());
        self.ground.push(*self.ground.last().expect("a row ended"));
        for i in start..end {
            self.indices.push(self.indices[i] ^ I::one());
            self.values.push(self.values[i]);
        }
        for i in end + 1..self.indices.len() {
            // A pair `2j, 2j + 1` arrives flipped, so only neighbors swap.
            if self.indices[i - 1] > self.indices[i] {
                self.indices.swap(i - 1, i);
                self.values.swap(i - 1, i);
            }
        }
        self.ptrs.push(self.indices.len());
    }

    /// An SDDM on `n_kept` vertices per sheet, `row` writing kept vertex `k`'s row on each sheet; split by vertex across threads, at least `min_kept` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        sheets: Sheets,
        n_kept: usize,
        min_kept: usize,
        row: impl Fn(Row<'_, T, I>, usize) + Sync,
    ) -> (UpperAdjacency<T, I>, Vec<T>) {
        let n = sheets.count() * n_kept;
        let rows = (0..n_kept).into_par_iter().with_min_len(min_kept).fold(
            || Self::new(n),
            |mut rows, k| {
                row(rows.row(), k);
                rows.end_row();
                if sheets == Sheets::Two {
                    rows.mirror_row(0);
                }
                rows
            },
        );
        let rows = rows.reduce_with(|mut rows, part| {
            let offset = rows.indices.len();
            rows.ptrs.extend(part.ptrs[1..].iter().map(|&p| p + offset));
            rows.indices.extend(part.indices);
            rows.values.extend(part.values);
            rows.ground.extend(part.ground);
            rows
        });
        rows.unwrap_or_else(|| Self::new(n)).finish()
    }

    pub(super) fn finish(self) -> (UpperAdjacency<T, I>, Vec<T>) {
        let n = self.ground.len();
        let rows = Rows::new(self.ptrs, self.indices, self.values, n);
        (UpperAdjacency::new(rows), self.ground)
    }
}

/// One row's open column sums, borrowed apart from the finished rows so the sums stay in registers.
pub(super) struct Row<'a, T, I> {
    sums: &'a mut [T],
    touched: &'a mut Vec<I>,
}

impl<T: Real, I: Index> Row<'_, T, I> {
    /// No zero test on `w`: it costs the exact rows 6-8%, so `end_row` drops what a zero addend leaves.
    pub(super) fn add(&mut self, column: usize, w: T) {
        debug_assert!(
            w >= T::zero(),
            "a column reads zero only before its first nonzero addend"
        );
        if self.sums[column] == T::zero() {
            self.touched.push(I::from_index(column));
        }
        self.sums[column] = self.sums[column] + w;
    }
}

/// A negative entry joins its ends on one sheet, a positive one across them.
pub(super) fn crosses<T: Real>(c: T) -> bool {
    c > T::zero()
}

/// How many copies of each vertex a matrix holds: two for a Gremban cover, interleaved by vertex.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Sheets {
    One,
    Two,
}

impl Sheets {
    pub(super) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// Vertex `k`'s copy on `sheet` of a cover: sheets interleave, so row `k⁻` mirrors its `k⁺` by flipping each column's low bit.
pub(super) fn cover_vertex(k: usize, sheet: usize) -> usize {
    2 * k + sheet
}

/// `x` on the first sheet, `−x` on the second: the antisymmetric lift a cover solves on.
pub(super) fn lift<T: Real>(x: &[T], rows: &mut [T]) {
    (rows.chunks_exact_mut(2).zip(x)).for_each(|(r, &v)| {
        r[0] = v;
        r[1] = -v;
    });
}

/// Half the sheets' difference back into `x`, undoing [`lift`].
pub(super) fn descend<T: Real>(rows: &[T], x: &mut [T]) {
    let two = T::one() + T::one();
    (x.iter_mut().zip(rows.chunks_exact(2))).for_each(|(x, r)| *x = (r[0] - r[1]) / two);
}
