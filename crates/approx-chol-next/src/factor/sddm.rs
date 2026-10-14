//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::sgdd::Grounding;
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
        let n = signed.n();
        let mut rows = SddmRows::new(2 * n);
        for i in 0..n {
            for (j, &a) in signed.upper(i) {
                rows.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
            }
            if let Some(surplus) = grounding.surplus() {
                rows.add(2 * n, surplus[i]);
            }
            rows.end_row();
            rows.mirror_row();
        }
        let (adjacency, ground) = rows.finish();
        let grounding = Grounding::of(ground);
        Sddm::new(Cow::Owned(adjacency), Cow::Owned(grounding), Sheets::Two)
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

/// An SDDM row by row, each column's addends summed in arrival order and stored as `−w` by column; column `n` is the row's ground weight.
pub(super) struct SddmRows<T> {
    sums: Vec<T>,
    touched: Vec<u32>,
    ptrs: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<T>,
    ground: Vec<T>,
}

impl<T: Real> SddmRows<T> {
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

    /// No zero test on `w`: it costs the exact rows 6-8%, so `end_row` drops what a zero addend leaves.
    pub(super) fn add(&mut self, column: usize, w: T) {
        debug_assert!(
            w >= T::zero(),
            "a column reads zero only before its first nonzero addend"
        );
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
            let sum = std::mem::replace(&mut self.sums[j as usize], T::zero());
            // A stored zero couples nothing, and a column pushed twice reads zero the second time.
            if sum != T::zero() {
                self.indices.push(j);
                self.values.push(-sum);
            }
        }
        self.touched.clear();
        self.ptrs.push(self.indices.len());
    }

    /// Repeats the row just ended on the other sheet, each column's sheet bit flipped.
    pub(super) fn mirror_row(&mut self) {
        let (start, end) = (self.ptrs[self.ptrs.len() - 2], self.indices.len());
        self.ground.push(*self.ground.last().expect("a row ended"));
        for i in start..end {
            self.indices.push(self.indices[i] ^ 1);
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

    /// Rows `0..n_rows` of an SDDM on `n` vertices, `row` writing every row it owns; split by row across threads, at least `min_rows` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        n: usize,
        n_rows: usize,
        min_rows: usize,
        row: impl Fn(&mut Self, usize) + Sync,
    ) -> (UpperAdjacency<T>, Vec<T>) {
        let rows = (0..n_rows).into_par_iter().with_min_len(min_rows).fold(
            || Self::new(n),
            |mut rows, k| {
                row(&mut rows, k);
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

    pub(super) fn finish(self) -> (UpperAdjacency<T>, Vec<T>) {
        let n = self.ground.len();
        let rows = Rows::new(self.ptrs, self.indices, self.values, n);
        (UpperAdjacency::new(rows), self.ground)
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
