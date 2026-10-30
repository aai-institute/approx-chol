//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use rayon::prelude::*;

use crate::sgdd::{within_band, Grounding, Sheets};
use crate::storage::{Rows, UpperAdjacency};
use crate::{narrow, widen, Index, Real};

/// `L + diag(s)`, `L`'s off-diagonal nonpositive; a twisted cover's `k⁺—k⁻` edges live in its closure, not its rows.
#[derive(Clone, Debug)]
pub(super) struct Sddm<T, I> {
    adjacency: UpperAdjacency<T, I>,
    closure: Closure<T>,
}

/// What `L` carries beyond its rows: a ground on one sheet or two, or a cover's twists in its place.
#[derive(Clone, Debug)]
enum Closure<T> {
    Folded(Grounding<T>),
    Cover(Grounding<T>),
    /// Each vertex's [`twist`] `k⁺ — k⁻`, zero where both copies float.
    Twisted(Box<[T]>),
}

impl<T: Real> Closure<T> {
    fn grounding(&self) -> Option<&Grounding<T>> {
        match self {
            Self::Folded(grounding) | Self::Cover(grounding) => Some(grounding),
            Self::Twisted(_) => None,
        }
    }
}

impl<T: Real, I: Index> Sddm<T, I> {
    /// Panics unless nonempty, with one surplus per row or one twist per pair.
    fn new(adjacency: UpperAdjacency<T, I>, closure: Closure<T>) -> Self {
        let n = adjacency.n();
        assert!(n > 0, "an SDDM has a vertex");
        match &closure {
            Closure::Folded(grounding) => assert!(grounding.fits(n), "one surplus per row"),
            Closure::Cover(grounding) => {
                assert!(grounding.fits(n), "one surplus per row");
                assert_eq!(n % 2, 0, "every sheet holds the same vertices");
            }
            Closure::Twisted(twists) => assert_eq!(2 * twists.len(), n, "one twist per pair"),
        }
        debug_assert!(adjacency.nonpositive(), "L's off-diagonal is nonpositive");
        Self { adjacency, closure }
    }

    /// Folded `M` on one sheet, or the Gremban cover of signed `M` on two.
    pub(super) fn of_block(
        adjacency: UpperAdjacency<T, I>,
        sheets: Sheets,
        grounding: Grounding<T>,
    ) -> Self {
        match sheets {
            Sheets::One => Self::new(adjacency, Closure::Folded(grounding)),
            Sheets::Two => {
                let n = adjacency.n();
                // The cover acts as `M − diag(s)` on `[z; −z]`.
                let rows = SddmRows::split(sheets, n, n, |mut sums, i| {
                    for (j, &a) in adjacency.upper(i) {
                        sums.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
                    }
                });
                Self::of_rows(rows, sheets, grounding.surplus())
            }
        }
    }

    /// Rows on `sheets`, each ground plus its vertex's `surplus`, one per vertex; on two, a ground within the band [`twists`].
    pub(super) fn of_rows(rows: SddmRows<T, I>, sheets: Sheets, surplus: Option<&[T]>) -> Self {
        let n = rows.sums.len() - 1;
        let adjacency = UpperAdjacency::new(Rows::new(rows.ptrs, rows.indices, rows.values, n));
        let mut ground = rows.ground;
        if let Some(surplus) = surplus {
            assert_eq!(sheets.count() * surplus.len(), n, "one surplus per vertex");
            let copies = (surplus.iter()).flat_map(|&s| std::iter::repeat_n(s, sheets.count()));
            (ground.iter_mut().zip(copies)).for_each(|(g, s)| *g = s + *g);
        }
        let grounding = Grounding::of(ground);
        let closure = match (sheets, grounding.surplus()) {
            (Sheets::One, _) => Closure::Folded(grounding),
            (Sheets::Two, Some(ground)) if twists(ground, &adjacency) => {
                Closure::Twisted(ground.chunks_exact(2).map(|g| twist(g[0], g[1])).collect())
            }
            (Sheets::Two, _) => Closure::Cover(grounding),
        };
        Self::new(adjacency, closure)
    }

    pub(super) fn n(&self) -> usize {
        self.adjacency.n()
    }

    pub(super) fn grounded(&self) -> bool {
        (self.closure.grounding()).is_some_and(Grounding::is_grounded)
    }

    pub(super) fn sheets(&self) -> Sheets {
        match self.closure {
            Closure::Folded(_) => Sheets::One,
            Closure::Cover(_) | Closure::Twisted(_) => Sheets::Two,
        }
    }

    /// Positive Laplacian weights, each edge once with any twists, the ground vertex `n` joined by each surplus.
    pub(super) fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
        let n = self.n();
        let twists = match &self.closure {
            Closure::Twisted(twists) => &twists[..],
            Closure::Folded(_) | Closure::Cover(_) => &[],
        };
        for i in 0..n {
            // Ahead of `k⁺`'s row, where its column sorts: a star merges duplicates in list order.
            let twist = twists.get(i / 2).filter(|&&t| i % 2 == 0 && t > T::zero());
            if let Some(&t) = twist {
                edge(I::from_index(i), I::from_index(i + 1), t);
            }
            for (j, &a) in self.adjacency.upper(i) {
                edge(I::from_index(i), I::from_index(j), -a);
            }
        }
        if let Some(grounding) = self.closure.grounding() {
            for k in 0..n {
                if let Some(s) = grounding.edge(k) {
                    edge(I::from_index(k), I::from_index(n), s);
                }
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

    pub(super) fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>)) {
        add(ColumnSums {
            sums: &mut self.sums,
            touched: &mut self.touched,
        });
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

    /// Repeats the row just ended on the other sheet, each column's sheet bit flipped.
    fn mirror_row(&mut self) {
        let (start, end) = (self.ptrs[self.ptrs.len() - 2], self.indices.len());
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

    /// An SDDM on `n_sheet` vertices per sheet, `row` writing vertex `k`'s row once, `k⁺`'s on a cover, which `k⁻` mirrors; split by vertex across threads, at least `min_len` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        sheets: Sheets,
        n_sheet: usize,
        min_len: usize,
        row: impl Fn(ColumnSums<'_, T, I>, usize) + Sync,
    ) -> Self {
        let n = sheets.count() * n_sheet;
        let rows = (0..n_sheet).into_par_iter().with_min_len(min_len).fold(
            || Self::new(n),
            |mut rows, k| {
                rows.row(|sums| row(sums, k));
                if sheets == Sheets::Two {
                    rows.mirror_row();
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
        rows.unwrap_or_else(|| Self::new(n))
    }

    /// The signed rows the sheets cover, `k⁻`'s mirrors skipped: `j⁺, j⁻` sum `P, N` into `N − P`, and `surplus` plus the ground gains `2 min(P, N)` on both rows, as `P + N − |N − P|` would by cancellation.
    pub(super) fn fold(
        self,
        sheets: Sheets,
        surplus: Option<&[T]>,
    ) -> (UpperAdjacency<T, I>, Grounding<T>) {
        let count = sheets.count();
        let n = (self.sums.len() - 1) / count;
        assert!(
            surplus.is_none_or(|s| s.len() == n),
            "one surplus per vertex"
        );
        let ground = |k: usize| self.ground[count * k];
        let mut surplus: Vec<T> = (0..n)
            .map(|k| surplus.map_or(ground(k), |s| s[k] + ground(k)))
            .collect();
        let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        let two = T::one() + T::one();
        for k in 0..n {
            let mut at = self.ptrs[count * k];
            let columns = &self.indices[at..self.ptrs[count * k + 1]];
            for pair in columns.chunk_by(|a, b| a.index() / count == b.index() / count) {
                let mut sums = [T::zero(); 2];
                for (c, &v) in pair.iter().zip(&self.values[at..]) {
                    sums[c.index() % count] = -v;
                }
                at += pair.len();
                let [same, opposite] = sums;
                let j = pair[0].index() / count;
                let gain = two * same.min(opposite);
                surplus[k] = surplus[k] + gain;
                surplus[j] = surplus[j] + gain;
                if opposite != same {
                    indices.push(I::from_index(j));
                    values.push(opposite - same);
                }
            }
            ptrs.push(indices.len());
        }
        let rows = Rows::new(ptrs, indices, values, n);
        (UpperAdjacency::new(rows), Grounding::of(surplus))
    }
}

/// One open row's sums by column and the columns they touch, borrowed apart from the finished rows so the sums stay in registers.
pub(super) struct ColumnSums<'a, T, I> {
    sums: &'a mut [T],
    touched: &'a mut Vec<I>,
}

impl<T: Real, I: Index> ColumnSums<'_, T, I> {
    /// No zero test on `w`: it costs the exact rows 6-8%, so ending the row drops what a zero addend leaves.
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

/// Whether `ground` is within the band of `ground + Σ|a_ij|`, both `ε`-scaled so neither overflows: a cover's copies then [`twist`] instead of grounding, so its `1/s` common mode cannot swamp `z`; a cover doubles both sums, so `M`'s decide as its own.
fn twists<T: Real, I: Index>(ground: &[T], adjacency: &UpperAdjacency<T, I>) -> bool {
    let epsilon = widen(T::epsilon());
    let total = crate::lane_sum(ground, epsilon);
    within_band::<T>(total, total + adjacency.scaled_abs_total(epsilon))
}

/// The edge `k⁺ — k⁻` that acts on `[z; −z]` as grounds `g⁺, g⁻` do; with `g⁺ ≠ g⁻` no ungrounded pair matches them, and the mean differs by less than the band. Zero where both copies float, else at least `MIN_POSITIVE`, as a share is.
fn twist<T: Real>(g_plus: T, g_minus: T) -> T {
    if g_plus == T::zero() && g_minus == T::zero() {
        return T::zero();
    }
    let quarter = narrow::<T>(0.25);
    (g_plus * quarter + g_minus * quarter).max(T::min_positive_value())
}

/// A negative entry joins its ends on one sheet, a positive one across them.
pub(super) fn crosses<T: Real>(c: T) -> bool {
    c > T::zero()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A frustrated 9-cycle of `2^1020` plus `extra` on vertex 0: `Σ|a_ij|` overflows `f64`, the band `≈ 2^974.5` does not.
    fn cycle_cover(extra: f64) -> Sddm<f64, u32> {
        let n = 9;
        let indices: Vec<u32> = [1, n - 1].into_iter().chain(2..n).collect();
        let ptrs: Vec<usize> = [0]
            .into_iter()
            .chain(2..=n as usize)
            .chain([n as usize])
            .collect();
        let values = vec![2f64.powi(1020); indices.len()];
        let signed = UpperAdjacency::new(Rows::new(ptrs, indices, values, n as usize));
        let surplus = [extra].into_iter().chain([0.0; 8]).collect();
        Sddm::of_block(signed, Sheets::Two, Grounding::of(surplus))
    }

    #[test]
    fn cover_twists_in_band_and_grounds_past_it_where_the_off_diagonal_overflows() {
        assert!(!cycle_cover(2f64.powi(970)).grounded(), "twists in band");
        assert!(cycle_cover(2f64.powi(1000)).grounded(), "grounds past it");
    }
}
