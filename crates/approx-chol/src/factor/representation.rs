//! What the approximate elimination factors: a Z-matrix and its slot map, on one sheet or a Gremban cover's two.

use rayon::prelude::*;

use super::wdd::{Unsigned, Wdd};
use crate::storage::{
    lane_sum, narrow, widen, Adjacency, Gauge, Grounded, Grounding, Index, Real, Rows, Sheets,
    WddCsr,
};

/// The elimination's slots over `M`'s `n` vertices: `n` floating, `n + 1` with the ground `g` appended, `2n` on a cover's sheets, `g` after them when grounded.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    Floating,
    Grounded,
    Cover { grounded: bool },
}

impl Layout {
    pub(super) fn of(sheets: Sheets, grounded: bool) -> Self {
        match (sheets, grounded) {
            (Sheets::One, false) => Self::Floating,
            (Sheets::One, true) => Self::Grounded,
            (Sheets::Two, grounded) => Self::Cover { grounded },
        }
    }

    pub(super) fn grounded(self) -> bool {
        matches!(self, Self::Grounded | Self::Cover { grounded: true })
    }

    pub(super) fn sheets(self) -> Sheets {
        match self {
            Self::Floating | Self::Grounded => Sheets::One,
            Self::Cover { .. } => Sheets::Two,
        }
    }

    fn slots(self, n: usize) -> usize {
        self.sheets().count() * n + usize::from(self.grounded())
    }

    /// `M`'s vertices over `slots`, if they fit this layout.
    #[cfg(feature = "serde")]
    pub(super) fn vertices(self, slots: usize) -> Option<usize> {
        let sheets = slots.checked_sub(usize::from(self.grounded()))?;
        let count = self.sheets().count();
        (sheets % count == 0 && sheets > 0).then_some(sheets / count)
    }

    /// The slot the solve pins: the ground, else the largest `|f|` on the first sheet, so the gauge's rounding is not amplified by `1/f`.
    pub(super) fn pinned(self, n: usize, gauge: Gauge) -> usize {
        match (self.grounded(), self.sheets()) {
            (true, _) => self.slots(n) - 1,
            (false, Sheets::One) => gauge.peak,
            (false, Sheets::Two) => cover_vertex(gauge.peak, 0),
        }
    }

    /// Scratch beside `x`: none floating, else every slot.
    pub(super) fn scratch_len(self, slots: usize) -> usize {
        match self {
            Self::Floating => 0,
            Self::Grounded | Self::Cover { .. } => slots,
        }
    }

    /// Lifts `x` onto the slots, `substitute`s there, and descends; the pinned ground reads zero, so its right-hand side is never read.
    pub(super) fn solve<T: Real>(
        self,
        x: &mut [T],
        scratch: &mut [T],
        slots: usize,
        substitute: impl FnOnce(&mut [T]),
    ) {
        match self {
            Self::Floating => substitute(x),
            Self::Grounded => {
                let rows = &mut scratch[..slots];
                rows[..x.len()].copy_from_slice(x);
                substitute(rows);
                x.copy_from_slice(&rows[..x.len()]);
            }
            Self::Cover { .. } => {
                let rows = &mut scratch[..slots];
                ascend(x, rows);
                substitute(rows);
                descend(rows, x);
            }
        }
    }
}

/// A Z-matrix with its slot map, only built together: `M` folded, or its Gremban cover, grounded or joined across its sheets by surplus edges.
pub(super) struct Represented<T, I> {
    matrix: Wdd<T, I, Unsigned>,
    layout: Layout,
}

impl<T: Real, I: Index> Represented<T, I> {
    /// Folded `M` on one sheet, or the Gremban cover of signed `M` on two.
    pub(super) fn sampled(
        adjacency: WddCsr<T, I>,
        sheets: Sheets,
        grounding: Grounding<T>,
    ) -> Self {
        match sheets {
            Sheets::One => Self::folded(Grounded {
                adjacency,
                grounding,
            }),
            Sheets::Two => {
                let n = adjacency.n();
                // The cover acts as `M − diag(s)` on `[z; −z]`.
                let rows = SddmRows::split(sheets, n, n, |mut sums, i| {
                    for (j, &a) in adjacency.upper(i) {
                        sums.add(cover_neighbor(j, 0, crosses(a)), a.abs());
                    }
                });
                Self::from_rows(rows, sheets, grounding.surplus())
            }
        }
    }

    /// Rows on `sheets`, each ground plus its vertex's `surplus`, one per vertex.
    pub(super) fn from_rows(rows: SddmRows<T, I>, sheets: Sheets, surplus: Option<&[T]>) -> Self {
        let n = rows.sums.len() - 1;
        let mut ground = rows.ground;
        if let Some(surplus) = surplus {
            assert_eq!(sheets.count() * surplus.len(), n, "one surplus per vertex");
            let copies = (surplus.iter()).flat_map(|&s| std::iter::repeat_n(s, sheets.count()));
            (ground.iter_mut().zip(copies)).for_each(|(g, s)| *g = s + *g);
        }
        let rows = Rows::new(rows.ptrs, rows.indices, rows.values, n);
        match sheets {
            Sheets::One => Self::folded(Grounded {
                adjacency: WddCsr::new(rows),
                grounding: Grounding::of(ground),
            }),
            Sheets::Two => Self::cover(rows, ground),
        }
    }

    fn folded(matrix: Grounded<T, WddCsr<T, I>>) -> Self {
        let layout = Layout::of(Sheets::One, matrix.grounding.is_grounded());
        Self::new(matrix, layout)
    }

    /// The one place a cover takes its surplus: as the ground `g`, or, within [`surplus_edges`]' band, as an edge [`twist`] `k⁺ — k⁻` per vertex, first in `k⁺`'s row.
    fn cover(rows: Rows<T, I>, ground: Vec<T>) -> Self {
        let adjacency = WddCsr::new(rows);
        assert_eq!(adjacency.n() % 2, 0, "every sheet holds the same vertices");
        let grounding = Grounding::of(ground);
        let matrix = match grounding.surplus() {
            Some(ground) if surplus_edges(ground, &adjacency) => {
                let twists = ground.chunks_exact(2).map(|g| twist(g[0], g[1]));
                Grounded {
                    adjacency: first_in_row(adjacency, twists),
                    grounding: Grounding::ungrounded(),
                }
            }
            _ => Grounded {
                adjacency,
                grounding,
            },
        };
        let layout = Layout::of(Sheets::Two, matrix.grounding.is_grounded());
        Self::new(matrix, layout)
    }

    /// Panics unless nonempty, with one surplus per row.
    fn new(matrix: Grounded<T, WddCsr<T, I>>, layout: Layout) -> Self {
        assert!(matrix.adjacency.n() > 0, "an SDDM has a vertex");
        Self {
            matrix: Wdd::z_matrix(matrix),
            layout,
        }
    }

    /// Its slots, the ground excluded.
    pub(super) fn n(&self) -> usize {
        self.matrix.n()
    }

    pub(super) fn layout(&self) -> Layout {
        self.layout
    }

    /// Positive Laplacian weights, each edge once, the ground vertex `n` joined by each surplus.
    pub(super) fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
        let n = self.n();
        for i in 0..n {
            for (j, &a) in self.matrix.adjacency().upper(i) {
                edge(I::from_index(i), I::from_index(j), -a);
            }
        }
        for k in 0..n {
            if let Some(s) = self.matrix.grounding().edge(k) {
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

    /// The signed rows the sheets cover, `k⁻`'s mirrors skipped: `j⁺, j⁻` sum `P, N` into `N − P`, and `kept` plus the ground gains `2 min(P, N)` on both rows, as `P + N − |N − P|` would by cancellation.
    pub(super) fn fold(self, sheets: Sheets, kept: Option<&[T]>) -> Grounded<T, WddCsr<T, I>> {
        let count = sheets.count();
        let n = (self.sums.len() - 1) / count;
        if let Some(s) = kept {
            assert_eq!(s.len(), n, "one surplus per vertex");
        }
        let ground = |k: usize| self.ground[count * k];
        let mut surplus: Vec<T> = (0..n)
            .map(|k| kept.map_or(ground(k), |s| s[k] + ground(k)))
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
        Grounded {
            adjacency: WddCsr::new(rows),
            grounding: Grounding::of(surplus),
        }
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

/// `adjacency` with each `t > 0` of `twists` as the edge `2k — 2k+1`, first in row `2k`: no other neighbor of `k⁺` sorts below `k⁻`.
fn first_in_row<T: Real, I: Index>(
    adjacency: WddCsr<T, I>,
    twists: impl Iterator<Item = T>,
) -> WddCsr<T, I> {
    let n = adjacency.n();
    let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for (k, t) in twists.enumerate() {
        for i in [2 * k, 2 * k + 1] {
            if i == 2 * k && t > T::zero() {
                indices.push(I::from_index(i + 1));
                values.push(-t);
            }
            for (j, &a) in adjacency.upper(i) {
                indices.push(I::from_index(j));
                values.push(a);
            }
            ptrs.push(indices.len());
        }
    }
    WddCsr::new(Rows::new(ptrs, indices, values, n))
}

/// `CoverSurplus::Auto`'s band, `main`'s for floating, its own so Reduction's can move apart.
const SURPLUS_EDGE_BAND: f64 = 5.0;

/// Whether `ground` is within the band of `ground + Σ|a_ij|`, both `ε`-scaled so neither overflows: a cover's copies then [`twist`] instead of grounding, so its `1/s` common mode cannot swamp `z`; a cover doubles both sums, so `M`'s decide as its own.
fn surplus_edges<T: Real, I: Index>(ground: &[T], adjacency: &WddCsr<T, I>) -> bool {
    let epsilon = widen(T::epsilon());
    let total = lane_sum(ground, epsilon);
    let trace = total + adjacency.scaled_abs_total(epsilon);
    total <= SURPLUS_EDGE_BAND * epsilon * trace
}

/// The edge `k⁺ — k⁻` that acts on `[z; −z]` as grounds `g⁺, g⁻` do; with `g⁺ ≠ g⁻` no ungrounded pair matches them, and the mean differs by less than the singular band. Zero where both copies float, else at least `MIN_POSITIVE`, as a share is.
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
fn cover_vertex(k: usize, sheet: usize) -> usize {
    2 * k + sheet
}

/// The slot of a neighbor `k` of a vertex on `sheet`, across the sheets when their entry `crosses`.
pub(super) fn cover_neighbor(k: usize, sheet: usize, crosses: bool) -> usize {
    cover_vertex(k, sheet ^ usize::from(crosses))
}

/// `[x; −x]` interleaved by vertex, any ground slot after it zero: the cover's right-hand side.
fn ascend<T: Real>(x: &[T], rows: &mut [T]) {
    let (pairs, ground) = rows.split_at_mut(2 * x.len());
    (pairs.chunks_exact_mut(2).zip(x)).for_each(|(r, &v)| {
        r[0] = v;
        r[1] = -v;
    });
    // A lift sums to zero, so the ground's right-hand side is zero.
    ground.fill(T::zero());
}

/// Half the sheets' difference back into `x`, undoing [`ascend`].
fn descend<T: Real>(rows: &[T], x: &mut [T]) {
    let two = T::one() + T::one();
    (x.iter_mut().zip(rows.chunks_exact(2))).for_each(|(x, r)| *x = (r[0] - r[1]) / two);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frustrated 9-cycle of `2^1020` plus `extra` on vertex 0: `Σ|a_ij|` overflows `f64`, the band `≈ 2^974.5` does not.
    fn cycle_cover(extra: f64) -> Represented<f64, u32> {
        let n = 9;
        let indices: Vec<u32> = [1, n - 1].into_iter().chain(2..n).collect();
        let ptrs: Vec<usize> = [0]
            .into_iter()
            .chain(2..=n as usize)
            .chain([n as usize])
            .collect();
        let values = vec![2f64.powi(1020); indices.len()];
        let signed = WddCsr::new(Rows::new(ptrs, indices, values, n as usize));
        let surplus = [extra].into_iter().chain([0.0; 8]).collect();
        Represented::sampled(signed, Sheets::Two, Grounding::of(surplus))
    }

    #[test]
    fn cover_twists_in_band_and_grounds_past_it_where_the_off_diagonal_overflows() {
        let grounded = |extra: f64| cycle_cover(extra).layout().grounded();
        assert!(!grounded(2f64.powi(970)), "twists in band");
        assert!(grounded(2f64.powi(1000)), "grounds past it");
    }
}
