//! The Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two.

use rayon::prelude::*;

use super::config::{CoverSurplus, FrustratedRoute};
use super::sign::{cancel, Unsigned};
use crate::sgdd::{Balance, Grounding};
use crate::storage::{Rows, UpperAdjacency};
use crate::{narrow, widen, Batch, Index, Real};

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

    /// Folded `M` on one sheet, or the Gremban cover of frustrated `M` on two, its surplus carried as `route` says.
    pub(super) fn of_block(
        adjacency: UpperAdjacency<T, I>,
        balance: Balance,
        grounding: Grounding<T>,
        route: FrustratedRoute,
    ) -> Self {
        match balance {
            Balance::Balanced => Self::new(adjacency, Closure::Folded(grounding)),
            Balance::Frustrated => {
                let n = adjacency.n();
                // The cover acts as `M − diag(s)` on `[z; −z]`.
                let rows = SddmRows::split(Layout::Cover, n, n, |mut sums, i| {
                    for (j, &a) in adjacency.upper(i) {
                        sums.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
                    }
                });
                Self::of_rows(rows, &grounding, route)
            }
        }
    }

    /// An SDDM's rows [`close`](SddmRows::close)d; on a cover, a ground twists as `route` says, and with none the cover floats. Panics on signed pairs, which close into `M`.
    pub(super) fn of_rows(
        rows: SddmRows<T, I>,
        block: &Grounding<T>,
        route: FrustratedRoute,
    ) -> Self {
        let layout = rows.layout;
        let (adjacency, grounding) = rows.close(block);
        let closure = match (layout, grounding.surplus()) {
            (Layout::Sheet, _) => Closure::Folded(grounding),
            (Layout::Cover, Some(ground)) if twists(route, ground, &adjacency) => {
                Closure::Twisted(ground.chunks_exact(2).map(|g| twist(g[0], g[1])).collect())
            }
            (Layout::Cover, _) => Closure::Cover(grounding),
            (Layout::SignedPairs, _) => panic!("signed pairs close into M, not an SDDM"),
        };
        Self::new(adjacency, closure)
    }

    pub(super) fn n(&self) -> usize {
        self.adjacency.n()
    }

    pub(super) fn grounded(&self) -> bool {
        (self.closure.grounding()).is_some_and(Grounding::is_grounded)
    }

    /// `L`'s weights are magnitudes.
    pub(super) fn sign(&self) -> Unsigned {
        Unsigned
    }

    /// Balanced on one sheet, frustrated on a cover's two.
    pub(super) fn balance(&self) -> Balance {
        match self.closure {
            Closure::Folded(_) => Balance::Balanced,
            Closure::Cover(_) | Closure::Twisted(_) => Balance::Frustrated,
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

/// Which columns a vertex's row sums into, and which rows there are.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Layout {
    /// A folded SDDM's: column `j`, a row per vertex.
    Sheet,
    /// A Gremban cover's: column [`cover_vertex`], rows `k⁺` and its mirror `k⁻`.
    Cover,
    /// `M`'s, for [`close`](SddmRows::close) to sign: `2j` its magnitudes alike in sign, `2j + 1` the unlike, a row per vertex, as `k⁺`'s on a cover.
    SignedPairs,
}

impl Layout {
    /// An SDDM's for `balance`.
    pub(super) fn sddm(balance: Balance) -> Self {
        match balance {
            Balance::Balanced => Self::Sheet,
            Balance::Frustrated => Self::Cover,
        }
    }

    /// `M`'s own for `balance`: a balanced block's rows are already `M`'s.
    pub(super) fn as_is(balance: Balance) -> Self {
        match balance {
            Balance::Balanced => Self::Sheet,
            Balance::Frustrated => Self::SignedPairs,
        }
    }

    /// Columns for `n` vertices: two to each of a cover's or a signed pair's, the copies a cover holds of each.
    pub(super) fn columns(self, n: usize) -> usize {
        match self {
            Self::Sheet => n,
            Self::Cover | Self::SignedPairs => 2 * n,
        }
    }
}

/// Rows in a [`Layout`].
pub(super) struct SddmRows<T, I> {
    layout: Layout,
    rows: RowSums<T, I>,
}

/// Rows summed one at a time, each column's addends in arrival order and stored as `−w` by column; column `n` is the row's ground weight. Apart from the layout: a task moves these per row, and a layout byte among them measured extra stores in each move.
struct RowSums<T, I> {
    sums: Vec<T>,
    touched: Vec<I>,
    ptrs: Vec<usize>,
    indices: Vec<I>,
    values: Vec<T>,
    ground: Vec<T>,
}

impl<T: Real, I: Index> RowSums<T, I> {
    fn new(n: usize) -> Self {
        Self {
            sums: vec![T::zero(); n + 1],
            touched: Vec::new(),
            ptrs: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
            ground: Vec::new(),
        }
    }

    fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>)) {
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
}

impl<T: Real, I: Index> SddmRows<T, I> {
    /// Rows of `n` vertices in `layout`.
    pub(super) fn new(layout: Layout, n: usize) -> Self {
        let rows = RowSums::new(layout.columns(n));
        Self { layout, rows }
    }

    pub(super) fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>)) {
        self.rows.row(add);
    }

    /// Rows of `n` vertices in `layout`, `row` writing vertex `k`'s row once, `k⁺`'s on a cover, which `k⁻` mirrors; split by vertex across threads, at least `min_len` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        layout: Layout,
        n: usize,
        min_len: usize,
        row: impl Fn(ColumnSums<'_, T, I>, usize) + Sync,
    ) -> Self {
        let columns = layout.columns(n);
        // A bool, tested against zero as the balance was: comparing the three-way layout measured a `cmp` more per row.
        let mirrored = layout == Layout::Cover;
        let rows = (0..n).into_par_iter().with_min_len(min_len).fold(
            || RowSums::new(columns),
            |mut rows, k| {
                rows.row(|sums| row(sums, k));
                if mirrored {
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
        let rows = rows.unwrap_or_else(|| RowSums::new(columns));
        Self { layout, rows }
    }

    /// The matrix the rows hold, and each row's ground plus its vertex's surplus, the last of `block`'s: an SDDM's `L` on its sheets, or `M` [`fold`](Self::fold)ed from signed pairs.
    pub(super) fn close(self, block: &Grounding<T>) -> (UpperAdjacency<T, I>, Grounding<T>) {
        let Self { layout, rows } = self;
        match layout {
            Layout::SignedPairs => fold(rows, block),
            Layout::Sheet | Layout::Cover => {
                let n = rows.sums.len() - 1;
                let ground = with_surplus(rows.ground, block, layout.columns(1));
                let rows = Rows::new(rows.ptrs, rows.indices, rows.values, n);
                (UpperAdjacency::new(rows), Grounding::of(ground))
            }
        }
    }
}

/// `M` from its signed pairs' rows: `j`'s `P, N` sum into `N − P`, and the surplus gains what that sum [`cancel`]s on both rows.
fn fold<T: Real, I: Index>(
    rows: RowSums<T, I>,
    block: &Grounding<T>,
) -> (UpperAdjacency<T, I>, Grounding<T>) {
    let n = rows.ground.len();
    let mut surplus = with_surplus(rows.ground, block, 1);
    let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for k in 0..n {
        let mut at = rows.ptrs[k];
        let columns = &rows.indices[at..rows.ptrs[k + 1]];
        for pair in columns.chunk_by(|a, b| a.index() / 2 == b.index() / 2) {
            let mut sums = [T::zero(); 2];
            for (c, &v) in pair.iter().zip(&rows.values[at..]) {
                sums[c.index() % 2] = -v;
            }
            at += pair.len();
            let [same, opposite] = sums;
            let j = pair[0].index() / 2;
            let (entry, gain) = cancel(opposite, -same);
            surplus[k] = surplus[k] + gain;
            surplus[j] = surplus[j] + gain;
            if entry != T::zero() {
                indices.push(I::from_index(j));
                values.push(entry);
            }
        }
        ptrs.push(indices.len());
    }
    let rows = Rows::new(ptrs, indices, values, n);
    (UpperAdjacency::new(rows), Grounding::of(surplus))
}

/// Each row's ground plus its vertex's surplus, the last of `block`'s, `copies` rows to a vertex.
fn with_surplus<T: Real>(mut ground: Vec<T>, block: &Grounding<T>, copies: usize) -> Vec<T> {
    if let Some(stated) = block.surplus() {
        let kept = stated[stated.len() - ground.len() / copies..].iter();
        let copied = kept.flat_map(|&s| std::iter::repeat_n(s, copies));
        (ground.iter_mut().zip(copied)).for_each(|(g, s)| *g = s + *g);
    }
    ground
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

/// [`CoverSurplus::Auto`]'s twist band, in `ε`.
const TWIST_BAND: f64 = 5.0;

/// Whether a cover's copies [`twist`] instead of grounding, as `route` says; `Auto` weighs `ground` against `ground + Σ|a_ij|`, both `ε`-scaled so neither overflows, and a cover doubles both sums, so `M`'s decide as its own.
fn twists<T: Real, I: Index>(
    route: FrustratedRoute,
    ground: &[T],
    adjacency: &UpperAdjacency<T, I>,
) -> bool {
    match route {
        FrustratedRoute::Cover(CoverSurplus::GroundVertex) => false,
        FrustratedRoute::Cover(CoverSurplus::SurplusEdges) => true,
        FrustratedRoute::Cover(CoverSurplus::Auto) => {
            let epsilon = widen(T::epsilon());
            let total = crate::lane_sum(ground, epsilon);
            let trace = total + adjacency.scaled_abs_total(epsilon);
            total <= TWIST_BAND * epsilon * trace
        }
    }
}

/// The edge `k⁺ — k⁻`, a quarter of `g⁺ + g⁻`, that acts on `[z; −z]` as grounds `g⁺, g⁻` together do; with `g⁺ ≠ g⁻` the sheets are not mirrored, and the edge holds their mean on both. Zero where both copies float, else at least `MIN_POSITIVE`, as a share is.
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

/// Vertex `k`'s first copy: the sheets interleave, as [`cover_vertex`] places them.
pub(super) fn first_copy(balance: Balance, k: usize) -> usize {
    Layout::sddm(balance).columns(k)
}

/// Slot `i` of `[x; −x]` interleaved by vertex: the lift a cover solves on.
pub(super) fn lifted<T: Real, V: Batch<T>>(x: &[V], i: usize) -> V {
    match i % 2 {
        0 => x[i / 2],
        _ => x[i / 2].neg(),
    }
}

/// Half the sheets' difference back into `x`, undoing [`lifted`].
pub(super) fn descend<T: Real, V: Batch<T>>(rows: &[V], x: &mut [V]) {
    let two = T::one() + T::one();
    (x.iter_mut().zip(rows.chunks_exact(2))).for_each(|(x, r)| *x = r[0].sub(r[1]).div(two));
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
        let grounding = Grounding::of(surplus);
        let route = FrustratedRoute::Cover(CoverSurplus::Auto);
        Sddm::of_block(signed, Balance::Frustrated, grounding, route)
    }

    #[test]
    fn cover_twists_in_band_and_grounds_past_it_where_the_off_diagonal_overflows() {
        assert!(!cycle_cover(2f64.powi(970)).grounded(), "twists in band");
        assert!(cycle_cover(2f64.powi(1000)).grounded(), "grounds past it");
    }
}
