//! The approximate Cholesky's input: a Laplacian with its surplus, on one sheet or a Gremban cover's two, or frustrated `M` as it is, signed.

use rayon::prelude::*;

use super::cholesky::{Arm, AsIs, Eliminated};
use super::config::{Config, CoverSurplus, FrustratedRoute};
use super::sign::{cancel, Signed, Unsigned};
use crate::sgdd::{Balance, FactorTail, Grounding};
use crate::storage::{Rows, UpperAdjacency};
use crate::threads::{task, ONE};
use crate::{narrow, widen, Batch, Index, Real, Threads};

/// `L + diag(s)`, `L`'s off-diagonal nonpositive; a twisted cover's `k⁺—k⁻` edges live in its closure, not its rows.
#[derive(Clone, Debug)]
pub(super) struct Sddm<T, I> {
    adjacency: UpperAdjacency<T, I>,
    closure: Closure<T>,
}

/// What `L` carries beyond its rows: a ground on one sheet or two, or a cover's twists in its place.
#[derive(Clone, Debug)]
pub(super) enum Closure<T> {
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
    pub(super) fn new(adjacency: UpperAdjacency<T, I>, closure: Closure<T>) -> Self {
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

    /// The Gremban cover of frustrated `M`, its surplus carried as `surplus` says.
    fn gremban(
        adjacency: &UpperAdjacency<T, I>,
        block: &Grounding<T>,
        surplus: CoverSurplus,
    ) -> Self {
        let n = adjacency.n();
        // The cover acts as `M` on `[z; −z]`.
        let rows = SddmRows::split(
            Factored::Cover(surplus),
            n,
            n,
            ONE,
            block.surplus(),
            |mut sums, i| {
                for (j, &a) in adjacency.upper(i) {
                    sums.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
                }
            },
        );
        let (adjacency, grounding) = rows.close();
        Self::cover(adjacency, grounding, surplus)
    }

    /// A cover's closed `L`, whose ground twists as `surplus` says; with none the cover floats.
    fn cover(
        adjacency: UpperAdjacency<T, I>,
        grounding: Grounding<T>,
        surplus: CoverSurplus,
    ) -> Self {
        let closure = match grounding.surplus() {
            Some(ground) if twists(surplus, ground, &adjacency) => {
                Closure::Twisted(ground.chunks_exact(2).map(|g| twist(g[0], g[1])).collect())
            }
            _ => Closure::Cover(grounding),
        };
        Self::new(adjacency, closure)
    }

    /// Balanced on one sheet, frustrated on a cover's two.
    pub(super) fn balance(&self) -> Balance {
        match self.closure {
            Closure::Folded(_) => Balance::Balanced,
            Closure::Cover(_) | Closure::Twisted(_) => Balance::Frustrated,
        }
    }

    fn grounded(&self) -> bool {
        (self.closure.grounding()).is_some_and(Grounding::is_grounded)
    }
}

impl<T: Real, I: Index> Eliminated<T, I> for Sddm<T, I> {
    /// `L`'s weights are magnitudes.
    type Sign = Unsigned;

    fn n(&self) -> usize {
        self.adjacency.n()
    }

    /// Positive Laplacian weights, each edge once with any twists, the ground vertex `n` joined by each surplus.
    fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
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
            ground_edges(grounding, n, edge);
        }
    }

    fn sign(&self) -> Unsigned {
        Unsigned
    }

    fn pin(&self, tail: Option<FactorTail<'_, T>>) -> Option<usize> {
        match self.grounded() {
            // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
            true => Some(self.n()),
            // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
            false => {
                (tail.filter(|f| f.spans_wide())).map(|f| first_copy(self.balance(), f.largest()))
            }
        }
    }
}

/// Frustrated `M` as it is, for the signed elimination: its off-diagonal of either sign, its surplus to a ground it always has, where cancellation lands.
pub(super) struct SignedWdd<T, I> {
    adjacency: UpperAdjacency<T, I>,
    grounding: Grounding<T>,
}

impl<T: Real, I: Index> SignedWdd<T, I> {
    /// Panics unless nonempty, with one surplus per row.
    pub(super) fn new(adjacency: UpperAdjacency<T, I>, grounding: Grounding<T>) -> Self {
        assert!(adjacency.n() > 0, "M has a vertex");
        assert!(grounding.fits(adjacency.n()), "one surplus per row");
        Self {
            adjacency,
            grounding,
        }
    }
}

impl<T: Real, I: Index> Eliminated<T, I> for SignedWdd<T, I> {
    type Sign = Signed<I>;

    fn n(&self) -> usize {
        self.adjacency.n()
    }

    /// `−a_ij`, its sign `a_ij`'s flipped, each edge once, the ground vertex `n` joined by each surplus.
    fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
        let n = self.n();
        for i in 0..n {
            for (j, &a) in self.adjacency.upper(i) {
                edge(I::from_index(i), I::from_index(j), -a);
            }
        }
        ground_edges(&self.grounding, n, edge);
    }

    fn sign(&self) -> Signed<I> {
        Signed {
            ground: I::from_index(self.n()),
        }
    }

    /// The ground, past every slot: a signed factor pins none.
    fn pin(&self, _: Option<FactorTail<'_, T>>) -> Option<usize> {
        Some(self.n())
    }
}

/// Each surplus as an edge to the ground vertex `n`.
fn ground_edges<T: Real, I: Index>(
    grounding: &Grounding<T>,
    n: usize,
    mut edge: impl FnMut(I, I, T),
) {
    for k in 0..n {
        if let Some(s) = grounding.edge(k) {
            edge(I::from_index(k), I::from_index(n), s);
        }
    }
}

/// What a block's Cholesky factors: `M` as it is, densely; or approximately its SDDM on one sheet, its Gremban cover carrying its surplus as said, or frustrated `M` signed.
#[derive(Clone, Copy)]
pub(super) enum Factored {
    AsIs(Balance),
    Folded,
    Cover(CoverSurplus),
    Signed,
}

impl Factored {
    /// For a block of `n` vertices, a bipartite one's kept side, under `config`: dense up to its `exact_max_dim`, else folded when balanced, else by its frustrated route.
    pub(super) fn of(config: &Config, balance: Balance, n: usize) -> Self {
        match (config.dense(n), balance, config.frustrated) {
            (true, _, _) => Self::AsIs(balance),
            (false, Balance::Balanced, _) => Self::Folded,
            (false, Balance::Frustrated, FrustratedRoute::Cover(surplus)) => Self::Cover(surplus),
            (false, Balance::Frustrated, FrustratedRoute::Signed) => Self::Signed,
        }
    }

    /// Frustrated `M`'s own rows are signed pairs, whether factored densely or signed.
    pub(super) fn layout(self) -> Layout {
        match self {
            Self::AsIs(Balance::Balanced) | Self::Folded => Layout::Sheet,
            Self::Cover(_) => Layout::Cover,
            Self::AsIs(Balance::Frustrated) | Self::Signed => Layout::SignedPairs,
        }
    }

    /// The arm of a block's `M`, its off-diagonal and surplus as they are.
    pub(super) fn arm<T: Real, I: Index>(
        self,
        adjacency: UpperAdjacency<T, I>,
        grounding: Grounding<T>,
    ) -> Arm<T, I> {
        match self {
            Self::Cover(surplus) => Arm::Sddm(Sddm::gremban(&adjacency, &grounding, surplus)),
            _ => self.closed(adjacency, grounding),
        }
    }

    /// The arm of the matrix rows in its [`layout`](Self::layout) close into.
    fn closed<T: Real, I: Index>(
        self,
        adjacency: UpperAdjacency<T, I>,
        grounding: Grounding<T>,
    ) -> Arm<T, I> {
        match self {
            Self::AsIs(balance) => Arm::AsIs(AsIs::new(adjacency, balance, grounding)),
            Self::Folded => Arm::Sddm(Sddm::new(adjacency, Closure::Folded(grounding))),
            Self::Cover(surplus) => Arm::Sddm(Sddm::cover(adjacency, grounding, surplus)),
            Self::Signed => Arm::Signed(SignedWdd::new(adjacency, grounding)),
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
    /// Columns for `n` vertices: two to each of a cover's or a signed pair's, the copies a cover holds of each.
    pub(super) fn columns(self, n: usize) -> usize {
        match self {
            Self::Sheet => n,
            Self::Cover | Self::SignedPairs => 2 * n,
        }
    }

    /// Rows for `n` vertices: a cover's two copies of each, else one.
    pub(super) fn rows(self, n: usize) -> usize {
        n << self.copy_shift()
    }

    /// Row `row`'s vertex, a cover's copies interleaved as [`cover_vertex`] places them.
    fn vertex_of(self, row: usize) -> usize {
        row >> self.copy_shift()
    }

    /// Each vertex's rows as a shift: one bit on a cover, which holds two copies; signed pairs are two columns of one row.
    fn copy_shift(self) -> u32 {
        match self {
            Self::Cover => 1,
            Self::Sheet | Self::SignedPairs => 0,
        }
    }
}

/// Rows in the [`Layout`] of what they are formed for, each ground ending with its vertex's `surplus`.
pub(super) struct SddmRows<'a, T, I> {
    factored: Factored,
    surplus: Option<&'a [T]>,
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

    /// One row, its ground plus `surplus`.
    fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>), surplus: Option<T>) {
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
        self.ground.push(surplus.map_or(ground, |s| s + ground));
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

impl<'a, T: Real, I: Index> SddmRows<'a, T, I> {
    /// Rows of `n` vertices for `factored`, each [`row`](Self::row) in order, one `surplus` per vertex.
    pub(super) fn new(factored: Factored, n: usize, surplus: Option<&'a [T]>) -> Self {
        let rows = RowSums::new(factored.layout().columns(n));
        Self {
            factored,
            surplus,
            rows,
        }
    }

    pub(super) fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>)) {
        let vertex = self.factored.layout().vertex_of(self.rows.ptrs.len() - 1);
        self.rows.row(add, self.surplus.map(|s| s[vertex]));
    }

    /// Rows of `n` vertices for `factored`, one `surplus` per vertex, `row` writing vertex `k`'s row once, `k⁺`'s on a cover, which `k⁻` mirrors; split by vertex across threads, at least `min_len` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        factored: Factored,
        n: usize,
        min_len: usize,
        threads: Threads,
        surplus: Option<&'a [T]>,
        row: impl Fn(ColumnSums<'_, T, I>, usize) + Sync,
    ) -> Self {
        let layout = factored.layout();
        let columns = layout.columns(n);
        // A bool, tested against zero as the balance was: comparing the three-way layout measured a `cmp` more per row.
        let mirrored = layout.copy_shift() != 0;
        let add = |mut rows: RowSums<T, I>, k| {
            rows.row(|sums| row(sums, k), surplus.map(|s| s[k]));
            if mirrored {
                rows.mirror_row();
            }
            rows
        };
        let Some(per) = threads.per_task(n, min_len) else {
            let rows = (0..n).fold(RowSums::new(columns), add);
            return Self {
                factored,
                surplus,
                rows,
            };
        };
        let rows = (0..n)
            .into_par_iter()
            .with_min_len(per)
            .fold(|| RowSums::new(columns), |rows, k| task(|| add(rows, k)));
        let rows = rows.reduce_with(|mut rows, part| {
            let offset = rows.indices.len();
            rows.ptrs.extend(part.ptrs[1..].iter().map(|&p| p + offset));
            rows.indices.extend(part.indices);
            rows.values.extend(part.values);
            rows.ground.extend(part.ground);
            rows
        });
        let rows = rows.unwrap_or_else(|| RowSums::new(columns));
        Self {
            factored,
            surplus,
            rows,
        }
    }

    /// The matrix the rows hold, and each row's ground: an SDDM's `L` on its sheets, or `M` [`fold`]ed from signed pairs.
    fn close(self) -> (UpperAdjacency<T, I>, Grounding<T>) {
        let Self { factored, rows, .. } = self;
        match factored.layout() {
            Layout::SignedPairs => fold(rows),
            Layout::Sheet | Layout::Cover => {
                let n = rows.sums.len() - 1;
                let ground = Grounding::of(rows.ground);
                let rows = Rows::new(rows.ptrs, rows.indices, rows.values, n);
                (UpperAdjacency::new(rows), ground)
            }
        }
    }

    /// The arm of what they are formed for, [`close`](Self::close)d.
    pub(super) fn arm(self) -> Arm<T, I> {
        let factored = self.factored;
        let (adjacency, grounding) = self.close();
        factored.closed(adjacency, grounding)
    }
}

/// `M` from its signed pairs' rows: `j`'s `P, N` sum into `N − P`, and the surplus gains what that sum [`cancel`]s on both rows.
fn fold<T: Real, I: Index>(rows: RowSums<T, I>) -> (UpperAdjacency<T, I>, Grounding<T>) {
    let RowSums {
        mut ptrs,
        mut indices,
        mut values,
        ground: mut surplus,
        ..
    } = rows;
    let n = surplus.len();
    // A pair folds into at most one entry, so each row compacts in place behind its reads.
    let (mut written, mut at) = (0, 0);
    for k in 0..n {
        let end = ptrs[k + 1];
        while at < end {
            let j = indices[at].index() / 2;
            let mut sums = [T::zero(); 2];
            while at < end && indices[at].index() / 2 == j {
                sums[indices[at].index() % 2] = -values[at];
                at += 1;
            }
            let [same, opposite] = sums;
            let (entry, gain) = cancel(opposite, -same);
            surplus[k] = surplus[k] + gain;
            surplus[j] = surplus[j] + gain;
            if entry != T::zero() {
                indices[written] = I::from_index(j);
                values[written] = entry;
                written += 1;
            }
        }
        ptrs[k + 1] = written;
    }
    indices.truncate(written);
    values.truncate(written);
    let rows = Rows::new(ptrs, indices, values, n);
    (UpperAdjacency::new(rows), Grounding::of(surplus))
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

/// Whether a cover's copies [`twist`] instead of grounding, as `surplus` says; `Auto` weighs `ground` against `ground + Σ|a_ij|`, both `ε`-scaled so neither overflows, and a cover doubles both sums, so `M`'s decide as its own.
fn twists<T: Real, I: Index>(
    surplus: CoverSurplus,
    ground: &[T],
    adjacency: &UpperAdjacency<T, I>,
) -> bool {
    match surplus {
        CoverSurplus::GroundVertex => false,
        CoverSurplus::SurplusEdges => true,
        CoverSurplus::Auto => {
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
    match balance {
        Balance::Balanced => k,
        Balance::Frustrated => cover_vertex(k, 0),
    }
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
        Sddm::gremban(&signed, &grounding, CoverSurplus::Auto)
    }

    #[test]
    fn cover_twists_in_band_and_grounds_past_it_where_the_off_diagonal_overflows() {
        assert!(!cycle_cover(2f64.powi(970)).grounded(), "twists in band");
        assert!(cycle_cover(2f64.powi(1000)).grounded(), "grounds past it");
    }
}
