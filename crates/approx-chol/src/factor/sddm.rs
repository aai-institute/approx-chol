//! What a block's Cholesky factors and the matrix it reads: `M` as it is, its SDDM on one sheet, or its Gremban cover's on two.

use rayon::prelude::*;

use super::config::{Config, CoverSurplus, FrustratedRoute};
use super::sign::cancel;
use crate::sgdd::{Balance, FactorTail, Grounding};
use crate::storage::{Rows, UpperAdjacency};
use crate::threads::{task, ONE};
use crate::{narrow, widen, Batch, Index, Real, Threads};

/// An SDD matrix, a signed graph's Laplacian plus a surplus: `M` itself, its off-diagonal nonpositive or of either sign, or `M`'s Gremban cover on two interleaved sheets, each vertex's copies sharing its surplus.
pub(super) struct Sdd<T, I> {
    off_diagonal: UpperAdjacency<T, I>,
    surplus: Grounding<T>,
    layout: Layout,
}

impl<T: Real, I: Index> Sdd<T, I> {
    /// Panics unless nonempty, a cover's sheets holding the same vertices, with one surplus per vertex.
    pub(super) fn new(
        off_diagonal: UpperAdjacency<T, I>,
        surplus: Grounding<T>,
        layout: Layout,
    ) -> Self {
        let (n, shift) = (off_diagonal.n(), layout.copy_shift());
        assert!(n > 0, "M has a vertex");
        assert_eq!(
            n,
            n >> shift << shift,
            "every sheet holds the same vertices"
        );
        assert!(surplus.fits(n >> shift), "one surplus per vertex");
        debug_assert!(
            layout == Layout::Signed || off_diagonal.nonpositive(),
            "a Z-matrix's off-diagonal is nonpositive"
        );
        Self {
            off_diagonal,
            surplus,
            layout,
        }
    }

    pub(super) fn layout(&self) -> Layout {
        self.layout
    }

    /// Its slots; a ground is the vertex past them.
    pub(super) fn n(&self) -> usize {
        self.off_diagonal.n()
    }

    /// Whether it is singular: a Z-matrix without surplus, as a cover is whose surplus went into its twists.
    pub(super) fn floats(&self) -> bool {
        self.layout != Layout::Signed && !self.surplus.is_grounded()
    }

    /// Each edge once as `(i, j, −a_ij)`, then each surplus as `(k, n, s_k)`.
    pub(super) fn grounded_edges(&self, mut edge: impl FnMut(I, I, T)) {
        let (n, shift) = (self.n(), self.layout.copy_shift());
        for i in 0..n {
            for (j, &a) in self.off_diagonal.upper(i) {
                edge(I::from_index(i), I::from_index(j), -a);
            }
        }
        for k in 0..n {
            if let Some(s) = self.surplus.edge(k >> shift) {
                edge(I::from_index(k), I::from_index(n), s);
            }
        }
    }

    /// The vertex no pivot eliminates: the ground, the vertex past the slots, or a floating matrix's gauge, by `tail`, on a cover its first copy.
    pub(super) fn pin(&self, tail: Option<FactorTail<'_, T>>) -> Option<usize> {
        match self.floats() {
            // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
            false => Some(self.n()),
            // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
            true => {
                (tail.filter(|f| f.spans_wide())).map(|f| f.largest() << self.layout.copy_shift())
            }
        }
    }
}

/// What a block's Cholesky factors: `M` as it is, densely; or approximately its SDDM on one sheet, its Gremban cover grounded or twisted, or frustrated `M` signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Factored {
    AsIs(Balance),
    Folded,
    /// Each copy's surplus an edge to the ground.
    Cover,
    /// Each vertex's surplus its [`twist`], an edge between its copies.
    TwistedCover,
    Signed,
}

impl Factored {
    /// For a block of `n` vertices, a bipartite one's kept side, under `config`: dense up to its `exact_max_dim`, else folded when balanced, else by its frustrated route, a cover twisting as [`twists`] decides from `M`'s `ground` and `off_diagonal`.
    pub(super) fn of<T: Real>(
        config: &Config,
        balance: Balance,
        n: usize,
        ground: Option<&[T]>,
        off_diagonal: impl FnOnce(f64) -> f64,
    ) -> Self {
        match (config.dense(n), balance, config.frustrated) {
            (true, _, _) => Self::AsIs(balance),
            (false, Balance::Balanced, _) => Self::Folded,
            (false, Balance::Frustrated, FrustratedRoute::Cover(surplus)) => match ground {
                Some(ground) if twists(surplus, ground, off_diagonal) => Self::TwistedCover,
                _ => Self::Cover,
            },
            (false, Balance::Frustrated, FrustratedRoute::Signed) => Self::Signed,
        }
    }

    /// Frustrated `M`'s own rows are signed pairs, whether factored densely or signed.
    pub(super) fn layout(self) -> Layout {
        match self {
            Self::AsIs(Balance::Balanced) | Self::Folded => Layout::Like,
            Self::Cover | Self::TwistedCover => Layout::Cover,
            Self::AsIs(Balance::Frustrated) | Self::Signed => Layout::Signed,
        }
    }

    /// A block's `M`, its off-diagonal and surplus as they are, as the matrix this factors.
    pub(super) fn on<T: Real, I: Index>(
        self,
        off_diagonal: UpperAdjacency<T, I>,
        surplus: Grounding<T>,
    ) -> Sdd<T, I> {
        match self.layout() {
            Layout::Cover => {
                // The cover acts as `M` on `[z; −z]`.
                let (n, ground) = (off_diagonal.n(), surplus.surplus().map(<[T]>::to_vec));
                let rows = SddmRows::split(self, n, n, ONE, ground, |mut sums, i| {
                    for (j, &a) in off_diagonal.upper(i) {
                        sums.add(cover_vertex(j, usize::from(crosses(a))), a.abs());
                    }
                });
                rows.close()
            }
            layout => Sdd::new(off_diagonal, surplus, layout),
        }
    }

    /// Each vertex's [`twist`] from its `ground`, where the cover twists.
    fn twists<T: Real>(self, ground: &Option<Vec<T>>) -> Option<&[T]> {
        ground.as_deref().filter(|_| self == Self::TwistedCover)
    }
}

/// What a matrix's rows hold, while they form and once they close.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// `M` with like edges only, a Z-matrix: column `j`, a row per vertex.
    Like,
    /// `M`'s Gremban cover, a Z-matrix on two sheets: column [`cover_vertex`], rows `k⁺` and `k⁻`.
    Cover,
    /// `M` of either sign, a row per vertex: forming, column `2j` sums its like magnitudes and `2j + 1` the unlike, as `k⁺`'s on a cover, which [`close`](SddmRows::close) folds into one signed entry.
    Signed,
}

impl Layout {
    /// Columns for `n` vertices: two to each of a cover's or a signed pair's, the copies a cover holds of each.
    pub(super) fn columns(self, n: usize) -> usize {
        match self {
            Self::Like => n,
            Self::Cover | Self::Signed => 2 * n,
        }
    }

    /// Rows for `n` vertices: a cover's two copies of each, else one.
    pub(super) fn rows(self, n: usize) -> usize {
        n << self.copy_shift()
    }

    /// Each vertex's rows as a shift: one bit on a cover, which holds two copies; signed pairs are two columns of one row.
    fn copy_shift(self) -> u32 {
        match self {
            Self::Cover => 1,
            Self::Like | Self::Signed => 0,
        }
    }
}

/// Rows in the [`Layout`] of what they are formed for, and each vertex's ground.
pub(super) struct SddmRows<T, I> {
    factored: Factored,
    ground: Option<Vec<T>>,
    rows: RowSums<T, I>,
}

/// Rows summed one at a time, each column's addends in arrival order and stored as `−w` by column. Apart from the layout: a task moves these per row, and a layout byte among them measured extra stores in each move.
struct RowSums<T, I> {
    sums: Vec<T>,
    touched: Vec<I>,
    ptrs: Vec<usize>,
    indices: Vec<I>,
    values: Vec<T>,
}

impl<T: Real, I: Index> RowSums<T, I> {
    fn new(n: usize) -> Self {
        Self {
            sums: vec![T::zero(); n],
            touched: Vec::new(),
            ptrs: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// One row, and where `ground` twists row `at`, a cover's `k⁺`, its edge to `k⁻` first, where the column sorts.
    fn row(&mut self, ground: Option<&[T]>, at: usize, add: impl FnOnce(ColumnSums<'_, T, I>)) {
        let mut sums = ColumnSums {
            sums: &mut self.sums,
            touched: &mut self.touched,
        };
        if let Some(t) = ground.filter(|_| at % 2 == 0).map(|g| twist(g[at / 2])) {
            sums.add(at + 1, t);
        }
        add(sums);
        self.touched.sort_unstable();
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

    /// Repeats the row just ended, slot `at`'s, on the other sheet, each column's sheet bit flipped, but for its twist, its edge to `at + 1`.
    fn mirror_row(&mut self, at: usize) {
        let (start, end) = (self.ptrs[self.ptrs.len() - 2], self.indices.len());
        let start = start + usize::from(self.indices.get(start) == Some(&I::from_index(at + 1)));
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
    /// Rows of `n` vertices for `factored`, each [`row`](Self::row) in order, one `ground` per vertex.
    pub(super) fn new(factored: Factored, n: usize, ground: Option<Vec<T>>) -> Self {
        let rows = RowSums::new(factored.layout().columns(n));
        Self {
            factored,
            ground,
            rows,
        }
    }

    pub(super) fn row(&mut self, add: impl FnOnce(ColumnSums<'_, T, I>)) {
        let twists = self.factored.twists(&self.ground);
        self.rows.row(twists, self.rows.ptrs.len() - 1, add);
    }

    /// Rows of `n` vertices for `factored`, one `ground` per vertex, `row` writing vertex `k`'s row once, `k⁺`'s on a cover, which `k⁻` mirrors; split by vertex across threads, at least `min_len` a task, so each sums alone and the bits match at any count.
    pub(super) fn split(
        factored: Factored,
        n: usize,
        min_len: usize,
        threads: Threads,
        ground: Option<Vec<T>>,
        row: impl Fn(ColumnSums<'_, T, I>, usize) + Sync,
    ) -> Self {
        let layout = factored.layout();
        let columns = layout.columns(n);
        // A bool, tested against zero as the balance was: comparing the three-way layout measured a `cmp` more per row.
        let mirrored = layout.copy_shift() != 0;
        let twists = factored.twists(&ground);
        let add = |mut rows: RowSums<T, I>, k| {
            rows.row(twists, 2 * k, |sums| row(sums, k));
            if mirrored {
                rows.mirror_row(2 * k);
            }
            rows
        };
        let rows = match threads.per_task(n, min_len) {
            None => (0..n).fold(RowSums::new(columns), add),
            Some(per) => {
                let rows = (0..n)
                    .into_par_iter()
                    .with_min_len(per)
                    .fold(|| RowSums::new(columns), |rows, k| task(|| add(rows, k)));
                let rows = rows.reduce_with(|mut rows, part| {
                    let offset = rows.indices.len();
                    rows.ptrs.extend(part.ptrs[1..].iter().map(|&p| p + offset));
                    rows.indices.extend(part.indices);
                    rows.values.extend(part.values);
                    rows
                });
                rows.unwrap_or_else(|| RowSums::new(columns))
            }
        };
        Self {
            factored,
            ground,
            rows,
        }
    }

    /// The matrix the rows hold, each vertex's ground its surplus but where it twisted: an SDDM's `L` on its sheets, or `M` [`fold`]ed from signed pairs.
    pub(super) fn close(self) -> Sdd<T, I> {
        let Self {
            factored,
            ground,
            rows,
        } = self;
        let ground = ground.unwrap_or_default();
        let (off_diagonal, surplus) = match factored.layout() {
            Layout::Signed => fold(rows, ground),
            Layout::Like | Layout::Cover => {
                let n = rows.sums.len();
                let rows = Rows::new(rows.ptrs, rows.indices, rows.values, n);
                (UpperAdjacency::new(rows), Grounding::of(ground))
            }
        };
        let surplus = match factored {
            Factored::TwistedCover => Grounding::ungrounded(),
            _ => surplus,
        };
        Sdd::new(off_diagonal, surplus, factored.layout())
    }
}

/// `M` from its signed pairs' rows: `j`'s `P, N` sum into `N − P`, and the ground, empty where there is none, gains what that sum [`cancel`]s on both rows.
fn fold<T: Real, I: Index>(
    rows: RowSums<T, I>,
    mut surplus: Vec<T>,
) -> (UpperAdjacency<T, I>, Grounding<T>) {
    let RowSums {
        mut ptrs,
        mut indices,
        mut values,
        ..
    } = rows;
    let n = ptrs.len() - 1;
    surplus.resize(n, T::zero());
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

/// Whether a cover's copies [`twist`] instead of grounding, as `surplus` says; `Auto` weighs `M`'s `ground` against it plus `off_diagonal(scale)`, `scale · Σ|a_ij|` over both triangles, both `ε`-scaled so neither overflows: a cover doubles both, so `M`'s decide as its own.
fn twists<T: Real>(
    surplus: CoverSurplus,
    ground: &[T],
    off_diagonal: impl FnOnce(f64) -> f64,
) -> bool {
    match surplus {
        CoverSurplus::GroundVertex => false,
        CoverSurplus::SurplusEdges => true,
        CoverSurplus::Auto => {
            let epsilon = widen(T::epsilon());
            let total = crate::lane_sum(ground, epsilon);
            let trace = total + off_diagonal(epsilon);
            total <= TWIST_BAND * epsilon * trace
        }
    }
}

/// The edge `k⁺ — k⁻`, a quarter of both copies' ground `g` each, that acts on `[z; −z]` as the two grounds together do. Zero where the vertex floats, else at least `MIN_POSITIVE`, as a share is.
fn twist<T: Real>(g: T) -> T {
    if g == T::zero() {
        return T::zero();
    }
    let quarter = g * narrow::<T>(0.25);
    (quarter + quarter).max(T::min_positive_value())
}

/// A negative entry joins its ends on one sheet, a positive one across them.
pub(super) fn crosses<T: Real>(c: T) -> bool {
    c > T::zero()
}

/// Vertex `k`'s copy on `sheet` of a cover: sheets interleave, so row `k⁻` mirrors its `k⁺` by flipping each column's low bit.
pub(super) fn cover_vertex(k: usize, sheet: usize) -> usize {
    2 * k + sheet
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
    fn cycle_cover(extra: f64) -> Sdd<f64, u32> {
        let config = Config {
            exact_max_dim: 0,
            frustrated: FrustratedRoute::Cover(CoverSurplus::Auto),
            ..Config::default()
        };
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
        let off_diagonal = |scale| signed.scaled_abs_total(scale);
        let factored = Factored::of(
            &config,
            Balance::Frustrated,
            n as usize,
            grounding.surplus(),
            off_diagonal,
        );
        let cover = factored.layout() == Layout::Cover;
        assert!(cover, "a frustrated block's cover is an SDDM");
        factored.on(signed, grounding)
    }

    #[test]
    fn cover_twists_in_band_and_grounds_past_it_where_the_off_diagonal_overflows() {
        assert!(cycle_cover(2f64.powi(970)).floats(), "twists in band");
        assert!(!cycle_cover(2f64.powi(1000)).floats(), "grounds past it");
    }
}
