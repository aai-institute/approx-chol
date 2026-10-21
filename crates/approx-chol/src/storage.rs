//! Sparse storage shapes every stage reads.

use crate::{narrow, widen, Index, Real};

/// Compressed sparse rows.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(crate) struct Rows<T, I> {
    /// Per row, so `usize` costs nothing per entry: an exact complement can pass `I::MAX` entries.
    ptrs: Vec<usize>,
    indices: Vec<I>,
    values: Vec<T>,
    n_cols: usize,
}

impl<T: Real, I: Index> Rows<T, I> {
    /// Panics unless `ptrs` ascends from `0` and each row's columns strictly ascend below `n_cols`.
    pub(crate) fn new(ptrs: Vec<usize>, indices: Vec<I>, values: Vec<T>, n_cols: usize) -> Self {
        assert_eq!(ptrs.first(), Some(&0), "rows start at the first entry");
        assert_eq!(
            ptrs.last(),
            Some(&indices.len()),
            "rows end at the last entry"
        );
        assert_eq!(indices.len(), values.len(), "one value per entry");
        let rows = Self {
            ptrs,
            indices,
            values,
            n_cols,
        };
        debug_assert!(
            rows.shaped(),
            "pointers and columns ascend, columns below n_cols"
        );
        rows
    }

    /// Pointers span `indices` and ascend; each row's columns strictly ascend below `n_cols`.
    fn shaped(&self) -> bool {
        let nnz = self.indices.len();
        let bounded = self.ptrs.first() == Some(&0) && self.ptrs.last() == Some(&nnz);
        // Every pointer ascends before any row is sliced: a middle one past `nnz` would panic.
        bounded
            && self.ptrs.windows(2).all(|span| span[0] <= span[1])
            && self.ptrs.windows(2).all(|span| {
                let row = &self.indices[span[0]..span[1]];
                row.windows(2).all(|j| j[0] < j[1])
                    && row.last().is_none_or(|&j| j.index() < self.n_cols)
            })
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.ptrs.len() - 1
    }

    pub(crate) fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub(crate) fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn row_len(&self, i: usize) -> usize {
        self.ptrs[i + 1] - self.ptrs[i]
    }

    // Outlined, every short row pays a call: solves through degree-2 eliminations cost 18% more.
    #[inline(always)]
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let span = self.ptrs[i]..self.ptrs[i + 1];
        (self.indices[span.clone()].iter().map(|&j| j.index())).zip(&self.values[span])
    }

    /// `out = |X| x`.
    pub(crate) fn abs_times(&self, x: &[f64], out: &mut [f64]) {
        for (i, out) in out.iter_mut().enumerate() {
            *out = self.row(i).map(|(j, &a)| widen(a.abs()) * x[j]).sum();
        }
    }

    /// `out = |X|ᵀ x`.
    pub(crate) fn abs_transpose_times(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (i, &x) in x.iter().enumerate() {
            self.row(i).for_each(|(j, &a)| out[j] += widen(a.abs()) * x);
        }
    }

    /// `out = |X|ᵀ diag(s)² |X| diag(t) x` in one pass, each row read twice while cached.
    pub(crate) fn abs_gram_times(&self, s: &[f64], t: &[f64], x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (span, &s) in self.ptrs.windows(2).zip(s) {
            let row = (self.indices[span[0]..span[1]].iter()).zip(&self.values[span[0]..span[1]]);
            let sum: f64 = row
                .clone()
                .map(|(&j, &a)| widen(a.abs()) * (x[j.index()] * t[j.index()]))
                .sum();
            let sum = s * s * sum;
            row.for_each(|(&j, &a)| out[j.index()] += widen(a.abs()) * sum);
        }
    }

    /// Row `i` times `by[i]`.
    pub(crate) fn scale_rows(&mut self, by: &[T]) {
        for (span, &s) in self.ptrs.windows(2).zip(by) {
            self.values[span[0]..span[1]]
                .iter_mut()
                .for_each(|v| *v = *v * s);
        }
    }

    pub(crate) fn transpose(&self) -> Self {
        let mut ptrs = vec![0; self.n_cols + 1];
        for &j in &self.indices {
            ptrs[j.index() + 1] += 1;
        }
        for j in 0..self.n_cols {
            ptrs[j + 1] += ptrs[j];
        }
        let mut next = ptrs[..self.n_cols].to_vec();
        let mut indices = vec![I::zero(); self.indices.len()];
        let mut values = vec![T::zero(); self.values.len()];
        for i in 0..self.n_rows() {
            for (j, &x) in self.row(i) {
                indices[next[j]] = I::from_index(i);
                values[next[j]] = x;
                next[j] += 1;
            }
        }
        Self {
            ptrs,
            indices,
            values,
            n_cols: self.n_rows(),
        }
    }

    /// `Σ|x|` in `f64`, row `i`'s at `i`, column `j`'s at `offset + j`: a row lands whole, after any column sharing its slot.
    pub(crate) fn abs_sums(&self, offset: usize) -> Vec<f64> {
        let mut sums = vec![0.0; offset + self.n_cols];
        for i in 0..self.n_rows() {
            let mut sum = 0.0;
            for (j, x) in self.row(i) {
                let x = widen(x.abs());
                sum += x;
                sums[offset + j] += x;
            }
            sums[i] += sum;
        }
        sums
    }

    fn nonpositive(&self) -> bool {
        self.values.iter().all(|&x| x <= T::zero())
    }

    pub(crate) fn positive(&self) -> bool {
        self.values.iter().all(|&x| x > T::zero())
    }

    /// `|a_ij| + |a_ji|` per stored entry, in `f64`: the other triangle mirrors it.
    fn abs_mirrored(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().map(|&x| 2.0 * widen(x.abs()))
    }

    /// `x_ij ← r_i x_ij c_j` for signs `r`, `c`: exact, so never out of range.
    fn signed(self, r: &[T], c: &[T]) -> Self {
        (self.scaled_seeing(r, c, |_, _, _| {})).expect("a sign moves no magnitude")
    }

    /// Scaled by `r` on rows and `c` on columns, with the new [`abs_sums`](Self::abs_sums) from the same pass.
    fn scaled_abs_sums(self, r: &[T], c: &[T], offset: usize) -> Option<(Self, Vec<f64>)> {
        let mut sums = vec![0.0; offset + self.n_cols];
        let mut row = (0, 0.0);
        let scaled = self.scaled_seeing(r, c, |i, j, x| {
            let x = widen(x.abs());
            if i != row.0 {
                sums[row.0] += row.1;
                row = (i, 0.0);
            }
            row.1 += x;
            sums[offset + j] += x;
        })?;
        sums[row.0] += row.1;
        Some((scaled, sums))
    }

    #[inline(always)]
    fn scaled_seeing(
        mut self,
        r: &[T],
        c: &[T],
        mut seen: impl FnMut(usize, usize, T),
    ) -> Option<Self> {
        for (i, span) in self.ptrs.windows(2).enumerate() {
            for p in span[0]..span[1] {
                let j = self.indices[p].index();
                let x = r[i] * self.values[p] * c[j];
                if x == T::zero() || !crate::in_range(x) {
                    return None;
                }
                self.values[p] = x;
                seen(i, j, x);
            }
        }
        Some(self)
    }
}

/// A symmetric off-diagonal's strict upper triangle by rows: each edge once.
#[derive(Clone, Debug)]
pub(crate) struct UpperAdjacency<T, I> {
    rows: Rows<T, I>,
}

impl<T: Real, I: Index> UpperAdjacency<T, I> {
    /// Panics unless square, every column above its row, and no stored zero.
    pub(crate) fn new(rows: Rows<T, I>) -> Self {
        assert_eq!(rows.n_rows(), rows.n_cols, "square");
        debug_assert!(
            (0..rows.n_rows()).all(|i| rows.row(i).all(|(j, _)| j > i)),
            "every column above its row"
        );
        debug_assert!(!rows.values.contains(&T::zero()), "no stored zero");
        Self { rows }
    }

    pub(crate) fn n(&self) -> usize {
        self.rows.n_rows()
    }

    pub(crate) fn nonpositive(&self) -> bool {
        self.rows.nonpositive()
    }

    pub(crate) fn upper_len(&self, i: usize) -> usize {
        self.rows.row_len(i)
    }

    pub(crate) fn nnz(&self) -> usize {
        self.rows.nnz()
    }

    pub(crate) fn abs_mirrored(&self) -> impl Iterator<Item = f64> + '_ {
        self.rows.abs_mirrored()
    }

    /// Row `i`'s neighbors above it.
    pub(crate) fn upper(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        self.rows.row(i)
    }
}

/// A block's off-diagonal by shape: `A_c`'s signed one, or `M`'s once congruent.
#[derive(Clone, Debug)]
pub(crate) enum OffDiagonal<T, I> {
    General(UpperAdjacency<T, I>),
    /// `C` of `[D_e C; Cᵀ D_k]`, both sides nonempty, the larger eliminated first as its rows.
    Bipartite(Rows<T, I>),
}

impl<T: Real, I: Index> OffDiagonal<T, I> {
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::General(adjacency) => adjacency.n(),
            Self::Bipartite(cross) => cross.n_rows() + cross.n_cols(),
        }
    }

    pub(crate) fn nonpositive(&self) -> bool {
        self.stored().0.nonpositive()
    }

    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        let (rows, offset) = self.stored();
        rows.abs_sums(offset).into_iter().map(narrow).collect()
    }

    pub(crate) fn abs_mirrored(&self) -> impl Iterator<Item = f64> + '_ {
        self.stored().0.abs_mirrored()
    }

    /// `S X S` for signs `S`: a sign moves no magnitude, so no entry leaves range.
    pub(crate) fn signed(self, s: &[T]) -> Self {
        match self {
            Self::General(adjacency) => Self::General(UpperAdjacency {
                rows: adjacency.rows.signed(s, s),
            }),
            Self::Bipartite(cross) => {
                let (eliminated, kept) = s.split_at(cross.n_rows());
                Self::Bipartite(cross.signed(eliminated, kept))
            }
        }
    }

    /// `F X F` with the new row sums from the same pass; `None` where an entry leaves range.
    pub(crate) fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        let (scaled, sums) = match self {
            Self::General(adjacency) => {
                let (rows, sums) = adjacency.rows.scaled_abs_sums(f, f, 0)?;
                (Self::General(UpperAdjacency { rows }), sums)
            }
            Self::Bipartite(cross) => {
                let (eliminated, kept) = f.split_at(cross.n_rows());
                let (cross, sums) = cross.scaled_abs_sums(eliminated, kept, eliminated.len())?;
                (Self::Bipartite(cross), sums)
            }
        };
        Some((scaled, sums.into_iter().map(narrow).collect()))
    }

    /// Each row's off-diagonal count, both triangles.
    pub(crate) fn degrees(&self) -> Vec<usize> {
        let (rows, offset) = self.stored();
        let mut degree = vec![0usize; self.n()];
        for (i, span) in rows.ptrs.windows(2).enumerate() {
            degree[i] += span[1] - span[0];
            for &j in &rows.indices[span[0]..span[1]] {
                degree[offset + j.index()] += 1;
            }
        }
        degree
    }

    /// `out = |X| diag(s) y`, each stored entry read for both its ends.
    pub(crate) fn abs_times(&self, s: &[f64], y: &[f64], out: &mut [f64]) {
        let (rows, offset) = self.stored();
        out.fill(0.0);
        for (i, span) in rows.ptrs.windows(2).enumerate() {
            let x_i = y[i] * s[i];
            let mut sum = 0.0;
            for (&j, &a) in rows.indices[span[0]..span[1]]
                .iter()
                .zip(&rows.values[span[0]..span[1]])
            {
                let (a, j) = (widen(a.abs()), offset + j.index());
                sum += a * (y[j] * s[j]);
                out[j] += a * x_i;
            }
            out[i] += sum;
        }
    }

    /// The stored entries, and where their columns start in local order.
    pub(crate) fn stored(&self) -> (&Rows<T, I>, usize) {
        match self {
            Self::General(adjacency) => (&adjacency.rows, 0),
            Self::Bipartite(cross) => (cross, cross.n_rows()),
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Rows);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Rows<T, I> {
    fn checked(self) -> Option<Self> {
        let values =
            self.values.len() == self.indices.len() && self.values.iter().all(|v| v.is_finite());
        (self.shaped() && values).then_some(self)
    }
}
