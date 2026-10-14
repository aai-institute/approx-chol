//! Sparse storage shapes every stage reads.

use std::sync::Arc;

use crate::{narrow, Real};

/// Compressed sparse rows.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(crate) struct Rows<T> {
    /// Per row, so `usize` costs nothing per entry: an exact complement can pass `u32::MAX` entries.
    ptrs: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<T>,
    n_cols: usize,
}

impl<T: Real> Rows<T> {
    /// Panics unless `ptrs` ascends from `0` and each row's columns strictly ascend below `n_cols`.
    pub(crate) fn new(ptrs: Vec<usize>, indices: Vec<u32>, values: Vec<T>, n_cols: usize) -> Self {
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
                    && row.last().is_none_or(|&j| (j as usize) < self.n_cols)
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

    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let span = self.ptrs[i]..self.ptrs[i + 1];
        (self.indices[span.clone()].iter().map(|&j| j as usize)).zip(&self.values[span])
    }

    /// Each value mapped, the pattern kept.
    pub(crate) fn map<U: Real>(self, f: impl FnMut(T) -> U) -> Rows<U> {
        Rows {
            ptrs: self.ptrs,
            indices: self.indices,
            values: self.values.into_iter().map(f).collect(),
            n_cols: self.n_cols,
        }
    }

    pub(crate) fn transpose(&self) -> Self {
        let mut ptrs = vec![0; self.n_cols + 1];
        for &j in &self.indices {
            ptrs[j as usize + 1] += 1;
        }
        for j in 0..self.n_cols {
            ptrs[j + 1] += ptrs[j];
        }
        let mut next = ptrs[..self.n_cols].to_vec();
        let mut indices = vec![0; self.indices.len()];
        let mut values = vec![T::zero(); self.values.len()];
        for i in 0..self.n_rows() {
            for (j, &x) in self.row(i) {
                indices[next[j]] = i as u32;
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

    /// `Σ|x|` per row and per column, in `f64` in row order.
    fn abs_sums(&self) -> (Vec<f64>, Vec<f64>) {
        let mut cols = vec![0.0; self.n_cols];
        let rows = (0..self.n_rows())
            .map(|i| {
                let mut sum = 0.0;
                for (j, x) in self.row(i) {
                    let x = crate::widen(x.abs());
                    sum += x;
                    cols[j] += x;
                }
                sum
            })
            .collect();
        (rows, cols)
    }

    fn nonpositive(&self) -> bool {
        self.values.iter().all(|&x| x <= T::zero())
    }

    pub(crate) fn positive(&self) -> bool {
        self.values.iter().all(|&x| x > T::zero())
    }

    fn abs_total(&self) -> f64 {
        crate::compensated_sum(self.values.iter().map(|&x| crate::widen(x.abs())))
    }

    /// `x_ij ← r_i x_ij c_j` for signs `r`, `c`: exact, so never out of range.
    fn signed(self, r: &[T], c: &[T]) -> Self {
        (self.scaled_seeing(r, c, |_, _, _| {})).expect("a sign moves no magnitude")
    }

    /// Scaled by `r` on rows and `c` on columns, with the new [`abs_sums`](Self::abs_sums) from the same pass.
    fn scaled_abs_sums(self, r: &[T], c: &[T]) -> Option<(Self, Vec<f64>, Vec<f64>)> {
        let (mut rows, mut cols) = (vec![0.0; self.n_rows()], vec![0.0; self.n_cols]);
        let scaled = self.scaled_seeing(r, c, |i, j, x| {
            let x = crate::widen(x.abs());
            rows[i] += x;
            cols[j] += x;
        })?;
        Some((scaled, rows, cols))
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
                let j = self.indices[p] as usize;
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
pub(crate) struct UpperAdjacency<T> {
    rows: Rows<T>,
}

impl<T: Real> UpperAdjacency<T> {
    /// Panics unless square, every column above its row, and no stored zero.
    pub(crate) fn new(rows: Rows<T>) -> Self {
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

    /// Row `i`'s neighbors above it.
    pub(crate) fn upper(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        self.rows.row(i)
    }
}

/// A component's off-diagonal by shape: `A_c`'s signed one, or `M`'s once congruent.
#[derive(Clone, Debug)]
pub(crate) enum OffDiagonal<T> {
    General(UpperAdjacency<T>),
    /// `C` of `[D_e C; Cᵀ D_k]`, both sides nonempty, the larger eliminated first as its rows.
    Bipartite(Arc<Rows<T>>),
}

impl<T: Real> OffDiagonal<T> {
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::General(adjacency) => adjacency.n(),
            Self::Bipartite(cross) => cross.n_rows() + cross.n_cols(),
        }
    }

    pub(crate) fn nonpositive(&self) -> bool {
        self.rows().nonpositive()
    }

    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        let (first, second) = self.rows().abs_sums();
        self.merged(first, second)
    }

    /// `Σ_ij |a_ij|` over both triangles, compensated.
    pub(crate) fn abs_total(&self) -> f64 {
        2.0 * self.rows().abs_total()
    }

    /// `S X S` for signs `S`: a sign moves no magnitude, so no entry leaves range.
    pub(crate) fn signed(self, s: &[T]) -> Self {
        match self {
            Self::General(adjacency) => Self::General(UpperAdjacency {
                rows: adjacency.rows.signed(s, s),
            }),
            Self::Bipartite(cross) => {
                let (eliminated, kept) = s.split_at(cross.n_rows());
                Self::Bipartite(Arc::new(
                    Arc::unwrap_or_clone(cross).signed(eliminated, kept),
                ))
            }
        }
    }

    /// `F X F` with the new row sums from the same pass; `None` where an entry leaves range.
    pub(crate) fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        let (scaled, first, second) = match self {
            Self::General(adjacency) => {
                let (rows, upper, lower) = adjacency.rows.scaled_abs_sums(f, f)?;
                (Self::General(UpperAdjacency { rows }), upper, lower)
            }
            Self::Bipartite(cross) => {
                let (eliminated, kept) = f.split_at(cross.n_rows());
                let (cross, rows, cols) =
                    Arc::unwrap_or_clone(cross).scaled_abs_sums(eliminated, kept)?;
                (Self::Bipartite(Arc::new(cross)), rows, cols)
            }
        };
        let sums = scaled.merged(first, second);
        Some((scaled, sums))
    }

    fn rows(&self) -> &Rows<T> {
        match self {
            Self::General(adjacency) => &adjacency.rows,
            Self::Bipartite(cross) => cross,
        }
    }

    /// An upper triangle's row and column sums add; a bipartite block's sides chain.
    fn merged(&self, first: Vec<f64>, second: Vec<f64>) -> Vec<T> {
        match self {
            Self::General(_) => (first.iter().zip(second))
                .map(|(u, l)| narrow(u + l))
                .collect(),
            Self::Bipartite(_) => first.into_iter().chain(second).map(narrow).collect(),
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Rows);

#[cfg(feature = "serde")]
impl<T: Real> Rows<T> {
    fn checked(self) -> Option<Self> {
        let values =
            self.values.len() == self.indices.len() && self.values.iter().all(|v| v.is_finite());
        (self.shaped() && values).then_some(self)
    }
}
