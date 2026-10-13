//! Sparse storage shapes every stage reads.

use std::sync::OnceLock;

use crate::Real;

/// Finite, and zero or at least `MIN_POSITIVE/EPSILON`, so no product turns subnormal (#163).
pub(crate) fn in_range<T: Real>(x: T) -> bool {
    x.is_finite() && (x == T::zero() || x.abs() >= T::min_positive_value() / T::epsilon())
}

/// Every `f64` sum starts here: a `Float` always casts up.
pub(crate) fn widen<T: Real>(x: T) -> f64 {
    x.to_f64().expect("a float casts to f64")
}

/// An `f64` sum in `T`: past `T::MAX` it reads infinite, which every range screen rejects.
pub(crate) fn narrow<T: Real>(sum: f64) -> T {
    T::from(sum).unwrap_or_else(T::infinity)
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
pub(crate) fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}

/// Compressed sparse rows.
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
        for span in ptrs.windows(2) {
            assert!(span[0] <= span[1], "row pointers ascend");
            let row = &indices[span[0]..span[1]];
            assert!(
                row.windows(2).all(|j| j[0] < j[1]),
                "columns strictly ascend"
            );
            assert!(
                row.last().is_none_or(|&j| (j as usize) < n_cols),
                "columns below n_cols"
            );
        }
        Self {
            ptrs,
            indices,
            values,
            n_cols,
        }
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

    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let span = self.ptrs[i]..self.ptrs[i + 1];
        (self.indices[span.clone()].iter().map(|&j| j as usize)).zip(&self.values[span])
    }

    /// Serial: `ct` builds it lazily inside parallel solves, where a rayon initializer deadlocks.
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
                    let x = widen(x.abs());
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

    /// Entries per row and per column.
    fn degrees(&self) -> (Vec<usize>, Vec<usize>) {
        let rows = self.ptrs.windows(2).map(|span| span[1] - span[0]).collect();
        let mut cols = vec![0; self.n_cols];
        for &j in &self.indices {
            cols[j as usize] += 1;
        }
        (rows, cols)
    }

    /// `x_ij ← r_i x_ij c_j`; `None` if a product leaves [`in_range`] or reaches zero.
    fn scaled(self, r: &[T], c: &[T]) -> Option<Self> {
        self.scaled_seeing(r, c, |_, _, _| {})
    }

    /// As [`scaled`](Self::scaled), with the new [`abs_sums`](Self::abs_sums) from the same pass.
    fn scaled_abs_sums(self, r: &[T], c: &[T]) -> Option<(Self, Vec<f64>, Vec<f64>)> {
        let (mut rows, mut cols) = (vec![0.0; self.n_rows()], vec![0.0; self.n_cols]);
        let scaled = self.scaled_seeing(r, c, |i, j, x| {
            let x = widen(x.abs());
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
                if x == T::zero() || !in_range(x) {
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
        let upper = (0..rows.n_rows()).all(|i| rows.row(i).all(|(j, _)| j > i));
        assert!(upper, "every column above its row");
        assert!(!rows.values.contains(&T::zero()), "no stored zero");
        Self { rows }
    }

    pub(crate) fn n(&self) -> usize {
        self.rows.n_rows()
    }

    /// `Σ_j |x_ij|` over both triangles.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        let (upper, lower) = self.rows.abs_sums();
        upper
            .iter()
            .zip(lower)
            .map(|(u, l)| narrow(u + l))
            .collect()
    }

    pub(crate) fn nonpositive(&self) -> bool {
        self.rows.nonpositive()
    }

    /// Entries per row over both triangles.
    pub(crate) fn degrees(&self) -> Vec<usize> {
        let (upper, lower) = self.rows.degrees();
        upper.iter().zip(lower).map(|(u, l)| u + l).collect()
    }

    /// Row `i`'s neighbors above it.
    pub(crate) fn upper(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        self.rows.row(i)
    }

    /// `F X F`; `None` if an entry leaves [`in_range`].
    pub(crate) fn scaled(self, f: &[T]) -> Option<Self> {
        let rows = self.rows.scaled(f, f)?;
        Some(Self { rows })
    }

    /// As [`scaled`](Self::scaled), with the new row sums from the same pass.
    pub(crate) fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        let (rows, upper, lower) = self.rows.scaled_abs_sums(f, f)?;
        let sums = (upper.iter().zip(lower))
            .map(|(u, l)| narrow(u + l))
            .collect();
        Some((Self { rows }, sums))
    }
}

/// One component's `C` of `[D₁ C; Cᵀ D₂]`, rows the eliminated side, `Cᵀ` built on first read.
#[derive(Clone, Debug)]
pub(crate) struct CrossBlock<T> {
    c: Rows<T>,
    ct: OnceLock<Rows<T>>,
}

impl<T: Real> CrossBlock<T> {
    /// Panics unless both sides are nonempty, the larger as rows, with no stored zero.
    pub(crate) fn new(c: Rows<T>) -> Self {
        assert!(c.n_rows() > 0 && c.n_cols() > 0, "both sides hold a vertex");
        assert!(c.n_rows() >= c.n_cols(), "the larger side is eliminated");
        assert!(!c.values.contains(&T::zero()), "no stored zero");
        Self {
            c,
            ct: OnceLock::new(),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.n_eliminated() + self.n_kept()
    }

    pub(crate) fn n_eliminated(&self) -> usize {
        self.c.n_rows()
    }

    pub(crate) fn n_kept(&self) -> usize {
        self.c.n_cols()
    }

    pub(crate) fn c(&self) -> &Rows<T> {
        &self.c
    }

    pub(crate) fn ct(&self) -> &Rows<T> {
        self.ct.get_or_init(|| self.c.transpose())
    }

    /// `Σ|c|` per row of `[0 C; Cᵀ 0]`: eliminated rows, then kept ones.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        let (eliminated, kept) = self.c.abs_sums();
        eliminated.into_iter().chain(kept).map(narrow).collect()
    }

    pub(crate) fn nonpositive(&self) -> bool {
        self.c.nonpositive()
    }

    /// Every entry positive: a signless block, folded by negating its kept side.
    pub(crate) fn positive(&self) -> bool {
        self.c.values.iter().all(|&x| x > T::zero())
    }

    /// Entries per row of `[0 C; Cᵀ 0]`: eliminated rows, then kept ones.
    pub(crate) fn degrees(&self) -> Vec<usize> {
        let (eliminated, kept) = self.c.degrees();
        eliminated.into_iter().chain(kept).collect()
    }

    /// `F C F` over the block's local order, a fresh block so no stale `Cᵀ` survives; `None` out of range.
    pub(crate) fn scaled(self, f: &[T]) -> Option<Self> {
        let (eliminated, kept) = f.split_at(self.n_eliminated());
        Some(Self::new(self.c.scaled(eliminated, kept)?))
    }

    /// As [`scaled`](Self::scaled), with the new row sums from the same pass.
    pub(crate) fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        let (eliminated, kept) = f.split_at(self.n_eliminated());
        let (c, rows, cols) = self.c.scaled_abs_sums(eliminated, kept)?;
        let sums = rows.into_iter().chain(cols).map(narrow).collect();
        Some((Self::new(c), sums))
    }
}
