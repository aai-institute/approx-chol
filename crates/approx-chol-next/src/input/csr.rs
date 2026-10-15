use std::borrow::Cow;

use num_traits::PrimInt;

use crate::in_range;
use crate::Real;

/// Borrowed CSR claiming a symmetric `A`; the claim is checked when it becomes an [`Sgdd`](crate::Sgdd).
#[derive(Clone, Copy, Debug)]
pub struct Csr<'a, T, I = u32> {
    n: usize,
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
}

impl<'a, T, I: PrimInt> Csr<'a, T, I> {
    /// Checks the arrays' shape, so no row slice can fail.
    pub fn new(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, CsrError> {
        if row_ptrs.len() != n + 1 {
            let found = row_ptrs.len();
            return Err(CsrError::LengthMismatch {
                expected: n + 1,
                found,
            });
        }
        if row_ptrs[0] != I::zero() {
            return Err(CsrError::RowPointers { row: 0 });
        }
        // Compared in `I`, so a negative pointer fails at the first row it closes.
        if let Some(row) = (0..n).find(|&i| row_ptrs[i] > row_ptrs[i + 1]) {
            return Err(CsrError::RowPointers { row });
        }
        let nnz = row_ptrs[n]
            .to_usize()
            .ok_or(CsrError::RowPointers { row: n })?;
        for found in [col_indices.len(), values.len()] {
            if found != nnz {
                return Err(CsrError::LengthMismatch {
                    expected: nnz,
                    found,
                });
            }
        }
        Ok(Self {
            n,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Borrowed CSR of the upper triangle of a symmetric `A`, diagonal included.
#[derive(Clone, Copy, Debug)]
pub struct UpperCsr<'a, T, I = u32>(pub(crate) Csr<'a, T, I>);

impl<'a, T, I: PrimInt> UpperCsr<'a, T, I> {
    /// The shape checks of [`Csr::new`]; an entry below the diagonal is caught at reduction.
    pub fn new(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, CsrError> {
        Csr::new(n, row_ptrs, col_indices, values).map(Self)
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.0.n
    }
}

/// A shape-checked index is non-negative and fits `usize`.
fn index<I: PrimInt>(k: I) -> usize {
    k.to_usize().expect("a shape-checked index fits usize")
}

/// Strictly ascending columns per row; scipy already emits them, so only rare input pays for a copy.
pub(crate) struct Canonical<'a, T: Clone, I: Clone> {
    n: usize,
    row_ptrs: Cow<'a, [I]>,
    col_indices: Cow<'a, [I]>,
    values: Cow<'a, [T]>,
}

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    /// Reads no value, sparing a stream: [`walk`](Self::walk) checks each as it reads it.
    pub(crate) fn of(csr: Csr<'a, T, I>) -> Result<Self, CsrError> {
        match is_canonical(&csr)? {
            true => Ok(Self {
                n: csr.n,
                row_ptrs: Cow::Borrowed(csr.row_ptrs),
                col_indices: Cow::Borrowed(csr.col_indices),
                values: Cow::Borrowed(csr.values),
            }),
            false => rewrite(&csr),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    fn arrays(&self) -> (&[I], &[I], &[T]) {
        (&self.row_ptrs, &self.col_indices, &self.values)
    }

    /// Row `i`'s canonical entries, stored zeros included.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let (row_ptrs, col_indices, values) = self.arrays();
        let span = index(row_ptrs[i])..index(row_ptrs[i + 1]);
        (col_indices[span.clone()].iter().map(|&j| index(j))).zip(values[span].iter().copied())
    }

    /// Row `i`'s stored entries above its diagonal, stored zeros included.
    pub(crate) fn upper_len(&self, i: usize) -> usize {
        let (row_ptrs, col_indices, _) = self.arrays();
        let row = &col_indices[index(row_ptrs[i])..index(row_ptrs[i + 1])];
        row.len() - row.partition_point(|&j| index(j) <= i)
    }

    /// Each nonzero `(i, j)`, `i < j`, mirrors checked by a per-row cursor (binary search: +6-15% build).
    pub(crate) fn walk(&self, mut edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let mut cursors = row_ptrs[..self.n].to_vec();
        for row in 0..self.n() {
            let row_end = row_ptrs[row + 1];
            // Claimed like any mirror: claiming diagonals up front would skip those below.
            let diagonal = self.claim(&mut cursors, row, row)?;
            if !in_range(diagonal) {
                return Err(CsrError::OutOfRange { row });
            }
            let mut cursor = cursors[row];
            while cursor < row_end {
                let col = index(col_indices[index(cursor)]);
                let upper = values[index(cursor)];
                cursor = cursor + I::one();
                if !in_range(upper) {
                    return Err(CsrError::OutOfRange { row });
                }
                // Duplicates can coalesce to exactly zero, which contributes no edge.
                if upper == T::zero() {
                    continue;
                }
                let lower = self.claim(&mut cursors, col, row)?;
                if !approximately_equal(upper, lower) {
                    return Err(CsrError::NotSymmetric { row });
                }
                edge(row, col, upper);
            }
        }
        Ok(())
    }

    /// Advances row `row`'s cursor to `col`; a merge-join only because each entry is claimed once. Stored zeros count as absent.
    fn claim(&self, cursors: &mut [I], row: usize, col: usize) -> Result<T, CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let row_end = row_ptrs[row + 1];
        let mut cursor = cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = index(col_indices[index(cursor)]);
            if at > col {
                break;
            }
            let value = values[index(cursor)];
            if !value.is_finite() {
                return Err(CsrError::OutOfRange { row });
            }
            cursor = cursor + I::one();
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                cursors[row] = cursor;
                return Err(CsrError::NotSymmetric { row });
            }
        }
        cursors[row] = cursor;
        Ok(found)
    }

    /// Each nonzero `(i, j)`, `i < j`, of upper storage, symmetric by shape: no mirror is found, only a row's first column checked.
    pub(crate) fn walk_upper(&self, mut edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        for row in 0..self.n() {
            let span = index(row_ptrs[row])..index(row_ptrs[row + 1]);
            let columns = &col_indices[span.clone()];
            // Columns ascend once canonical, so the first decides that none lies below the diagonal.
            if columns.first().is_some_and(|&j| index(j) < row) {
                return Err(CsrError::BelowDiagonal { row });
            }
            for (&col, &value) in columns.iter().zip(&values[span]) {
                if !in_range(value) {
                    return Err(CsrError::OutOfRange { row });
                }
                let col = index(col);
                if col > row && value != T::zero() {
                    edge(row, col, value);
                }
            }
        }
        Ok(())
    }
}

/// Mirrors within 8 ulps of the larger are equal: scipy's assembly rounds the two triangles apart.
fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

/// Accumulated, not short-circuited: the `all` form measured 0.4-2% slower.
fn is_canonical<T, I: PrimInt>(csr: &Csr<'_, T, I>) -> Result<bool, CsrError> {
    // `None` means `I` cannot represent `n`, so every `I` value is below it.
    let limit = I::from(csr.n);
    let mut canonical = true;
    for (row, bounds) in csr.row_ptrs.windows(2).enumerate() {
        let columns = &csr.col_indices[index(bounds[0])..index(bounds[1])];
        if columns
            .iter()
            .any(|&j| j < I::zero() || limit.is_some_and(|limit| j >= limit))
        {
            return Err(CsrError::ColumnOutOfRange { row });
        }
        for pair in columns.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    Ok(canonical)
}

/// Only non-canonical input pays this copy, in its own index type.
fn rewrite<'a, T: Real, I: PrimInt>(csr: &Csr<'_, T, I>) -> Result<Canonical<'a, T, I>, CsrError> {
    let nnz = csr.col_indices.len();
    let mut row_ptrs = Vec::with_capacity(csr.n + 1);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(I, T)> = Vec::new();
    row_ptrs.push(I::zero());
    for (row, bounds) in csr.row_ptrs.windows(2).enumerate() {
        let (from, to) = (index(bounds[0]), index(bounds[1]));
        entries.clear();
        entries.extend(
            csr.col_indices[from..to]
                .iter()
                .copied()
                .zip(csr.values[from..to].iter().copied()),
        );
        // One row's degree, not nnz. Stable, so duplicates sum in stored order.
        entries.sort_by_key(|&(col, _)| col);
        for group in entries.chunk_by(|left, right| left.0 == right.0) {
            let folded = group[1..].iter().fold(group[0].1, |sum, &(_, v)| sum + v);
            // Only this fold can overflow; caught here so downstream stays all-finite.
            if !folded.is_finite() {
                return Err(CsrError::OutOfRange { row });
            }
            col_indices.push(group[0].0);
            values.push(folded);
        }
        // At most the caller's own `nnz`, which its last row pointer already holds in `I`.
        row_ptrs.push(I::from(col_indices.len()).expect("a coalesced count fits the input's type"));
    }
    Ok(Canonical {
        n: csr.n,
        row_ptrs: Cow::Owned(row_ptrs),
        col_indices: Cow::Owned(col_indices),
        values: Cow::Owned(values),
    })
}

/// Why a CSR is not a symmetric matrix; `row` is in the caller's numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrError {
    /// An array's length disagrees with `n` or with `row_ptrs`.
    LengthMismatch {
        /// The length the other arrays fix.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// `row_ptrs` does not start at `0`, or falls or turns negative at `row`.
    RowPointers {
        /// The first row whose span is not a range.
        row: usize,
    },
    /// A column index lies outside `0..n`.
    ColumnOutOfRange {
        /// The offending row.
        row: usize,
    },
    /// An entry has no mirror within 8 ulps.
    NotSymmetric {
        /// The offending row.
        row: usize,
    },
    /// An upper triangle stores an entry, or a repeated column's sum, below its diagonal.
    BelowDiagonal {
        /// The offending row.
        row: usize,
    },
    /// An entry or a repeated column's sum is not finite, or nonzero below `MIN_POSITIVE/EPSILON`.
    OutOfRange {
        /// The offending row.
        row: usize,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "CSR array has length {found}, expected {expected}")
            }
            Self::RowPointers { row } => {
                write!(f, "row pointers do not ascend from 0 at row {row}")
            }
            Self::ColumnOutOfRange { row } => {
                write!(f, "row {row} has a column index outside 0..n")
            }
            Self::NotSymmetric { row } => write!(f, "row {row} has an entry with no equal mirror"),
            Self::BelowDiagonal { row } => {
                write!(
                    f,
                    "row {row} of an upper triangle has an entry below the diagonal"
                )
            }
            Self::OutOfRange { row } => {
                write!(f, "row {row} has an entry outside the representable range")
            }
        }
    }
}

impl std::error::Error for CsrError {}
