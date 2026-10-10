use num_traits::PrimInt;

use crate::storage::in_range;
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
    /// Checks the arrays' shape, that `row_ptrs` ascends from `0`, and that every vertex and entry fits a `u32` index, so no row slice can fail.
    pub fn new(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, CsrError> {
        if n > u32::MAX as usize {
            return Err(CsrError::TooLarge { count: n });
        }
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
            return Err(CsrError::RowPointers { row: row as u32 });
        }
        let nnz = row_ptrs[n]
            .to_usize()
            .ok_or(CsrError::RowPointers { row: n as u32 })?;
        if nnz > u32::MAX as usize {
            return Err(CsrError::TooLarge { count: nnz });
        }
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

/// A shape-checked index is non-negative and fits `usize`.
fn index<I: PrimInt>(k: I) -> usize {
    k.to_usize().expect("a shape-checked index fits usize")
}

/// Strictly ascending columns per row; scipy already emits them, so only rare input pays for a copy.
pub(crate) struct Canonical<'a, T, I> {
    input: Csr<'a, T, I>,
    /// `None` when the caller's arrays are already canonical.
    rewritten: Option<Rewritten<T, I>>,
}

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    /// Reads no value on the canonical path, sparing a stream: [`walk`](Self::walk) checks each as it reads it.
    pub(crate) fn of(csr: Csr<'a, T, I>) -> Result<Self, CsrError> {
        let rewritten = match is_canonical(&csr)? {
            true => None,
            false => Some(rewrite(&csr)?),
        };
        Ok(Self {
            input: csr,
            rewritten,
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.input.n
    }

    fn arrays(&self) -> (&[I], &[I], &[T]) {
        match &self.rewritten {
            Some(r) => (&r.row_ptrs, &r.col_indices, &r.values),
            None => (
                self.input.row_ptrs,
                self.input.col_indices,
                self.input.values,
            ),
        }
    }

    /// Row `i`'s canonical entries, stored zeros included.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let (row_ptrs, col_indices, values) = self.arrays();
        let span = index(row_ptrs[i])..index(row_ptrs[i + 1]);
        (col_indices[span.clone()].iter().map(|&j| index(j))).zip(values[span].iter().copied())
    }

    /// Visits each nonzero `(i, j)`, `i < j`, with its upper value, checking mirrors by a monotone per-row cursor (search: +6-15% build).
    pub(crate) fn walk(&self, mut edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);
        for row in 0..self.n() {
            let row_end = index(row_ptrs[row + 1]) as u32;
            // Claimed like any mirror: claiming diagonals up front would skip those below.
            let diagonal = mirrors.claim(row, row)?;
            if !in_range(diagonal) {
                return Err(CsrError::OutOfRange { row: row as u32 });
            }
            let mut cursor = mirrors.cursors[row];
            while cursor < row_end {
                let col = index(col_indices[cursor as usize]);
                let upper = values[cursor as usize];
                cursor += 1;
                if !in_range(upper) {
                    return Err(CsrError::OutOfRange { row: row as u32 });
                }
                // Duplicates can coalesce to exactly zero, which contributes no edge.
                if upper == T::zero() {
                    continue;
                }
                let lower = mirrors.claim(col, row)?;
                if !approximately_equal(upper, lower) {
                    return Err(CsrError::NotSymmetric { row: row as u32 });
                }
                edge(row, col, upper);
            }
        }
        Ok(())
    }
}

/// A merge-join only because [`Canonical`] guarantees each entry is claimed once.
struct Mirrors<'a, T, I> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    cursors: Vec<u32>,
}

impl<'a, T: Real, I: PrimInt> Mirrors<'a, T, I> {
    fn new(row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T]) -> Self {
        let cursors = row_ptrs[..row_ptrs.len() - 1]
            .iter()
            .map(|&ptr| index(ptr) as u32)
            .collect();
        Self {
            row_ptrs,
            col_indices,
            values,
            cursors,
        }
    }

    /// Stored zeros count as absent.
    fn claim(&mut self, row: usize, col: usize) -> Result<T, CsrError> {
        let row_end = index(self.row_ptrs[row + 1]) as u32;
        let mut cursor = self.cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = index(self.col_indices[cursor as usize]);
            if at > col {
                break;
            }
            let value = self.values[cursor as usize];
            if !value.is_finite() {
                return Err(CsrError::OutOfRange { row: row as u32 });
            }
            cursor += 1;
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                self.cursors[row] = cursor;
                return Err(CsrError::NotSymmetric { row: row as u32 });
            }
        }
        self.cursors[row] = cursor;
        Ok(found)
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

/// Accumulated, not short-circuited: measured 0.4-2% of the build over the `all` form; a column outside `0..n` ends it.
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
            return Err(CsrError::ColumnOutOfRange { row: row as u32 });
        }
        for pair in columns.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    Ok(canonical)
}

/// Canonical arrays rebuilt from non-canonical input, in its own index type.
struct Rewritten<T, I> {
    row_ptrs: Vec<I>,
    col_indices: Vec<I>,
    values: Vec<T>,
}

/// Only non-canonical input pays this copy.
fn rewrite<T: Real, I: PrimInt>(csr: &Csr<'_, T, I>) -> Result<Rewritten<T, I>, CsrError> {
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
                return Err(CsrError::OutOfRange { row: row as u32 });
            }
            col_indices.push(group[0].0);
            values.push(folded);
        }
        // At most the caller's own `nnz`, which its last row pointer already holds in `I`.
        row_ptrs.push(I::from(col_indices.len()).expect("a coalesced count fits the input's type"));
    }
    Ok(Rewritten {
        row_ptrs,
        col_indices,
        values,
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
    /// More vertices or entries than a `u32` index reaches.
    TooLarge {
        /// The count that overflows.
        count: usize,
    },
    /// `row_ptrs` does not start at `0`, or falls or turns negative at `row`.
    RowPointers {
        /// The first row whose span is not a range.
        row: u32,
    },
    /// A column index lies outside `0..n`.
    ColumnOutOfRange {
        /// The offending row.
        row: u32,
    },
    /// An entry has no mirror within 8 ulps.
    NotSymmetric {
        /// The offending row.
        row: u32,
    },
    /// An entry, or a repeated column's sum, is not finite, or nonzero below `MIN_POSITIVE/EPSILON`, where products turn subnormal.
    OutOfRange {
        /// The offending row.
        row: u32,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "CSR array has length {found}, expected {expected}")
            }
            Self::TooLarge { count } => write!(f, "{count} exceeds a u32 index"),
            Self::RowPointers { row } => {
                write!(f, "row pointers do not ascend from 0 at row {row}")
            }
            Self::ColumnOutOfRange { row } => {
                write!(f, "row {row} has a column index outside 0..n")
            }
            Self::NotSymmetric { row } => write!(f, "row {row} has an entry with no equal mirror"),
            Self::OutOfRange { row } => {
                write!(f, "row {row} has an entry outside the representable range")
            }
        }
    }
}

impl std::error::Error for CsrError {}
