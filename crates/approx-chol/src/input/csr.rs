use std::borrow::Cow;
use std::ops::Range;

use num_traits::{cast, PrimInt};

use super::error::{dimension, length, InputError};
use super::index;
use crate::in_range;
use crate::Real;

/// Borrowed CSR claiming a symmetric `A`; shape is checked here, the claim when it factors.
#[derive(Clone, Copy, Debug)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
}

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// Checks the arrays' shape, so no row slice can fail.
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, InputError> {
        let n = n as usize;
        length("row_ptrs", n + 1, row_ptrs.len())?;
        length("values", col_indices.len(), values.len())?;
        let row_ptr_last = representable(row_ptrs, n)?;
        if row_ptrs[0] != I::zero() {
            representable(row_ptrs, 0)?;
            return Err(InputError::RowPtrs { row: 0 });
        }
        length("col_indices", row_ptr_last, col_indices.len())?;
        // From 0 and in order up to a decrease, only the pointer after it can be negative.
        if let Some(row) = (0..n).find(|&i| row_ptrs[i] > row_ptrs[i + 1]) {
            representable(row_ptrs, row + 1)?;
            return Err(InputError::RowPtrs { row });
        }
        Ok(Self {
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Row pointer array (length `n + 1`).
    pub fn row_ptrs(&self) -> &'a [I] {
        self.row_ptrs
    }

    /// Column index array (length `nnz`).
    pub fn col_indices(&self) -> &'a [I] {
        self.col_indices
    }

    /// Value array (length `nnz`).
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }
}

impl<T: Clone, I: PrimInt> CsrRef<'_, T, I> {
    /// Owned copy with `u32` indices; [`InputError::IndexOutOfRange`] if an index does not fit.
    pub fn to_owned_u32(&self) -> Result<OwnedCsr<T, u32>, InputError> {
        Ok(OwnedCsr {
            row_ptrs: cast_slice("row_ptrs", self.row_ptrs)?,
            col_indices: cast_slice("col_indices", self.col_indices)?,
            values: self.values.to_vec(),
        })
    }
}

/// Borrowed CSR of the upper triangle of a symmetric `A`, diagonal included.
#[derive(Clone, Copy, Debug)]
pub struct UpperCsrRef<'a, T = f64, I = u32>(CsrRef<'a, T, I>);

impl<'a, T, I: PrimInt> UpperCsrRef<'a, T, I> {
    /// The shape checks of [`CsrRef::new`]; a nonzero below the diagonal is caught when it factors.
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, InputError> {
        CsrRef::new(row_ptrs, col_indices, values, n).map(Self)
    }

    pub(crate) fn lower(self) -> CsrRef<'a, T, I> {
        self.0
    }
}

/// Owned CSR matrix, for sources that index in `usize`.
#[derive(Clone, Debug)]
pub struct OwnedCsr<T = f64, I = u32> {
    row_ptrs: Vec<I>,
    col_indices: Vec<I>,
    values: Vec<T>,
}

impl<T: Clone, I: PrimInt> OwnedCsr<T, I> {
    /// Owned CSR from `usize` arrays, shape-checked as [`CsrRef::new`]; an index past `I` is an error.
    pub fn try_from_usize(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[T],
        n: usize,
    ) -> Result<Self, InputError> {
        let n = dimension(n)?;
        let row_ptrs = cast_slice("row_ptrs", row_ptrs)?;
        let col_indices = cast_slice("col_indices", col_indices)?;
        CsrRef::new(&row_ptrs, &col_indices, values, n)?;
        Ok(Self {
            row_ptrs,
            col_indices,
            values: values.to_vec(),
        })
    }
}

impl<T, I: PrimInt> OwnedCsr<T, I> {
    /// Infallible: the constructors check shape and the fields are private.
    pub fn as_csr_ref(&self) -> CsrRef<'_, T, I> {
        CsrRef {
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values: &self.values,
        }
    }
}

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for CsrRef<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        owned.as_csr_ref()
    }
}

fn representable<I: PrimInt>(row_ptrs: &[I], entry: usize) -> Result<usize, InputError> {
    (row_ptrs[entry].to_usize()).ok_or(InputError::IndexOutOfRange {
        parameter: "row_ptrs",
        entry,
    })
}

/// Reserves up front: collecting into `Option<Vec<_>>` drops the size hint and cost 3.5x the traffic.
fn cast_slice<S: PrimInt, D: PrimInt>(
    parameter: &'static str,
    src: &[S],
) -> Result<Vec<D>, InputError> {
    let mut out = Vec::with_capacity(src.len());
    for (entry, &value) in src.iter().enumerate() {
        out.push(cast::<S, D>(value).ok_or(InputError::IndexOutOfRange { parameter, entry })?);
    }
    Ok(out)
}

/// Strictly ascending columns per row, all in range, values not yet read; scipy already emits them, so only rare input pays for a copy.
struct Unchecked<'a, T: Clone, I: Clone> {
    row_ptrs: Cow<'a, [I]>,
    col_indices: Cow<'a, [I]>,
    values: Cow<'a, [T]>,
}

/// A checked symmetric `A`: canonical rows, every value in range, symmetric by its mirrors or by upper storage.
pub(crate) struct Canonical<'a, T: Clone, I: Clone>(Unchecked<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    /// `A` stored whole: every value in range, each nonzero's mirror within 8 ulps.
    pub(crate) fn whole(csr: CsrRef<'a, T, I>) -> Result<Self, InputError> {
        let unchecked = Unchecked::of(csr)?;
        unchecked.check_mirrors()?;
        Ok(Self(unchecked))
    }

    /// `A` stored by its upper triangle: every value in range, no nonzero below the diagonal.
    pub(crate) fn upper(csr: CsrRef<'a, T, I>) -> Result<Self, InputError> {
        let unchecked = Unchecked::of(csr)?;
        unchecked.check_upper()?;
        Ok(Self(unchecked))
    }

    pub(crate) fn n(&self) -> usize {
        self.0.n()
    }

    /// Row `i`'s canonical entries, stored zeros included.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let (row_ptrs, _, _) = self.0.arrays();
        self.0.entries(index(row_ptrs[i])..index(row_ptrs[i + 1]))
    }

    /// Row `i`'s span above its diagonal, found from the row's end: reads those columns and one more.
    fn above_span(&self, i: usize) -> Range<usize> {
        let (row_ptrs, col_indices, _) = self.0.arrays();
        let end = index(row_ptrs[i + 1]);
        let row = &col_indices[index(row_ptrs[i])..end];
        end - row.iter().rev().take_while(|&&j| index(j) > i).count()..end
    }

    /// Row `i`'s entries above its diagonal, stored zeros included.
    pub(crate) fn above(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, T)> + '_ {
        self.0.entries(self.above_span(i))
    }

    /// Row `i`'s diagonal, zero when not stored, and its entries above it, stored zeros included.
    pub(crate) fn upper_row(
        &self,
        i: usize,
    ) -> (T, impl ExactSizeIterator<Item = (usize, T)> + '_) {
        let (row_ptrs, col_indices, values) = self.0.arrays();
        let above = self.above_span(i);
        let at = above.start.checked_sub(1);
        let stored = at.filter(|&k| k >= index(row_ptrs[i]) && index(col_indices[k]) == i);
        // A stored `−0` reads as absent, as it couples nothing.
        let diagonal = match stored.map(|k| values[k]) {
            Some(d) if d != T::zero() => d,
            _ => T::zero(),
        };
        (diagonal, self.0.entries(above))
    }
}

impl<'a, T: Real, I: PrimInt> Unchecked<'a, T, I> {
    /// Reads no value, sparing a stream, unless it rewrites: the checks after it read each.
    fn of(csr: CsrRef<'a, T, I>) -> Result<Self, InputError> {
        match is_canonical(&csr)? {
            true => Ok(Self {
                row_ptrs: Cow::Borrowed(csr.row_ptrs),
                col_indices: Cow::Borrowed(csr.col_indices),
                values: Cow::Borrowed(csr.values),
            }),
            false => rewrite(&csr),
        }
    }

    fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    fn arrays(&self) -> (&[I], &[I], &[T]) {
        (&self.row_ptrs, &self.col_indices, &self.values)
    }

    /// The entries in `span` of the arrays, stored zeros included.
    fn entries(&self, span: Range<usize>) -> impl ExactSizeIterator<Item = (usize, T)> + '_ {
        let (_, col_indices, values) = self.arrays();
        (col_indices[span.clone()].iter().map(|&j| index(j))).zip(values[span].iter().copied())
    }

    /// Each nonzero `(i, j)`, `i < j`, its mirror found by a per-row cursor (binary search: +6-15% build).
    fn check_mirrors(&self) -> Result<(), InputError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let mut cursors = row_ptrs[..self.n()].to_vec();
        for row in 0..self.n() {
            let row_end = row_ptrs[row + 1];
            // Claimed like any mirror: claiming diagonals up front would skip those below.
            let diagonal = self.claim(&mut cursors, row, row)?;
            if !in_range(diagonal) {
                return Err(InputError::InadmissibleRow { row });
            }
            let mut cursor = cursors[row];
            while cursor < row_end {
                let col = index(col_indices[index(cursor)]);
                let upper = values[index(cursor)];
                cursor = cursor + I::one();
                if !in_range(upper) {
                    return Err(InputError::InadmissibleRow { row });
                }
                // Duplicates can coalesce to exactly zero, which needs no mirror.
                if upper == T::zero() {
                    continue;
                }
                let lower = self.claim(&mut cursors, col, row)?;
                if !approximately_equal(upper, lower) {
                    return Err(InputError::NotSymmetric { row });
                }
            }
        }
        Ok(())
    }

    /// Advances row `row`'s cursor to `col`; a merge-join only because each entry is claimed once. Stored zeros count as absent.
    fn claim(&self, cursors: &mut [I], row: usize, col: usize) -> Result<T, InputError> {
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
                return Err(InputError::InadmissibleRow { row });
            }
            cursor = cursor + I::one();
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                cursors[row] = cursor;
                return Err(InputError::NotSymmetric { row });
            }
        }
        cursors[row] = cursor;
        Ok(found)
    }

    /// Upper storage is symmetric by shape: no mirror is found, and a nonzero below the diagonal is rejected.
    fn check_upper(&self) -> Result<(), InputError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        for row in 0..self.n() {
            let span = index(row_ptrs[row])..index(row_ptrs[row + 1]);
            for (&col, &value) in col_indices[span.clone()].iter().zip(&values[span]) {
                if !in_range(value) {
                    return Err(InputError::InadmissibleRow { row });
                }
                if index(col) < row && value != T::zero() {
                    return Err(InputError::OutsideTriangle { row });
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
fn is_canonical<T, I: PrimInt>(csr: &CsrRef<'_, T, I>) -> Result<bool, InputError> {
    // `None` means `I` cannot represent `n`, so every `I` value is below it.
    let limit = I::from(csr.n());
    let mut canonical = true;
    let outside = |j: I| j < I::zero() || limit.is_some_and(|limit| j >= limit);
    for bounds in csr.row_ptrs.windows(2) {
        let start = index(bounds[0]);
        let columns = &csr.col_indices[start..index(bounds[1])];
        if let Some(at) = columns.iter().position(|&j| outside(j)) {
            return Err(InputError::IndexOutOfRange {
                parameter: "col_indices",
                entry: start + at,
            });
        }
        for pair in columns.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    Ok(canonical)
}

/// Only non-canonical input pays this copy, in its own index type.
fn rewrite<'a, T: Real, I: PrimInt>(
    csr: &CsrRef<'_, T, I>,
) -> Result<Unchecked<'a, T, I>, InputError> {
    let nnz = csr.col_indices.len();
    let mut row_ptrs = Vec::with_capacity(csr.row_ptrs.len());
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
                return Err(InputError::InadmissibleRow { row });
            }
            col_indices.push(group[0].0);
            values.push(folded);
        }
        // At most the caller's own `nnz`, which its last row pointer already holds in `I`.
        row_ptrs.push(I::from(col_indices.len()).expect("a coalesced count fits the input's type"));
    }
    Ok(Unchecked {
        row_ptrs: Cow::Owned(row_ptrs),
        col_indices: Cow::Owned(col_indices),
        values: Cow::Owned(values),
    })
}
