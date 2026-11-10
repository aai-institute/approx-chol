use core::cmp::Ordering;
use std::borrow::Cow;
use std::ops::Range;

use num_traits::PrimInt;

use super::error::{length, value_in_range, InputError};
use super::index;
use crate::{diagonal_in_range, in_range, Real};

/// Borrowed CSR of a symmetric `A`, checked whole by [`CsrRef::new`].
#[derive(Clone, Copy, Debug)]
pub struct CsrRef<'a, T, I = u32>(pub(crate) CsrArrays<'a, T, I>);

impl<'a, T: Real, I: PrimInt> CsrRef<'a, T, I> {
    /// Checks shape, column range, every value [in range](InputError::OutOfRange), symmetry within 8 ulps and each [diagonal](InputError::Diagonal).
    pub fn new(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, InputError> {
        let arrays = CsrArrays::shaped(n, row_ptrs, col_indices, values)?;
        arrays.canonical()?.check_mirrors()?;
        Ok(Self(arrays))
    }
}

/// Borrowed CSR of the upper triangle of a symmetric `A`, diagonal included, checked whole by [`TriangleCsrRef::upper`].
#[derive(Clone, Copy, Debug)]
pub struct TriangleCsrRef<'a, T, I = u32>(pub(crate) CsrArrays<'a, T, I>);

impl<'a, T: Real, I: PrimInt> TriangleCsrRef<'a, T, I> {
    /// Checks shape, column range, every value [in range](InputError::OutOfRange), no nonzero below the diagonal and each [diagonal](InputError::Diagonal).
    pub fn upper(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, InputError> {
        let arrays = CsrArrays::shaped(n, row_ptrs, col_indices, values)?;
        arrays.canonical()?.check_upper()?;
        Ok(Self(arrays))
    }
}

/// The caller's arrays, borrowed, and whether their rows are already canonical.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CsrArrays<'a, T, I> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    order: Order,
}

/// Canonical rows have strictly ascending columns; others are sorted and coalesced each time they are read.
#[derive(Clone, Copy, Debug)]
enum Order {
    Canonical,
    Unsorted,
}

impl<'a, T: Real, I: PrimInt> CsrArrays<'a, T, I> {
    /// Row pointers in order and every column below `n`; no value read.
    fn shaped(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, InputError> {
        length("row_ptrs", n.saturating_add(1), row_ptrs.len())?;
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
        let order = match is_canonical(row_ptrs, col_indices)? {
            true => Order::Canonical,
            false => Order::Unsorted,
        };
        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            order,
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    /// Unsorted rows are rewritten here, at construction to check them and again to factor: keeping the copy would cost `Input` its `Copy`.
    fn canonical(&self) -> Result<Canonical<'a, T, I>, InputError> {
        match self.order {
            Order::Canonical => Ok(Canonical {
                row_ptrs: Cow::Borrowed(self.row_ptrs),
                col_indices: Cow::Borrowed(self.col_indices),
                values: Cow::Borrowed(self.values),
            }),
            Order::Unsorted => rewrite(self),
        }
    }
}

fn representable<I: PrimInt>(row_ptrs: &[I], entry: usize) -> Result<usize, InputError> {
    (row_ptrs[entry].to_usize()).ok_or(InputError::IndexOutOfRange {
        parameter: "row_ptrs",
        entry,
    })
}

/// A symmetric `A` in canonical rows; checked whenever reached from outside this module, as only [`Canonical::of`] is.
pub(crate) struct Canonical<'a, T: Clone, I: Clone> {
    row_ptrs: Cow<'a, [I]>,
    col_indices: Cow<'a, [I]>,
    values: Cow<'a, [T]>,
}

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    pub(crate) fn of(arrays: CsrArrays<'a, T, I>) -> Self {
        arrays.canonical().expect("checked when constructed")
    }

    pub(crate) fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    /// The entries in `span` of the arrays, stored zeros included.
    fn entries(&self, span: Range<usize>) -> impl ExactSizeIterator<Item = (usize, T)> + '_ {
        let columns = self.col_indices[span.clone()].iter().map(|&j| index(j));
        columns.zip(self.values[span].iter().copied())
    }

    fn span(&self, i: usize) -> Range<usize> {
        index(self.row_ptrs[i])..index(self.row_ptrs[i + 1])
    }

    /// Row `i`'s canonical entries, stored zeros included.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        self.entries(self.span(i))
    }

    /// Row `i`'s span above its diagonal, found from the row's end: reads those columns and one more.
    fn above_span(&self, i: usize) -> Range<usize> {
        let span = self.span(i);
        let end = span.end;
        end - self.col_indices[span]
            .iter()
            .rev()
            .take_while(|&&j| index(j) > i)
            .count()..end
    }

    /// Row `i`'s entries above its diagonal, stored zeros included.
    pub(crate) fn above(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, T)> + '_ {
        self.entries(self.above_span(i))
    }

    /// Row `i`'s diagonal, zero when not stored, and its entries above it, stored zeros included.
    pub(crate) fn upper_row(
        &self,
        i: usize,
    ) -> (T, impl ExactSizeIterator<Item = (usize, T)> + '_) {
        let above = self.above_span(i);
        let at = above.start.checked_sub(1);
        let stored = at.filter(|&k| k >= self.span(i).start && index(self.col_indices[k]) == i);
        // A stored `−0` reads as absent, as it couples nothing.
        let diagonal = match stored.map(|k| self.values[k]) {
            Some(d) if d != T::zero() => d,
            _ => T::zero(),
        };
        (diagonal, self.entries(above))
    }

    /// Each nonzero `(i, j)`, `i < j`, its mirror found by a per-row cursor (binary search: +6-15% build).
    fn check_mirrors(&self) -> Result<(), InputError> {
        let mut cursors = self.row_ptrs[..self.n()].to_vec();
        for row in 0..self.n() {
            let row_end = self.row_ptrs[row + 1];
            // Claimed like any mirror: claiming diagonals up front would skip those below.
            let diagonal = self.claim(&mut cursors, row, row)?;
            let mut cursor = cursors[row];
            while cursor < row_end {
                let at = index(cursor);
                let col = index(self.col_indices[at]);
                let upper = value_in_range("values", &self.values, at)?;
                cursor = cursor + I::one();
                // Duplicates can coalesce to exactly zero, which needs no mirror.
                if upper == T::zero() {
                    continue;
                }
                let lower = self.claim(&mut cursors, col, row)?;
                if !approximately_equal(upper, lower) {
                    return Err(InputError::NotSymmetric { row });
                }
            }
            // Every entry of the row is read and in range by now, so its own nonzeros are its edges.
            let coupled = || self.values[self.span(row)].iter().any(|&v| v != T::zero());
            check_diagonal(diagonal, coupled, row)?;
        }
        Ok(())
    }

    /// Advances row `row`'s cursor to `col`; a merge-join only because each entry is claimed once. Stored zeros count as absent.
    fn claim(&self, cursors: &mut [I], row: usize, col: usize) -> Result<T, InputError> {
        let row_end = self.row_ptrs[row + 1];
        let mut cursor = cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = index(cursor);
            let j = index(self.col_indices[at]);
            if j > col {
                break;
            }
            let value = value_in_range("values", &self.values, at)?;
            cursor = cursor + I::one();
            if j == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                return Err(InputError::NotSymmetric { row });
            }
        }
        cursors[row] = cursor;
        Ok(found)
    }

    /// Upper storage is symmetric by shape: no mirror is found, and a nonzero below the diagonal is rejected.
    fn check_upper(&self) -> Result<(), InputError> {
        // Row `j`'s edges to rows above it are stored in those rows, so they are marked ahead.
        let mut coupled_above = vec![false; self.n()];
        for row in 0..self.n() {
            let (mut diagonal, mut coupled) = (T::zero(), false);
            let span = self.span(row);
            for (at, &col) in span.clone().zip(&self.col_indices[span]) {
                let col = index(col);
                let value = value_in_range("values", &self.values, at)?;
                if value == T::zero() {
                    continue;
                }
                match col.cmp(&row) {
                    Ordering::Less => return Err(InputError::OutsideTriangle { row }),
                    Ordering::Equal => diagonal = value,
                    Ordering::Greater => (coupled, coupled_above[col]) = (true, true),
                }
            }
            check_diagonal(diagonal, || coupled || coupled_above[row], row)?;
        }
        Ok(())
    }
}

/// Positive and at most `MAX/4`, or zero on a row with no nonzero edge.
fn check_diagonal<T: Real>(
    diagonal: T,
    coupled: impl FnOnce() -> bool,
    row: usize,
) -> Result<(), InputError> {
    let admissible = match diagonal {
        d if d > T::zero() => diagonal_in_range(d),
        d if d == T::zero() => !coupled(),
        _ => false,
    };
    match admissible {
        true => Ok(()),
        false => Err(InputError::Diagonal { row }),
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

/// Reads no value, sparing a stream: the checks after it read each. Accumulated, not short-circuited: the `all` form measured 0.4-2% slower.
fn is_canonical<I: PrimInt>(row_ptrs: &[I], col_indices: &[I]) -> Result<bool, InputError> {
    // `None` means `I` cannot represent `n`, so every `I` value is below it.
    let limit = I::from(row_ptrs.len() - 1);
    let mut canonical = true;
    let outside = |j: I| j < I::zero() || limit.is_some_and(|limit| j >= limit);
    for bounds in row_ptrs.windows(2) {
        let start = index(bounds[0]);
        let columns = &col_indices[start..index(bounds[1])];
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

/// Sorts and coalesces each row into a copy in the input's own index type; each value, and each repeated column's sum, is checked at the caller's position.
fn rewrite<'a, T: Real, I: PrimInt>(
    csr: &CsrArrays<'_, T, I>,
) -> Result<Canonical<'a, T, I>, InputError> {
    let nnz = csr.col_indices.len();
    let mut row_ptrs = Vec::with_capacity(csr.row_ptrs.len());
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(I, T, usize)> = Vec::new();
    row_ptrs.push(I::zero());
    for bounds in csr.row_ptrs.windows(2) {
        entries.clear();
        for at in index(bounds[0])..index(bounds[1]) {
            let value = value_in_range("values", csr.values, at)?;
            entries.push((csr.col_indices[at], value, at));
        }
        // One row's degree, not nnz. Stable, so duplicates sum in stored order.
        entries.sort_by_key(|&(col, _, _)| col);
        for group in entries.chunk_by(|left, right| left.0 == right.0) {
            let (col, first, at) = group[0];
            let folded = group[1..].iter().fold(first, |sum, &(_, v, _)| sum + v);
            // A single entry was checked above; only a sum can leave the range here.
            if group.len() > 1 && !in_range(folded) {
                return Err(InputError::OutOfRange {
                    parameter: "values",
                    entry: at,
                });
            }
            col_indices.push(col);
            values.push(folded);
        }
        // At most the caller's own `nnz`, which its last row pointer already holds in `I`.
        row_ptrs.push(I::from(col_indices.len()).expect("a coalesced count fits the input's type"));
    }
    Ok(Canonical {
        row_ptrs: Cow::Owned(row_ptrs),
        col_indices: Cow::Owned(col_indices),
        values: Cow::Owned(values),
    })
}
