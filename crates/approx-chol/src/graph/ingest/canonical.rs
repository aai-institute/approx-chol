use super::index;
use crate::types::Real;
use crate::{CsrError, CsrRef, Error, IndexKind};
use num_traits::PrimInt;

/// Strictly ascending columns per row; scipy already emits them, so only rare input pays for a copy.
pub(super) struct Canonical<'a, T, I> {
    input: CsrRef<'a, T, I>,
    /// `None` when the caller's arrays are already canonical.
    rewritten: Option<Rewritten<T, I>>,
}

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    /// Reads no value on the canonical path, sparing a stream: `validate` checks each as it reads it.
    pub(super) fn of(csr: CsrRef<'a, T, I>) -> Result<Self, Error> {
        // Every position downstream, mirror cursors included, is a `u32`.
        if u32::try_from(csr.col_indices().len()).is_err() {
            return Err(Error::InvalidCsr(CsrError::IndexExceedsIndexType {
                kind: IndexKind::RowPtr,
            }));
        }
        if is_canonical(csr.row_ptrs(), csr.col_indices()) {
            return Ok(Self {
                input: csr,
                rewritten: None,
            });
        }
        // Before rewriting, so the position stays the caller's own.
        if let Some(position) = csr.values().iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { position });
        }
        Ok(Self {
            input: csr,
            rewritten: Some(rewrite(csr)?),
        })
    }

    pub(super) fn arrays(&self) -> (&[I], &[I], &[T]) {
        match &self.rewritten {
            Some(r) => (&r.row_ptrs, &r.col_indices, &r.values),
            None => (
                self.input.row_ptrs(),
                self.input.col_indices(),
                self.input.values(),
            ),
        }
    }

    /// Counted before coalescing: [`rewrite`]'s own additions land in the row sum too.
    pub(super) fn terms(&self) -> impl Iterator<Item = u32> + '_ {
        self.input
            .row_ptrs()
            .windows(2)
            .map(|bounds| (index(bounds[1]) - index(bounds[0])) as u32)
    }
}

/// Accumulated, not short-circuited: measured 0.4-2% of the build over the `all` form.
fn is_canonical<I: PrimInt>(row_ptrs: &[I], col_indices: &[I]) -> bool {
    let mut canonical = true;
    for bounds in row_ptrs.windows(2) {
        let row = &col_indices[index(bounds[0])..index(bounds[1])];
        for pair in row.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    canonical
}

/// Canonical arrays rebuilt from non-canonical input, in its own index type.
struct Rewritten<T, I> {
    row_ptrs: Vec<I>,
    col_indices: Vec<I>,
    values: Vec<T>,
}

/// Only non-canonical input pays this copy.
fn rewrite<T: Real, I: PrimInt>(csr: CsrRef<'_, T, I>) -> Result<Rewritten<T, I>, Error> {
    let nnz = csr.col_indices().len();
    let mut row_ptrs = Vec::with_capacity(csr.n() + 1);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(I, T)> = Vec::new();
    row_ptrs.push(I::zero());
    for (row, bounds) in csr.row_ptrs().windows(2).enumerate() {
        let (from, to) = (index(bounds[0]), index(bounds[1]));
        entries.clear();
        entries.extend(
            csr.col_indices()[from..to]
                .iter()
                .copied()
                .zip(csr.values()[from..to].iter().copied()),
        );
        // One row's degree, not nnz. Stable, so duplicates sum in stored order.
        entries.sort_by_key(|&(col, _)| col);
        for group in entries.chunk_by(|left, right| left.0 == right.0) {
            let folded = group[1..].iter().fold(group[0].1, |sum, &(_, v)| sum + v);
            // Only this fold can overflow; caught here so downstream stays all-finite.
            if !folded.is_finite() {
                return Err(Error::NonFiniteRow { row });
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
