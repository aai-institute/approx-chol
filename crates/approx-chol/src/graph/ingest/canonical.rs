use crate::types::Real;
use crate::{CsrRef, Error};
use num_traits::PrimInt;

/// A row pointer of a validated [`CsrRef`]: in `0..=nnz`, so in `usize` whatever `J` is.
#[inline(always)]
pub(super) fn row_ptr<J: PrimInt>(ptr: J) -> usize {
    ptr.to_usize().expect("a validated row pointer is a usize")
}

/// Each row's addition count, from the caller's own pointers: [`rewrite`]'s coalescing
/// additions land in the row sum too.
pub(super) fn terms<J: PrimInt>(row_ptrs: &[J]) -> impl Iterator<Item = u32> + '_ {
    row_ptrs
        .windows(2)
        .map(|bounds| (row_ptr(bounds[1]) - row_ptr(bounds[0])) as u32)
}

/// Strictly ascending columns per row, which scipy already emits, so only rare input
/// pays for a rewritten copy. Accumulated, not short-circuited: measured 0.4-2% of the
/// build over the `all` form.
pub(super) fn is_canonical<J: PrimInt>(row_ptrs: &[J], col_indices: &[J]) -> bool {
    let mut canonical = true;
    for bounds in row_ptrs.windows(2) {
        let row = &col_indices[row_ptr(bounds[0])..row_ptr(bounds[1])];
        for pair in row.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    canonical
}

/// Canonical arrays rebuilt from non-canonical input.
pub(super) struct Rewritten<T> {
    pub(super) row_ptrs: Vec<u32>,
    pub(super) col_indices: Vec<u32>,
    pub(super) values: Vec<T>,
}

/// Only non-canonical input pays this copy.
pub(super) fn rewrite<T: Real>(csr: CsrRef<'_, T, u32>) -> Result<Rewritten<T>, Error> {
    let nnz = csr.col_indices().len();
    let mut row_ptrs = Vec::with_capacity(csr.n() + 1);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(u32, T)> = Vec::new();
    row_ptrs.push(0u32);
    for (row, (cols, vals)) in csr.rows().enumerate() {
        entries.clear();
        entries.extend(cols.iter().copied().zip(vals.iter().copied()));
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
        row_ptrs.push(col_indices.len() as u32);
    }
    Ok(Rewritten {
        row_ptrs,
        col_indices,
        values,
    })
}
