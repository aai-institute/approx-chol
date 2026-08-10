use crate::types::Real;
use crate::{CsrRef, Error};

/// The input read as ingestion needs it: strictly ascending columns within every row.
/// The caller's arrays already are that whenever scipy produced them, which is the only
/// shape within emits, so the rewritten copy is what rare input costs rather than what
/// every input pays.
pub(super) struct Canonical<'a, T> {
    input: CsrRef<'a, T, u32>,
    /// `None` when the caller's arrays are already canonical.
    rewritten: Option<Rewritten<T>>,
}

impl<'a, T: Real> Canonical<'a, T> {
    /// Canonical input never reads a value here: [`validate`] visits every stored entry
    /// exactly once, so checking each as it is read spares ingestion a whole stream over
    /// the widest of the three arrays.
    pub(super) fn of(csr: CsrRef<'a, T, u32>) -> Result<Self, Error> {
        if is_canonical(csr.row_ptrs(), csr.col_indices()) {
            return Ok(Self {
                input: csr,
                rewritten: None,
            });
        }
        // Rows tile the value array in order, so this position is the caller's own.
        // Scanning before rewriting is what keeps it so, and is why non-finite is
        // reported in preference to the non-canonical shape.
        if let Some(position) = csr.values().iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { position });
        }
        Ok(Self {
            input: csr,
            rewritten: Some(rewrite(csr)?),
        })
    }

    pub(super) fn arrays(&self) -> (&[u32], &[u32], &[T]) {
        match &self.rewritten {
            Some(r) => (&r.row_ptrs, &r.col_indices, &r.values),
            None => (
                self.input.row_ptrs(),
                self.input.col_indices(),
                self.input.values(),
            ),
        }
    }

    /// Additions charged to a row's excess, counted on the caller's arrays before
    /// coalescing: [`rewrite`] folds each duplicate group with its own additions, and
    /// those land in the row's sum too. For canonical input this is the diagonal plus
    /// the row's degree.
    pub(super) fn terms(&self) -> impl Iterator<Item = u32> + '_ {
        self.input
            .row_ptrs()
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
    }
}

/// Accumulated rather than short-circuited: canonical input is the overwhelming case and
/// runs every entry regardless, so the early exit only adds a branch per entry to the
/// path that always takes it. Measured worth 0.4-2% of the build against the `all` form.
fn is_canonical(row_ptrs: &[u32], col_indices: &[u32]) -> bool {
    let mut canonical = true;
    for bounds in row_ptrs.windows(2) {
        let row = &col_indices[bounds[0] as usize..bounds[1] as usize];
        for pair in row.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    canonical
}

/// Canonical arrays rebuilt from non-canonical input.
struct Rewritten<T> {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<T>,
}

/// Only non-canonical input pays this copy.
fn rewrite<T: Real>(csr: CsrRef<'_, T, u32>) -> Result<Rewritten<T>, Error> {
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
            // Every input was finite, so only this fold can overflow. Caught here so the
            // arrays handed on are all-finite, which is what lets the reads downstream
            // report a position in the caller's numbering.
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
