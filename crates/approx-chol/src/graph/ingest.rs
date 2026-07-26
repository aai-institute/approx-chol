//! CSR to elimination graph: canonicalize, pair each off-diagonal with its
//! mirror, and close the row deficits with a Gremban ground vertex.

use super::{add_edge_pair, count_components, AdjListGraph, Edge, EdgeCount, GraphBuild};
use crate::{CsrError, CsrRef, Error, Real};

pub(super) fn from_sddm<T: Real, C: EdgeCount>(
    csr: CsrRef<'_, T, u32>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    parse(&Canonical::from_csr(csr)?)
}

/// CSR whose columns ascend strictly within every row, which also rules out
/// duplicates. [`Mirrors`] is a merge-join only under that guarantee, so [`parse`]
/// takes this rather than raw arrays.
enum Canonical<'a, T> {
    Borrowed(CsrRef<'a, T, u32>),
    Rewritten {
        row_ptrs: Vec<u32>,
        col_indices: Vec<u32>,
        values: Vec<T>,
    },
}

impl<'a, T: Real> Canonical<'a, T> {
    fn from_csr(csr: CsrRef<'a, T, u32>) -> Result<Self, Error> {
        // Rows tile the value array in order, so one pass answers both questions:
        // non-finite wins over non-canonical either way, and the reported position
        // stays in the caller's numbering rather than a rewritten copy's.
        let mut canonical = true;
        let mut row_start = 0;
        for (cols, vals) in csr.rows() {
            if let Some(offset) = vals.iter().position(|value| !value.is_finite()) {
                return Err(Error::NonFiniteValue {
                    position: row_start + offset,
                });
            }
            canonical = canonical && cols.windows(2).all(|pair| pair[0] < pair[1]);
            row_start += vals.len();
        }
        if canonical {
            return Ok(Self::Borrowed(csr));
        }
        Ok(Self::rewrite(csr))
    }

    /// Sum duplicate entries as scipy's `sum_duplicates` does. Only non-canonical
    /// input pays this copy; the shape scipy produces and the only one within
    /// emits takes the borrowed arm.
    fn rewrite(csr: CsrRef<'a, T, u32>) -> Self {
        let nnz = csr.col_indices().len();
        let mut row_ptrs = Vec::with_capacity(csr.n() + 1);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut entries: Vec<(u32, T)> = Vec::new();
        row_ptrs.push(0u32);
        for (cols, vals) in csr.rows() {
            entries.clear();
            entries.extend(cols.iter().copied().zip(vals.iter().copied()));
            // One row's degree, not nnz. Stable, so duplicates sum in stored order.
            entries.sort_by_key(|&(col, _)| col);
            for group in entries.chunk_by(|left, right| left.0 == right.0) {
                col_indices.push(group[0].0);
                values.push(group[1..].iter().fold(group[0].1, |sum, &(_, v)| sum + v));
            }
            row_ptrs.push(col_indices.len() as u32);
        }
        Self::Rewritten {
            row_ptrs,
            col_indices,
            values,
        }
    }

    fn parts(&self) -> (&[u32], &[u32], &[T]) {
        match self {
            Self::Borrowed(csr) => (csr.row_ptrs(), csr.col_indices(), csr.values()),
            Self::Rewritten {
                row_ptrs,
                col_indices,
                values,
            } => (row_ptrs, col_indices, values),
        }
    }
}

/// Canonical arrays plus one cursor per row, each advancing only forward. Every
/// entry is claimed at most once across the whole walk, which is what makes
/// mirror pairing a merge-join instead of a per-edge search.
struct Mirrors<'a, T> {
    row_ptrs: &'a [u32],
    col_indices: &'a [u32],
    values: &'a [T],
    cursors: Vec<u32>,
}

impl<'a, T: Real> Mirrors<'a, T> {
    fn new(row_ptrs: &'a [u32], col_indices: &'a [u32], values: &'a [T]) -> Self {
        let cursors = row_ptrs[..row_ptrs.len() - 1].to_vec();
        Self {
            row_ptrs,
            col_indices,
            values,
            cursors,
        }
    }

    /// Consume `row`'s entry at `col`, treating stored zeros as absent and
    /// returning zero when it is missing.
    fn claim(&mut self, row: usize, col: usize) -> Result<T, Error> {
        let row_end = self.row_ptrs[row + 1];
        let mut cursor = self.cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = self.col_indices[cursor as usize] as usize;
            if at > col {
                break;
            }
            let value = self.values[cursor as usize];
            cursor += 1;
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                self.cursors[row] = cursor;
                return Err(Error::Asymmetric { edge: (at, row) });
            }
        }
        self.cursors[row] = cursor;
        Ok(found)
    }
}

fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

/// Walk each row once, claiming each upper-triangle entry's mirror through a
/// monotone per-row cursor.
fn parse<T: Real, C: EdgeCount>(
    canonical: &Canonical<T>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    let (row_ptrs, col_indices, values) = canonical.parts();
    let n = row_ptrs.len() - 1;
    let mut adj: Vec<Vec<Edge<T, C>>> = (0..n)
        .map(|row| Vec::with_capacity((row_ptrs[row + 1] - row_ptrs[row]) as usize))
        .collect();

    // Read the diagonal by search rather than a full pass, so the off-diagonal
    // accumulation below can start from it.
    let mut diag = vec![T::zero(); n];
    for (row, diagonal) in diag.iter_mut().enumerate() {
        let start = row_ptrs[row] as usize;
        let cols = &col_indices[start..row_ptrs[row + 1] as usize];
        let offset = cols.partition_point(|&col| (col as usize) < row);
        if cols.get(offset) == Some(&(row as u32)) {
            *diagonal = values[start + offset];
        }
    }
    let mut row_sums = diag.clone();

    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);
    for row in 0..n {
        let row_end = row_ptrs[row + 1];
        // The diagonal is claimed like any mirror: it advances the cursor past
        // everything below it, which by now should be only the zeros that
        // contribute no edge. Its value was already read above.
        mirrors.claim(row, row)?;
        let mut cursor = mirrors.cursors[row];

        while cursor < row_end {
            let col = col_indices[cursor as usize] as usize;
            let upper = values[cursor as usize];
            cursor += 1;
            // Duplicates can coalesce to exactly zero, which contributes no edge.
            if upper == T::zero() {
                continue;
            }
            let lower = mirrors.claim(col, row)?;
            if !approximately_equal(upper, lower) {
                return Err(Error::Asymmetric { edge: (row, col) });
            }
            if upper > T::zero() {
                return Err(Error::PositiveOffDiagonal { edge: (row, col) });
            }
            // Zero was skipped and positive rejected, so `upper` is negative here
            // and `-upper` is the edge weight. A NaN coalesced from opposing
            // infinities never reaches this line — it fails the symmetry check.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + upper;
            add_edge_pair(&mut adj, row, col, -upper);
        }
    }
    augment(adj, diag, row_sums)
}

/// Clamp each row's surplus to non-negative, then close the remaining deficits
/// with a Gremban ground vertex and reject disconnected input.
fn augment<T: Real, C: EdgeCount>(
    mut adj: Vec<Vec<Edge<T, C>>>,
    mut diag: Vec<T>,
    mut row_sums: Vec<T>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    // Absolute surplus floor separating genuine diagonal dominance from a
    // Laplacian's rounding noise; scaled by each row's magnitude below.
    let tolerance = T::by_precision(1e-6, 1e-10);
    let mut surplus_sum = T::zero();
    let mut grounded = 0usize;
    for (row, (sum, &d)) in row_sums.iter_mut().zip(diag.iter()).enumerate() {
        // Every off-diagonal folded into the sum was negative, so the row's
        // magnitude sum is `|d| + d - sum` and needs no second accumulator.
        let scale = d.abs() + d - *sum;
        // Every check below succeeds on an infinite deficit (`-inf < -inf`).
        if !sum.is_finite() || !scale.is_finite() {
            return Err(Error::NonFiniteRow { row });
        }
        let row_tolerance = tolerance * scale;
        if *sum < -row_tolerance {
            return Err(Error::NotDiagonallyDominant { row });
        }
        // The cap stops the relative arm swallowing real dominance at large
        // scale; both ends are pinned by `*_scale_*` ingestion tests.
        if *sum < T::zero() || sum.abs() <= row_tolerance.min(T::epsilon().sqrt()) {
            *sum = T::zero();
        } else {
            surplus_sum = surplus_sum + *sum;
            grounded += 1;
        }
    }

    let m = adj.len();
    if grounded > 0 {
        if m >= u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType {
                    n: m.saturating_add(1),
                },
            ));
        }
        adj.push(Vec::with_capacity(grounded));
        diag.push(surplus_sum);
        for (row, &surplus) in row_sums.iter().enumerate() {
            // The clamp above left every surplus non-negative.
            if surplus > T::zero() {
                add_edge_pair(&mut adj, row, m, surplus);
            }
        }
    }

    // Reject before the expensive elimination: >1 component (over the real
    // vertices) means a block can't reach ground. See `Error::Disconnected`.
    let components = count_components(&adj, m);
    if components > 1 {
        return Err(Error::Disconnected { components });
    }

    Ok(GraphBuild {
        graph: AdjListGraph::from_adjacency(adj),
        diagonal: diag,
    })
}
