//! CSR to elimination graph: canonicalize, pair each off-diagonal with its
//! mirror, and close the row deficits with a Gremban ground vertex.

use super::{add_edge_pair, count_components, AdjListGraph, Edge, EdgeCount, GraphBuild};
use crate::types::count_as_scalar;
use crate::{CsrError, CsrRef, Error, Real};

pub(super) fn from_sddm<T: Real, C: EdgeCount>(
    csr: CsrRef<'_, T, u32>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    // A rewrite has to outlive the walk that borrows it, so it lives in this frame
    // rather than inside `Mirrors`.
    let rewritten = canonicalize(csr)?;
    match &rewritten {
        Some(r) => parse(Mirrors::new(&r.row_ptrs, &r.col_indices, &r.values)),
        None => parse(Mirrors::new(
            csr.row_ptrs(),
            csr.col_indices(),
            csr.values(),
        )),
    }
}

/// Canonical arrays rebuilt from non-canonical input.
struct Rewritten<T> {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<T>,
}

/// `None` when the caller's columns already ascend strictly within every row, the
/// shape scipy produces and the only one within emits.
fn canonicalize<T: Real>(csr: CsrRef<'_, T, u32>) -> Result<Option<Rewritten<T>>, Error> {
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
        return Ok(None);
    }
    Ok(Some(rewrite(csr)))
}

/// Sort each row and sum duplicate entries as scipy's `sum_duplicates` does. Only
/// non-canonical input pays this copy.
fn rewrite<T: Real>(csr: CsrRef<'_, T, u32>) -> Rewritten<T> {
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
    Rewritten {
        row_ptrs,
        col_indices,
        values,
    }
}

/// Canonical CSR — columns ascend strictly within every row, which also rules out
/// duplicates — plus one cursor per row, each advancing only forward. Every entry
/// is claimed at most once across the whole walk, which is a merge-join only under
/// that guarantee; hence [`canonicalize`] rather than raw arrays.
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
    mut mirrors: Mirrors<'_, T>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    let (row_ptrs, col_indices, values) = (mirrors.row_ptrs, mirrors.col_indices, mirrors.values);
    let n = row_ptrs.len() - 1;
    let mut adj: Vec<Vec<Edge<T, C>>> = (0..n)
        .map(|row| Vec::with_capacity((row_ptrs[row + 1] - row_ptrs[row]) as usize))
        .collect();

    let mut diag = vec![T::zero(); n];
    // Off-diagonal contributions only; `augment` folds in the diagonal, which is
    // not known for a row until that row's own claim below.
    let mut row_sums = vec![T::zero(); n];

    for row in 0..n {
        let row_end = row_ptrs[row + 1];
        // The diagonal is claimed like any mirror, which both yields its value and
        // advances the cursor past everything below it — by now only the zeros that
        // contribute no edge. Claiming every diagonal up front instead would skip
        // the mirrors that live below them.
        diag[row] = mirrors.claim(row, row)?;
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
///
/// `row_sums` arrives holding each row's off-diagonal total; the diagonal joins it
/// here, the first point at which every row's is known.
fn augment<T: Real, C: EdgeCount>(
    mut adj: Vec<Vec<Edge<T, C>>>,
    mut diag: Vec<T>,
    mut row_sums: Vec<T>,
) -> Result<GraphBuild<AdjListGraph<C, T>, T>, Error> {
    // Relative floor below which a surplus is not worth a ground edge, scaled by
    // each row's magnitude below.
    let tolerance = T::by_precision(1e-6, 1e-10);
    let mut surplus_sum = T::zero();
    let mut grounded = 0usize;
    for (row, (sum, &d)) in row_sums.iter_mut().zip(diag.iter()).enumerate() {
        *sum = d + *sum;
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
        // Grounding is worth it only above both floors. The policy floor is
        // relative, capped absolutely so a 1e12-scale row's real surplus is not
        // swallowed; the noise floor is the error this row's own sum could have
        // accumulated over its `terms` additions, without which a high-degree row
        // at scale grounds on rounding alone at a weight orders above the truth.
        // Both ends are pinned by the `*_scale_*` and `*_noise_floor_*` tests.
        let policy = row_tolerance.min(T::epsilon().sqrt());
        let terms = count_as_scalar::<T, _>(adj[row].len() + 1);
        let noise = T::epsilon() * scale * terms;
        if *sum < T::zero() || *sum <= policy.max(noise) {
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
