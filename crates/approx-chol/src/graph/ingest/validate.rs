use super::canonical::Canonical;
use super::sets::DisjointSets;
use crate::graph::BlockLayout;
use crate::types::{count_as_scalar, Real};
use crate::{CsrError, Error};

/// One forward-only cursor per row. The walk is a merge-join only because every entry
/// is claimed at most once, which is what [`Canonical`] guarantees.
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

    /// Stored zeros count as absent.
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
            if !value.is_finite() {
                return Err(Error::NonFiniteValue {
                    position: cursor as usize,
                });
            }
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

/// Everything ingestion learns from the CSR before any graph exists.
pub(super) struct Ingested<T> {
    /// One per real vertex, and the ground vertex's last when grounded.
    pub(super) diagonal: Vec<T>,
    pub(super) grounding: Grounding<T>,
    /// Blocks implied by the connectivity the validating walk unioned as it went, rather
    /// than by a pass of its own: every edge is already being visited there.
    pub(super) layout: Option<BlockLayout>,
}

/// A matrix whose rows all balance is a bare Laplacian on each of its components, and
/// has no ground vertex to attach.
pub(super) enum Grounding<T> {
    Floating,
    Grounded {
        /// Indexed by row, zero where the row balances — the row-sum accumulator reused
        /// in place rather than a second array.
        surpluses: Vec<T>,
        /// How many of those are positive, which is the ground vertex's degree.
        degree: usize,
    },
}

/// One pass per row, claiming each upper-triangle entry's mirror. Every stored entry is
/// read exactly once between the claims and the loop below, which is what lets
/// [`Canonical::of`] leave finiteness to this walk. Accumulates what the
/// augmentation decision needs and builds nothing: which arm each block gets is not
/// known until its dimension is, and that waits on the layout this pass feeds.
pub(super) fn validate<T: Real>(canonical: &Canonical<'_, T>) -> Result<Ingested<T>, Error> {
    let (row_ptrs, col_indices, values) = canonical.arrays();
    let n = row_ptrs.len() - 1;
    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);

    let mut sets = DisjointSets::new(n);
    let mut diagonal = vec![T::zero(); n];
    // Off-diagonal contributions only; the diagonal joins in `ground`, which is
    // not known for a row until that row's own claim below.
    let mut row_sums = vec![T::zero(); n];

    for row in 0..n {
        let row_end = row_ptrs[row + 1];
        // The diagonal is claimed like any mirror, which both yields its value and
        // advances the cursor past everything below it — by now only the zeros that
        // contribute no edge. Claiming every diagonal up front instead would skip
        // the mirrors that live below them.
        diagonal[row] = mirrors.claim(row, row)?;
        let mut cursor = mirrors.cursors[row];
        let mut root = sets.find(row as u32);

        while cursor < row_end {
            let col = col_indices[cursor as usize] as usize;
            let upper = values[cursor as usize];
            if !upper.is_finite() {
                return Err(Error::NonFiniteValue {
                    position: cursor as usize,
                });
            }
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
            // Each row sums the value it stores, not the one the graph symmetrizes to:
            // charging `upper` to both would make the tolerated mirror difference look
            // like `col`'s own surplus and ground it.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + lower;
            root = sets.union_resolved(root, col as u32);
        }
    }
    ground(diagonal, row_sums, canonical.terms(), sets)
}

/// How far one row's diagonal exceeds its off-diagonal mass, judged against the noise
/// the row's own scale and term count can carry.
enum RowBalance<T> {
    NonFinite,
    Deficit,
    Negligible,
    /// Worth closing with a ground edge.
    Surplus(T),
}

impl<T: Real> RowBalance<T> {
    /// `terms` is how many additions produced `excess`, not the row's degree.
    fn of(diagonal: T, off_diagonal_sum: T, terms: u32) -> Self {
        let excess = diagonal + off_diagonal_sum;
        // Every off-diagonal folded into the sum was negative, so the row's magnitude
        // sum is `|d| + d - excess` and needs no second accumulator. Subtracting before
        // the second add keeps a diagonal above `MAX / 2` from overflowing to infinity.
        let scale = (diagonal.abs() - excess) + diagonal;
        // A non-finite sum forces a non-finite scale, so scale alone decides. Every
        // comparison below succeeds on an infinite deficit (`-inf < -inf`).
        if !scale.is_finite() {
            return Self::NonFinite;
        }
        // The most this row's own additions could have invented, and so the only
        // departure from zero-sum the row cannot account for. One floor for both signs:
        // a departure this clears is real evidence whichever way it points, and
        // forgiving more in one direction than the other grounds a row for a drift that
        // would be dismissed as noise with its sign flipped.
        let accumulated = T::epsilon() * scale * count_as_scalar::<T, _>(terms);
        if excess < -accumulated {
            return Self::Deficit;
        }
        if excess <= accumulated {
            return Self::Negligible;
        }
        Self::Surplus(excess)
    }
}

/// `row_sums` arrives holding each row's off-diagonal total; the diagonal joins it
/// here, the first point at which every row's is known.
fn ground<T: Real>(
    mut diagonal: Vec<T>,
    mut row_sums: Vec<T>,
    terms: impl Iterator<Item = u32>,
    mut sets: DisjointSets,
) -> Result<Ingested<T>, Error> {
    let mut total = T::zero();
    let mut degree = 0usize;
    for (row, ((sum, &d), count)) in row_sums
        .iter_mut()
        .zip(diagonal.iter())
        .zip(terms)
        .enumerate()
    {
        *sum = match RowBalance::of(d, *sum, count) {
            RowBalance::NonFinite => return Err(Error::NonFiniteRow { row }),
            RowBalance::Deficit => return Err(Error::NotDiagonallyDominant { row }),
            RowBalance::Negligible => T::zero(),
            RowBalance::Surplus(excess) => {
                total = total + excess;
                degree += 1;
                excess
            }
        };
    }

    let m = diagonal.len();
    if degree == 0 {
        return Ok(Ingested {
            diagonal,
            grounding: Grounding::Floating,
            layout: sets.layout(),
        });
    }
    if m >= u32::MAX as usize {
        return Err(Error::InvalidCsr(
            CsrError::MatrixDimensionExceedsIndexType {
                n: m.saturating_add(1),
            },
        ));
    }
    diagonal.push(total);
    // The ground vertex is absent from the CSR, so the rows it closes are unioned
    // through it here: it links every component holding a surplus into one block, and
    // connectivity read from the CSR alone would hand back each of them separately.
    let vertex = sets.push();
    let mut root = vertex;
    for (row, &surplus) in row_sums.iter().enumerate() {
        if surplus > T::zero() {
            root = sets.union_resolved(root, row as u32);
        }
    }
    Ok(Ingested {
        diagonal,
        grounding: Grounding::Grounded {
            surpluses: row_sums,
            degree,
        },
        layout: sets.layout(),
    })
}
