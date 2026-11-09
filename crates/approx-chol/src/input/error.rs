//! Why an input was rejected, in the caller's terms, and the checks that say so.

use num_traits::PrimInt;

use crate::Real;

/// Why an input was rejected: the defect, and where it sits in the caller's terms. A `parameter` is the constructor's own name for the argument, an `entry` a position in it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// A slice's length differs from the one the input fixed: `n`, another slice's, or the last row pointer.
    LengthMismatch {
        /// The slice.
        parameter: &'static str,
        /// The length the input fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// `row_ptrs[row]` is out of order: the first is not 0, or it exceeds the next.
    RowPtrs {
        /// The offending pointer's position.
        row: usize,
    },
    /// An index names a vertex, row or column that does not exist, or does not fit the target index type.
    IndexOutOfRange {
        /// The slice.
        parameter: &'static str,
        /// The offending entry.
        entry: usize,
    },
    /// A stated value is not finite.
    OutOfRange {
        /// The slice.
        parameter: &'static str,
        /// The offending entry.
        entry: usize,
    },
    /// A stated value that must be `≥ 0` is negative.
    Negative {
        /// The slice.
        parameter: &'static str,
        /// The offending entry.
        entry: usize,
    },
    /// A Laplacian's edge joins a vertex to itself.
    SelfLoop {
        /// The offending edge.
        edge: usize,
    },
    /// A width-2 column names one row at both ends.
    RepeatedRow {
        /// The offending column.
        column: usize,
    },
    /// Every surplus is zero: the matrix is a [`Laplacian`](super::Laplacian).
    ZeroSurplus,
    /// An entry has no mirror within 8 ulps.
    NotSymmetric {
        /// The offending row.
        row: usize,
    },
    /// An upper triangle stores a nonzero entry, or a repeated column's nonzero sum, below its diagonal.
    OutsideTriangle {
        /// The offending row.
        row: usize,
    },
    /// A CSR entry or a repeated column's sum is not finite, or nonzero below `MIN_POSITIVE/EPSILON`.
    InadmissibleRow {
        /// The offending row.
        row: usize,
    },
    /// A converted matrix is not square.
    NotSquare {
        /// Its row count.
        rows: usize,
        /// Its column count.
        cols: usize,
    },
    /// A converted matrix's row leaves unused slots after its entries.
    RowGap {
        /// The first row with unused slots.
        row: usize,
    },
}

impl core::fmt::Display for InputError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                parameter,
                expected,
                found,
            } => write!(f, "{parameter} has length {found}, expected {expected}"),
            Self::RowPtrs { row } => write!(
                f,
                "row_ptrs[{row}] is out of order: the first must be 0, each at most the next"
            ),
            Self::IndexOutOfRange { parameter, entry } => {
                write!(f, "{parameter}[{entry}] is out of range")
            }
            Self::OutOfRange { parameter, entry } => {
                write!(f, "{parameter}[{entry}] is not finite")
            }
            Self::Negative { parameter, entry } => write!(f, "{parameter}[{entry}] is negative"),
            Self::SelfLoop { edge } => write!(f, "edge {edge} joins a vertex to itself"),
            Self::RepeatedRow { column } => write!(f, "column {column} names one row twice"),
            Self::ZeroSurplus => write!(f, "every surplus is zero, so the matrix is a Laplacian"),
            Self::NotSymmetric { row } => write!(f, "row {row} has an entry with no equal mirror"),
            Self::OutsideTriangle { row } => write!(
                f,
                "row {row} of an upper triangle has a nonzero below the diagonal"
            ),
            Self::InadmissibleRow { row } => {
                write!(f, "row {row} has an entry outside the representable range")
            }
            Self::NotSquare { rows, cols } => write!(f, "matrix is not square ({rows}x{cols})"),
            Self::RowGap { row } => write!(f, "row {row} leaves unused slots after its entries"),
        }
    }
}

impl std::error::Error for InputError {}

/// `parameter` has `found` entries where the input fixed `expected`.
pub(super) fn length(
    parameter: &'static str,
    expected: usize,
    found: usize,
) -> Result<(), InputError> {
    match found == expected {
        true => Ok(()),
        false => Err(InputError::LengthMismatch {
            parameter,
            expected,
            found,
        }),
    }
}

pub(super) fn finite<T: Real>(parameter: &'static str, values: &[T]) -> Result<(), InputError> {
    match values.iter().position(|x| !x.is_finite()) {
        Some(entry) => Err(InputError::OutOfRange { parameter, entry }),
        None => Ok(()),
    }
}

/// The first entry not finite and `≥ 0`, by which of the two it fails.
pub(super) fn nonnegative<T: Real>(
    parameter: &'static str,
    values: &[T],
) -> Result<(), InputError> {
    match values
        .iter()
        .position(|&x| !(x.is_finite() && x >= T::zero()))
    {
        Some(entry) if values[entry].is_finite() => Err(InputError::Negative { parameter, entry }),
        Some(entry) => Err(InputError::OutOfRange { parameter, entry }),
        None => Ok(()),
    }
}

/// Every index in `parameter` below `n`.
pub(super) fn indices<I: PrimInt>(
    parameter: &'static str,
    n: usize,
    slice: &[I],
) -> Result<(), InputError> {
    match slice.iter().position(|&index| !below(n, index)) {
        Some(entry) => Err(InputError::IndexOutOfRange { parameter, entry }),
        None => Ok(()),
    }
}

fn below<I: PrimInt>(n: usize, row: I) -> bool {
    row.to_usize().is_some_and(|row| row < n)
}

/// One `b` per `a`, each column's two ends distinct rows below `n`; `repeated` names a column with one row twice.
pub(super) fn check_ends<I: PrimInt>(
    n: usize,
    (a_name, a): (&'static str, &[I]),
    (b_name, b): (&'static str, &[I]),
    repeated: fn(usize) -> InputError,
) -> Result<(), InputError> {
    length(b_name, a.len(), b.len())?;
    for (entry, (end_a, end_b)) in a.iter().zip(b).enumerate() {
        if !below(n, *end_a) {
            return Err(InputError::IndexOutOfRange {
                parameter: a_name,
                entry,
            });
        }
        if !below(n, *end_b) {
            return Err(InputError::IndexOutOfRange {
                parameter: b_name,
                entry,
            });
        }
        if end_a == end_b {
            return Err(repeated(entry));
        }
    }
    Ok(())
}
