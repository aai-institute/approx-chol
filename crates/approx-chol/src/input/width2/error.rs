//! Why a view rejected its slices, and the checks that say so.

use num_traits::PrimInt;

use crate::Real;

/// Why a view's `new` rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the column count.
    LengthMismatch {
        /// The column count the first slice fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A column names a row its view does not have.
    IndexOutOfRange {
        /// The offending column of `V`.
        column: usize,
    },
    /// A column names one vertex at both ends.
    RepeatedVertex {
        /// The offending column of `V`.
        column: usize,
    },
    /// A weight is negative or not finite.
    InvalidWeight {
        /// The offending column of `V`.
        column: usize,
    },
    /// A ground is negative or not finite.
    InvalidGround {
        /// The offending vertex.
        vertex: usize,
    },
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "view slice has length {found}, expected {expected}")
            }
            Self::IndexOutOfRange { column } => write!(f, "column {column} names a missing row"),
            Self::RepeatedVertex { column } => write!(f, "column {column} names one vertex twice"),
            Self::InvalidWeight { column } => {
                write!(f, "column {column} has a negative or non-finite weight")
            }
            Self::InvalidGround { vertex } => {
                write!(f, "vertex {vertex} has a negative or non-finite ground")
            }
        }
    }
}

impl std::error::Error for ViewError {}

pub(super) fn nonnegative<T: Real>(x: T) -> bool {
    x.is_finite() && x >= T::zero()
}

/// A slice of `found` entries where the view fixed `expected`.
pub(super) fn length(expected: usize, found: usize) -> Result<(), ViewError> {
    match found == expected {
        true => Ok(()),
        false => Err(ViewError::LengthMismatch { expected, found }),
    }
}

/// The first entry `valid` rejects, reported by `error` at its index.
pub(super) fn check<T: Real>(
    entries: &[T],
    valid: fn(T) -> bool,
    error: fn(usize) -> ViewError,
) -> Result<(), ViewError> {
    match entries.iter().position(|&x| !valid(x)) {
        Some(at) => Err(error(at)),
        None => Ok(()),
    }
}

/// Each row below `n`.
pub(super) fn check_rows<I: PrimInt>(n: usize, rows: &[I]) -> Result<(), ViewError> {
    let in_range = |row: &I| row.to_usize().is_some_and(|row| row < n);
    match rows.iter().position(|row| !in_range(row)) {
        Some(column) => Err(ViewError::IndexOutOfRange { column }),
        None => Ok(()),
    }
}

/// One `b` per `a`, each column's two ends distinct rows below `n`.
pub(super) fn check_ends<I: PrimInt>(n: usize, a: &[I], b: &[I]) -> Result<(), ViewError> {
    length(a.len(), b.len())?;
    let in_range = |end: &I| end.to_usize().is_some_and(|end| end < n);
    for (column, (end_a, end_b)) in a.iter().zip(b).enumerate() {
        if !in_range(end_a) || !in_range(end_b) {
            return Err(ViewError::IndexOutOfRange { column });
        }
        if end_a == end_b {
            return Err(ViewError::RepeatedVertex { column });
        }
    }
    Ok(())
}
