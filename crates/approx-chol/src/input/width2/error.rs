//! Why an input constructor rejected its slices, and the checks that say so.

use num_traits::PrimInt;

use crate::Real;

/// Why an input constructor rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the one the input fixed: the edge or column count, or `n`.
    LengthMismatch {
        /// The length the input fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// An edge or column names a vertex or row that does not exist.
    IndexOutOfRange {
        /// The offending edge or column.
        entry: usize,
    },
    /// An edge or column names one vertex at both ends.
    RepeatedVertex {
        /// The offending edge or column.
        entry: usize,
    },
    /// A weight is negative or not finite.
    InvalidWeight {
        /// The offending edge or column.
        entry: usize,
    },
    /// A column scale is not finite.
    InvalidScale {
        /// The offending column.
        entry: usize,
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
                write!(f, "slice has length {found}, expected {expected}")
            }
            Self::IndexOutOfRange { entry } => write!(f, "entry {entry} names a missing vertex"),
            Self::RepeatedVertex { entry } => write!(f, "entry {entry} names one vertex twice"),
            Self::InvalidWeight { entry } => {
                write!(f, "entry {entry} has a negative or non-finite weight")
            }
            Self::InvalidScale { entry } => write!(f, "entry {entry} has a non-finite scale"),
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

/// A slice of `found` entries where the input fixed `expected`.
pub(super) fn length(expected: usize, found: usize) -> Result<(), ViewError> {
    match found == expected {
        true => Ok(()),
        false => Err(ViewError::LengthMismatch { expected, found }),
    }
}

/// The first entry `valid` rejects, reported by `error` at its index.
pub(super) fn check<T: Copy>(
    entries: &[T],
    valid: impl Fn(T) -> bool,
    error: fn(usize) -> ViewError,
) -> Result<(), ViewError> {
    match entries.iter().position(|&x| !valid(x)) {
        Some(at) => Err(error(at)),
        None => Ok(()),
    }
}

pub(super) fn below<I: PrimInt>(n: usize, row: I) -> bool {
    row.to_usize().is_some_and(|row| row < n)
}

/// One `b` per `a`, each column's two ends distinct rows below `n`.
pub(super) fn check_ends<I: PrimInt>(n: usize, a: &[I], b: &[I]) -> Result<(), ViewError> {
    length(a.len(), b.len())?;
    for (entry, (end_a, end_b)) in a.iter().zip(b).enumerate() {
        if !below(n, *end_a) || !below(n, *end_b) {
            return Err(ViewError::IndexOutOfRange { entry });
        }
        if end_a == end_b {
            return Err(ViewError::RepeatedVertex { entry });
        }
    }
    Ok(())
}
