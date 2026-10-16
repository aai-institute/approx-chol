use std::convert::Infallible;

use crate::{CsrError, ReduceError};

/// Why an input did not factor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The arrays are not a symmetric CSR matrix.
    InvalidCsr(CsrError),
    /// A block has no SDDM form.
    Reduce(ReduceError),
}

impl From<CsrError> for Error {
    fn from(error: CsrError) -> Self {
        Self::InvalidCsr(error)
    }
}

impl From<ReduceError> for Error {
    fn from(error: ReduceError) -> Self {
        Self::Reduce(error)
    }
}

/// Lets an infallible `From` view pass where [`factorize`](crate::factorize) takes `TryInto`.
impl From<Infallible> for Error {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidCsr(error) => write!(f, "invalid CSR: {error}"),
            Self::Reduce(error) => write!(f, "cannot reduce: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCsr(error) => Some(error),
            Self::Reduce(error) => Some(error),
        }
    }
}
