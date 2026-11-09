use std::convert::Infallible;

use crate::{InputError, ReduceError};

/// Why an input did not factor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The input is not a matrix of its form.
    Input(InputError),
    /// A block has no SDDM form.
    Reduce(ReduceError),
}

impl From<InputError> for Error {
    fn from(error: InputError) -> Self {
        Self::Input(error)
    }
}

impl From<ReduceError> for Error {
    fn from(error: ReduceError) -> Self {
        Self::Reduce(error)
    }
}

/// Lets an infallible `From` input pass where [`factorize`](crate::factorize) takes `TryInto`.
impl From<Infallible> for Error {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Input(error) => write!(f, "invalid input: {error}"),
            Self::Reduce(error) => write!(f, "cannot reduce: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            Self::Reduce(error) => Some(error),
        }
    }
}
