//! What the caller states about `A`, each view by its type.

mod csr;
mod width2;

pub use csr::{Csr, CsrError};
pub(crate) use width2::Columns;
pub use width2::{Gram, Graph, Side, Sides, ViewError, Weights};
