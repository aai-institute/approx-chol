//! What the caller states about `A`, each view by its type.

mod csr;
mod width2;

pub(crate) use csr::Canonical;
pub use csr::{Csr, CsrError};
pub use width2::bipartite::{BipartiteGram, BipartiteGraph, Incidence, Side, Weights};
pub(crate) use width2::IntoColumns;
pub use width2::{FactorWidth2, ViewError};
