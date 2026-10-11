//! What the caller states about `A`, each view by its type.

mod csr;
mod width2;

pub(crate) use csr::{Canonical, UpperCanonical};
pub use csr::{Csr, CsrError, UpperCsr};
pub use width2::bipartite::{BipartiteGram, BipartiteGraph, BipartiteLaplacian, Incidence, Side};
pub use width2::edges::Weights;
pub use width2::laplacian::Laplacian;
pub use width2::signed::SignedGraph;
pub(crate) use width2::{Column, IntoColumns};
pub use width2::{FactorWidth2, ViewError};
