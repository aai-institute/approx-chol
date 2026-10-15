//! What the caller states about `A`, each view by its type.

mod csr;
mod width2;

pub(crate) use csr::Canonical;
pub use csr::{Csr, CsrError, UpperCsr};
pub(crate) use width2::bipartite::BipartiteLaplacian;
pub use width2::bipartite::{BipartiteGram, FactorWidth1};
pub use width2::edges::Weights;
pub use width2::laplacian::Laplacian;
pub use width2::signed::SignedGraph;
pub(crate) use width2::{BipartiteColumns, Column, IntoColumns};
pub use width2::{FactorWidth2, ViewError};
