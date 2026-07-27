mod builder;
pub(crate) mod clique_tree;
pub(crate) mod decomposition;
mod star;

pub use builder::{Builder, Config};
pub use decomposition::{Factor, SolveError};
