//! Low-level API: the [`Builder`] behind [`factorize`](crate::factorize), and the
//! clique-tree samplers on their own, for callers that eliminate a star outside a
//! full factorization.

pub use crate::approx_chol::clique_tree::{clique_tree_sample, clique_tree_sample_multi};
pub use crate::approx_chol::Builder;
