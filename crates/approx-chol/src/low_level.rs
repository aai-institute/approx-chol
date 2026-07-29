//! Low-level API: the [`Builder`] behind [`factorize`](crate::factorize), and the
//! clique-tree sampler on its own, for callers that eliminate a star outside a
//! full factorization.

pub use crate::approx_chol::{clique_tree_sample, Builder};
