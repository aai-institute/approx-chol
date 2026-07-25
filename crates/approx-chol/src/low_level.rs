//! Low-level API for advanced use cases.
//!
//! This module exposes concrete types from the factorization internals for
//! power users who need custom factorization pipelines or research access to
//! elimination sequences and samplers.
//!
//! Most users should prefer the high-level [`factorize`](crate::factorize)
//! and [`factorize_with`](crate::factorize_with) functions; see [`Builder`] for
//! the entry point and a usage example.
//!
//! [`Builder`]: crate::low_level::Builder

pub use crate::approx_chol::clique_tree::{clique_tree_sample, clique_tree_sample_multi};
pub use crate::approx_chol::Builder;
