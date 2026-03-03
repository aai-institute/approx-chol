//! Low-level API for advanced use cases.
//!
//! This module exposes concrete types from the factorization internals for
//! power users who need custom factorization pipelines, ordering control,
//! or research access to elimination sequences and samplers.
//!
//! Most users should prefer the high-level [`factorize`](crate::factorize)
//! and [`factorize_with`](crate::factorize_with) functions.
//!
//! # Usage
//!
//! ```
//! use approx_chol::{Config, CsrRef};
//! use approx_chol::low_level::{Builder, Ordering};
//!
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let factor = Builder::new(Config::default())
//!     .ordering(Ordering::StaticAMD)
//!     .build(csr)?;
//! assert_eq!(factor.n(), 4);
//! # Ok::<(), approx_chol::Error>(())
//! ```

// Builder for custom factorization pipelines
pub use crate::approx_chol::Builder;

// Ordering enum (elimination strategy selection)
pub use crate::approx_chol::Ordering;

// Sampling
pub use crate::sampling::CdfSampler;

// Decomposition internals (read-only views into the factor)
pub use crate::approx_chol::decomposition::{EliminationSequence, EliminationStep};

// Star clique sampling utilities (AC + AC2 variants)
pub use crate::star_clique::{clique_tree_sample, clique_tree_sample_multi};
