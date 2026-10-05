mod config;
mod factorization;
mod pipeline;

pub use config::{Backend, Config, ExactFailure};
#[cfg(feature = "serde")]
pub use factorization::FACTOR_FORMAT_VERSION;
pub use factorization::{CliqueTreeSampler, Factor, Fallback, SolveError};
pub(crate) use pipeline::factorize;
