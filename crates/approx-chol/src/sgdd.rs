//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod null_vectors;
mod reduction;
mod scaling;
mod signature;

pub use block::ReduceError;
pub(crate) use block::{Balance, Grounding, Restriction};
pub(crate) use block::{IrreducibleBlock, Sgdd};
pub use scaling::{Existence, ScalingPolicy, ScalingReport};
