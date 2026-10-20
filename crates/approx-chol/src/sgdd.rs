//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod null_vectors;
mod reduction;
mod scaling;
mod signature;

pub use block::ReduceError;
pub(crate) use block::{Balance, Grounding};
pub(crate) use block::{IrreducibleBlock, Sgdd, Summands};
pub(crate) use congruence::Restriction;
pub(crate) use reduction::within_band;
pub use scaling::{ScalingPolicy, ScalingReport};
