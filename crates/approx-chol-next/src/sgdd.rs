//! `A` as its connected components, each reduced to `M = F · A_c · F`, decided once per input.

mod component;
mod congruence;
mod decomposition;
mod null_vectors;
mod reduction;
mod scaling;
mod signature;

pub(crate) use component::{Balance, Frame, Grounding, OffDiagonal};
pub use component::{Component, FromCsrError, ReduceError, Sgdd};
pub use scaling::{Existence, ScalingPolicy, ScalingReport};
