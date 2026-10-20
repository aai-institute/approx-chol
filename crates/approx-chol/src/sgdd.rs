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
pub(crate) use block::{IrreducibleBlock, Summands};
pub(crate) use congruence::{FactorTail, Restriction};
pub(crate) use reduction::within_band;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::{Index, Real};
use decomposition::Components;

/// `A` as its reduced summands; built only from extraction, so they cover `A` once.
#[derive(Clone, Debug)]
pub(crate) struct Sgdd<T = f64, I = u32> {
    n: usize,
    summands: Summands<IrreducibleBlock<T, I>, I, T>,
}

impl<T: Real, I: Index> Sgdd<T, I> {
    fn new(components: Components<IrreducibleBlock<T, I>, I, T>) -> Self {
        let Components {
            partition,
            summands,
        } = components;
        Self {
            n: partition.n(),
            summands,
        }
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn into_summands(self) -> Summands<IrreducibleBlock<T, I>, I, T> {
        self.summands
    }
}
