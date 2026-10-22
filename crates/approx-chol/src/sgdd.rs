//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod reduction;
mod scaling;

pub(crate) use block::IrreducibleBlock;
pub use block::ReduceError;
pub(crate) use block::{Balance, Grounding};
pub(crate) use congruence::{FactorTail, Restriction, Selection};
pub(crate) use decomposition::Summands;
pub(crate) use reduction::within_band;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::{in_range, Index, Real};
use decomposition::Components;

/// `A`'s reduced summands; built only from extraction, so they cover `A` once.
pub(crate) type Reduced<T, I> = Summands<IrreducibleBlock<T, I>, I, T>;

impl<T: Real, I: Index> Reduced<T, I> {
    /// Dimension of `A`, which the summands cover once.
    pub(crate) fn dimension(&self) -> usize {
        let coupled: usize = self.blocks.iter().map(IrreducibleBlock::vertex_count).sum();
        coupled + self.lone.len() + self.dead.len()
    }
}

impl<T: Real, I: Index> Components<IrreducibleBlock<T, I>, I, T> {
    fn into_reduced(self) -> Reduced<T, I> {
        let Components {
            partition,
            summands,
        } = self;
        debug_assert_eq!(partition.n(), summands.dimension(), "summands cover A once");
        summands
    }
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
