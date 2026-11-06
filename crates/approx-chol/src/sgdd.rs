//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod reduction;
mod scaling;

pub use block::ReduceError;
pub(crate) use block::{BlockComplement, IrreducibleBlock};
pub(crate) use block::{Grounding, Sheets};
pub(crate) use congruence::{FactorTail, Restriction, Selection};
pub(crate) use decomposition::Summands;
pub(crate) use reduction::within_band;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::{in_range, Index, Real};
use decomposition::Components;

/// `A`'s reduced summands, covering `A` once by extraction; only debug builds check it.
pub(crate) type Reduced<T, I> = Summands<IrreducibleBlock<T, I>, I, T>;

/// A bipartite `A`'s Schur complement onto one side, reduced; lone and dead vertices in the kept side's numbering.
pub(crate) type Complement<T, I> = Summands<BlockComplement<T, I>, I, T>;

impl<T: Real, I: Index> Components<IrreducibleBlock<T, I>, I, T> {
    fn into_reduced(self) -> Reduced<T, I> {
        let Components {
            partition,
            summands,
        } = self;
        let coupled: usize = summands
            .blocks
            .iter()
            .map(IrreducibleBlock::vertex_count)
            .sum();
        let dimension = coupled + summands.lone.len() + summands.dead.len();
        debug_assert_eq!(partition.n(), dimension, "summands cover A once");
        summands
    }
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
