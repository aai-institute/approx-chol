//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod reduction;
mod scaling;

pub use block::ReduceError;
use block::WithClamp;
pub(crate) use block::{BlockComplement, IrreducibleBlock};
pub(crate) use block::{Grounding, Sheets};
pub(crate) use congruence::{Congruence, FactorTail, Selection};
pub(crate) use decomposition::Summands;
pub(crate) use reduction::within_band;
pub use scaling::{Clamp, ScalingOutcome, ScalingPolicy};

use crate::{in_range, Index, Real};
use decomposition::Components;

/// A reduction's summands, and the clamps of `A`'s components, ordered by first vertex in `A`'s numbering.
#[derive(Clone, Debug)]
pub(crate) struct Reduction<B, I, T> {
    pub(crate) summands: Summands<B, I, T>,
    pub(crate) clamps: Vec<Clamp>,
}

/// `A`'s reduced summands, covering `A` once by extraction; only debug builds check it.
pub(crate) type Reduced<T, I> = Reduction<IrreducibleBlock<T, I>, I, T>;

/// A bipartite `A`'s Schur complement onto one side, reduced; lone and dead vertices in the kept side's numbering.
pub(crate) type Complement<T, I> = Reduction<BlockComplement<T, I>, I, T>;

impl<T: Real, I: Index> Components<WithClamp<T, I>, I, T> {
    fn into_reduced(self) -> Reduced<T, I> {
        let Components {
            partition,
            summands: Summands { blocks, lone, dead },
        } = self;
        let mut clamps = Vec::new();
        let blocks: Vec<_> = (blocks.into_iter())
            .map(|(block, clamp)| {
                clamps.extend(clamp.map(|clamp| *clamp));
                block
            })
            .collect();
        // A Gram lists its folded blocks before its frustrated ones.
        clamps.sort_unstable_by_key(|clamp| clamp.vertices[0]);
        let coupled: usize = blocks.iter().map(IrreducibleBlock::vertex_count).sum();
        let dimension = coupled + lone.len() + dead.len();
        debug_assert_eq!(partition.n(), dimension, "summands cover A once");
        let summands = Summands { blocks, lone, dead };
        Reduction { summands, clamps }
    }
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
