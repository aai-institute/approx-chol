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
pub(crate) use congruence::{Congruence, FactorTail};
pub(crate) use decomposition::{Components, Summands};
pub(crate) use reduction::within_band;
pub use scaling::{Clamp, ScalingOutcome, ScalingPolicy};

use crate::{in_range, Index, Real};

/// A reduction's summands, and the clamps of `A`'s components, ordered by lowest vertex in `A`'s numbering.
#[derive(Clone, Debug)]
pub(crate) struct Reduction<B, I, T> {
    pub(crate) summands: Summands<B, I, T>,
    pub(crate) clamps: Vec<Clamp>,
}

/// `A`'s reduced summands, covering `A` once: each vertex is in one component of its partition.
pub(crate) type Reduced<T, I> = Reduction<IrreducibleBlock<T, I>, I, T>;

/// A bipartite `A`'s Schur complement onto one side, reduced; every vertex in the kept side's numbering.
pub(crate) type Complement<T, I> = Reduction<BlockComplement<T, I>, I, T>;

impl<T: Real, I: Index> Summands<WithClamp<T, I>, I, T> {
    fn into_reduced(self) -> Reduced<T, I> {
        let Summands { components, blocks } = self;
        let mut clamps = Vec::new();
        let blocks: Vec<_> = (blocks.into_iter())
            .map(|(block, clamp)| {
                clamps.extend(clamp.map(|clamp| *clamp));
                block
            })
            .collect();
        let summands = Summands { components, blocks };
        Reduction { summands, clamps }
    }
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
