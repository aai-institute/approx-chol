//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod reduction;
mod scaling;

pub use block::ReduceError;
pub(crate) use block::{Balance, Grounding};
pub(crate) use block::{BlockComplement, IrreducibleBlock};
pub(crate) use congruence::{Congruence, FactorTail};
pub(crate) use decomposition::{Components, Summands};
pub(crate) use reduction::within_band;
pub use scaling::{Clamp, ScalingOutcome, ScalingPolicy};

use crate::{in_range, Real};

/// `A`'s reduced summands, covering `A` once: each vertex is in one component of its partition.
pub(crate) type Reduced<T, I> = Summands<IrreducibleBlock<T, I>, I, T>;

/// A bipartite `A`'s Schur complement onto one side, reduced; every vertex in the kept side's numbering.
pub(crate) type Complement<T, I> = Summands<BlockComplement<T, I>, I, T>;

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
