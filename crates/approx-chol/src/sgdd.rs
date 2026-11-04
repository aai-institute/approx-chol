//! `A` as its irreducible blocks (connected components), each reduced to `M = F · A_c · F`, decided once per input.

mod block;
mod congruence;
mod decomposition;
mod reduction;
mod scaling;

pub use block::ReduceError;
pub(crate) use block::{Reduced, ReducedComplement, ReducedComponent};
pub(crate) use congruence::{Lift, Null, Selection};
pub(crate) use decomposition::{Lone, Summands};
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::storage::{in_range, Adjacency, Index, OffDiagonal, Real};
use decomposition::Components;

/// `A`'s reduced summands, covering `A` once by extraction; only debug builds check it.
pub(crate) type Reduction<T, I> = Summands<ReducedComponent<T, I>, I, T>;

/// A bipartite `A`'s Schur complement onto one side, reduced; lone and dead vertices in the kept side's numbering.
pub(crate) type Complement<T, I> = Summands<ReducedComplement<T, I>, I, T>;

impl<T: Real, I: Index, P: Adjacency<T, I>> Components<Reduced<T, I, P>, I, T> {
    fn into_summands(self) -> Summands<Reduced<T, I, P>, I, T> {
        let Components {
            partition,
            summands,
        } = self;
        let coupled: usize = summands.blocks.iter().map(Reduced::vertex_count).sum();
        let dimension = coupled + summands.lone.len() + summands.dead.len();
        debug_assert_eq!(partition.n(), dimension, "summands cover A once");
        summands
    }
}

impl<T: Real, I: Index, P: Into<OffDiagonal<T, I>>> Summands<Reduced<T, I, P>, I, T> {
    /// Each block as Cholesky receives it, whichever its shape.
    fn into_reduction(self) -> Reduction<T, I> {
        let Self { blocks, lone, dead } = self;
        let blocks = blocks.into_iter().map(Reduced::into_component).collect();
        Summands { blocks, lone, dead }
    }
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
