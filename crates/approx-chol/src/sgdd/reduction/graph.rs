use num_traits::PrimInt;

use crate::input::Gram;
use crate::sgdd::block::{Congruent, ReduceError};
use crate::sgdd::decomposition::Components;
use crate::sgdd::decomposition::{assemble_incidence, AdjacencyLayout, IncidenceBlock};
use crate::sgdd::Reduced;
use crate::storage::{Balance, Index, Real};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` an incidence matrix, so dominant by shape; across a bipartition it is `e_a + e_b`, summed switched.
    pub(super) fn from_incidence<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: Gram<'_, T, J>,
        surplus: Option<&[T]>,
    ) -> Result<Self, ReduceError> {
        assemble_incidence(cells, v, surplus)?
            .extract(
                |IncidenceBlock {
                     adjacency,
                     grounding,
                 }| {
                    let m = Congruent::incidence(adjacency);
                    m.block(Balance::Balanced(grounding), None)
                },
            )
            .map(Components::into_reduced)
    }
}
