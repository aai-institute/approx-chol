use num_traits::PrimInt;

use super::ShapedBy;
use crate::input::Gram;
use crate::sgdd::block::{Congruent, ReduceError};
use crate::sgdd::decomposition::Components;
use crate::sgdd::decomposition::{assemble_incidence, AdjacencyLayout, IncidenceBlock};
use crate::storage::{Balance, Index, Real};

/// `V` an incidence matrix, so dominant by shape; across a bipartition it is `e_a + e_b`, summed switched.
pub(super) fn from_incidence<T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: Gram<'_, T, J>,
    surplus: Option<&[T]>,
) -> Result<ShapedBy<T, I, S>, ReduceError> {
    assemble_incidence(cells, v, surplus)?
        .extract(
            |IncidenceBlock {
                 adjacency,
                 grounding,
             }| {
                let m = Congruent::incidence(adjacency);
                m.block(grounding, Balance::Balanced, None)
            },
        )
        .map(Components::into_summands)
}
