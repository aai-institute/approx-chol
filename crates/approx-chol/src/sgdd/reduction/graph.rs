use num_traits::PrimInt;

use super::Congruent;
use crate::input::FactorWidth2;
use crate::sgdd::block::{Balance, ReduceError};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::Components;
use crate::sgdd::decomposition::{assemble_incidence, AdjacencyLayout};
use crate::sgdd::Reduced;
use crate::storage::OffDiagonal;
use crate::{Index, Real};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` an incidence matrix, so dominant by shape; across a bipartition it is `e_a + e_b`, summed switched.
    pub(super) fn from_incidence<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
    ) -> Result<Self, ReduceError> {
        assemble_incidence(cells, v)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let congruence = match &off_diagonal {
                    OffDiagonal::Bipartite(c) => Congruence::NegateFrom(c.n_rows()),
                    OffDiagonal::General(_) => Congruence::Identity,
                };
                let m = Congruent {
                    vertices,
                    congruence,
                    off_diagonal,
                    report: None,
                };
                m.block(Balance::Balanced(grounding))
            })
            .map(Components::into_reduced)
    }
}
