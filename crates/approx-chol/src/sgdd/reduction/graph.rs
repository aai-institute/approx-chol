use num_traits::PrimInt;

use crate::input::FactorWidth2;
use crate::sgdd::block::{Balance, Congruent, ReduceError};
use crate::sgdd::decomposition::{assemble_incidence, AdjacencyLayout};
use crate::sgdd::scaling::ScaleVector;
use crate::sgdd::Reduced;
use crate::{Index, Real, Threads};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` an incidence matrix, so dominant by shape; across a bipartition it is `e_a + e_b`, summed switched.
    pub(super) fn from_incidence<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        threads: Threads,
    ) -> Result<Self, ReduceError> {
        assemble_incidence(cells, v)?.reduced(threads, |vertices, (off_diagonal, grounding), _| {
            let m = Congruent::incidence(vertices, off_diagonal);
            Ok(m.block(Balance::Balanced, grounding, ScaleVector::Identity, 0.0))
        })
    }
}
