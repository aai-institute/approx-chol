use num_traits::PrimInt;

use super::{folding_signs, Congruent};
use crate::input::{Gram, Switching};
use crate::sgdd::block::{Balance, ReduceError};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_incidence, AdjacencyLayout};
use crate::sgdd::Sgdd;
use crate::{Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// `V` an incidence matrix, so dominant by shape: a switching its partition fixes is read into the cells, else the signs are searched.
    pub(super) fn from_incidence<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: Gram<'_, T, J>,
    ) -> Result<Self, ReduceError> {
        // Under `NegateRight`, the right part is read negated, so every cell is switched as it is summed.
        let (switching, v) = v
            .switched()
            .map_or((None, v), |(switching, read)| (Some(switching), read));
        assemble_incidence(cells, v)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let congruence = match switching {
                    Some(Switching::Identity) => Congruence::Identity,
                    Some(Switching::NegateRight) => Congruence::NegateFrom(off_diagonal.stored().1),
                    None => {
                        return match folding_signs(&off_diagonal) {
                            Some(signs) => Congruent::new(vertices, off_diagonal, signs)
                                .block(Balance::Balanced(grounding)),
                            None => Congruent::new(vertices, off_diagonal, Congruence::Identity)
                                .block(Balance::Frustrated(grounding)),
                        }
                    }
                };
                let m = Congruent {
                    vertices,
                    congruence,
                    off_diagonal,
                    report: None,
                };
                m.block(Balance::Balanced(grounding))
            })
            .map(Sgdd::new)
    }
}
