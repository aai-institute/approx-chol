use num_traits::PrimInt;

use super::{folding_signs, Congruent};
use crate::input::{End, Fold, Gram};
use crate::sgdd::block::{Balance, ReduceError};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_incidence, Shape};
use crate::sgdd::Sgdd;
use crate::{Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// `V` an incidence matrix, so dominant by shape: a fold its shape fixes is read into the cells, else the signs are searched.
    pub(super) fn from_incidence<J: PrimInt, G: Shape, A: End<T>, B: End<T>>(
        v: Gram<'_, T, J, G, A, B>,
    ) -> Result<Self, ReduceError> {
        let fold = v.fold();
        let cells = match fold {
            // Read negated, the head's part folds every cell as it is summed.
            Some(Fold::NegatedHead) => assemble_incidence(v.negated_head())?,
            _ => assemble_incidence(v)?,
        };
        cells
            .extract(|(vertices, off_diagonal, grounding)| {
                let congruence = match fold {
                    Some(Fold::Identity) => Congruence::Identity,
                    Some(Fold::NegatedHead) => Congruence::NegateFrom(off_diagonal.stored().1),
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
