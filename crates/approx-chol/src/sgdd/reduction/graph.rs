use num_traits::PrimInt;

use super::{folding_signs, Congruent};
use crate::input::{End, Fold, Gram};
use crate::sgdd::block::{Balance, ReduceError};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_incidence, Shape};
use crate::sgdd::Sgdd;
use crate::{Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// `V` an incidence matrix, so dominant by shape: `read` names the fold its columns were read under, else the signs are searched.
    pub(super) fn from_incidence<J: PrimInt, G: Shape, A: End<T>, B: End<T>>(
        v: Gram<'_, T, J, G, A, B>,
        read: Option<Fold>,
    ) -> Result<Self, ReduceError> {
        assemble_incidence(v)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let congruence = match read {
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
