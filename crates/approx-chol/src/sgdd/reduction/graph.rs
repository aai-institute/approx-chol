use num_traits::PrimInt;

use super::{into_sgdd, Congruent};
use crate::input::{BipartiteLaplacian, Laplacian, SignedGraph};
use crate::sgdd::block::{Balance, ReduceError, Sgdd};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_bipartite, assemble_laplacian, assemble_signed};
use crate::sgdd::signature;
use crate::{Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// Dominant with surplus `g` by shape: a block floats exactly when its ground is zero.
    pub fn from_laplacian<J: PrimInt>(graph: Laplacian<'_, T, J>) -> Result<Self, ReduceError> {
        assemble_laplacian(graph)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let m = Congruent::new(vertices, off_diagonal, Congruence::Identity);
                m.block(Balance::Balanced(grounding))
            })
            .map(into_sgdd)
    }

    /// `M`, a bipartite Laplacian, with the kept side negated folding `A` into it.
    pub fn from_bipartite_laplacian<J: PrimInt>(
        m: BipartiteLaplacian<'_, T, J>,
    ) -> Result<Self, ReduceError> {
        assemble_bipartite::<T, I>(m)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let m = Congruent {
                    vertices,
                    congruence: Congruence::NegateFrom(off_diagonal.stored().1),
                    off_diagonal,
                    report: None,
                };
                m.block(Balance::Balanced(grounding))
            })
            .map(into_sgdd)
    }

    /// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
    pub fn from_signed_graph<J: PrimInt>(
        graph: SignedGraph<'_, T, J>,
    ) -> Result<Self, ReduceError> {
        assemble_signed(graph)?
            .extract(|(vertices, off_diagonal, grounding)| {
                match signature::folding_signs(&off_diagonal) {
                    Some(signs) => Congruent::new(vertices, off_diagonal, signs)
                        .block(Balance::Balanced(grounding)),
                    None => Congruent::new(vertices, off_diagonal, Congruence::Identity)
                        .block(Balance::Frustrated(grounding)),
                }
            })
            .map(into_sgdd)
    }
}
