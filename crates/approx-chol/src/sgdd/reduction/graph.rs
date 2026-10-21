use num_traits::PrimInt;

use super::Congruent;
use crate::input::{BipartiteLaplacian, Laplacian};
use crate::sgdd::block::{Balance, ReduceError};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_bipartite, assemble_laplacian};
use crate::sgdd::Sgdd;
use crate::{Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// Dominant with surplus `g` by shape: a block floats exactly when its ground is zero.
    pub(crate) fn from_laplacian<J: PrimInt>(
        graph: Laplacian<'_, T, J>,
    ) -> Result<Self, ReduceError> {
        assemble_laplacian(graph)?
            .extract(|(vertices, off_diagonal, grounding)| {
                let m = Congruent::new(vertices, off_diagonal, Congruence::Identity);
                m.block(Balance::Balanced(grounding))
            })
            .map(Sgdd::new)
    }

    /// `M`, a bipartite Laplacian, with the kept side negated folding `A` into it.
    pub(crate) fn from_bipartite_laplacian<J: PrimInt>(
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
            .map(Sgdd::new)
    }
}
