use num_traits::PrimInt;

use super::{into_sgdd, Congruent};
use crate::input::{BipartiteLaplacian, Laplacian, SignedGraph};
use crate::sgdd::block::{Grounding, ReduceError, Sgdd};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_bipartite, assemble_laplacian, assemble_signed};
use crate::sgdd::signature;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, Index, Real};

impl<'a, T: Real, J: PrimInt, I: Index> TryFrom<Laplacian<'a, T, J>> for Sgdd<T, I> {
    type Error = ReduceError;

    /// Dominant with surplus `g` by shape: a block floats exactly when its ground is zero.
    fn try_from(graph: Laplacian<'a, T, J>) -> Result<Self, Self::Error> {
        assemble_laplacian(graph)?
            .extract(|(vertices, off_diagonal, grounding)| {
                // A lone vertex has no cell, so only a ground makes it a block.
                if vertices.len() == 1 && !grounding.is_grounded() {
                    return Ok(None);
                }
                let m = Congruent::new(vertices, off_diagonal, Congruence::Identity);
                m.balanced(grounding).map(Some)
            })
            .map(into_sgdd)
    }
}

impl<'a, T: Real, J: PrimInt, I: Index> TryFrom<BipartiteLaplacian<'a, T, J>> for Sgdd<T, I> {
    type Error = ReduceError;

    /// `M`, a bipartite Laplacian, with the kept side negated folding `A` into it; a lone vertex is dead.
    fn try_from(m: BipartiteLaplacian<'a, T, J>) -> Result<Self, Self::Error> {
        assemble_bipartite::<T, I>(m)?
            .extract(|(vertices, off_diagonal)| {
                let OffDiagonal::Bipartite(cross) = &off_diagonal else {
                    return Ok(None);
                };
                let signs = Congruence::NegateFrom(cross.n_rows());
                if let Some(i) =
                    (off_diagonal.abs_row_sums().iter()).position(|&d| !diagonal_in_range(d))
                {
                    let vertex = vertices[i].index();
                    return Err(ReduceError::OutOfRange { vertex });
                }
                let m = Congruent {
                    vertices,
                    congruence: signs,
                    off_diagonal,
                };
                m.balanced(Grounding::ungrounded()).map(Some)
            })
            .map(into_sgdd)
    }
}

impl<'a, T: Real, J: PrimInt, I: Index> TryFrom<SignedGraph<'a, T, J>> for Sgdd<T, I> {
    type Error = ReduceError;

    /// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
    fn try_from(graph: SignedGraph<'a, T, J>) -> Result<Self, Self::Error> {
        assemble_signed(graph)?
            .extract(|(vertices, off_diagonal, grounding)| {
                // A lone vertex has no cell, so only its cancelled pairs make it a block.
                if vertices.len() == 1 && !grounding.is_grounded() {
                    return Ok(None);
                }
                let block = match signature::folding_signs(&off_diagonal) {
                    Some(signs) => {
                        Congruent::new(vertices, off_diagonal, signs).balanced(grounding)
                    }
                    None => Congruent::new(vertices, off_diagonal, Congruence::Identity)
                        .frustrated(grounding),
                };
                block.map(Some)
            })
            .map(into_sgdd)
    }
}
