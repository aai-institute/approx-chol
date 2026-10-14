use num_traits::PrimInt;

use super::{into_sgdd, Congruent};
use crate::input::{BipartiteLaplacian, Laplacian, SignedGraph};
use crate::sgdd::block::{Grounding, ReduceError, Sgdd};
use crate::sgdd::congruence::Congruence;
use crate::sgdd::decomposition::{assemble_bipartite, assemble_general, assemble_signed, Cells};
use crate::sgdd::signature;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, narrow, widen, Real};

impl<'a, T: Real, I: PrimInt> TryFrom<Laplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant with surplus `g` by shape: a block floats exactly when its ground is zero.
    fn try_from(graph: Laplacian<'a, T, I>) -> Result<Self, Self::Error> {
        let ground = graph.ground();
        assemble_general(graph)?
            .extract(|(vertices, off_diagonal, diagonal): Cells<T>| {
                let gather =
                    |g: &[T]| Grounding::of(vertices.iter().map(|&v| g[v as usize]).collect());
                let grounding = ground.map_or_else(Grounding::ungrounded, gather);
                let g = |i: usize| grounding.surplus().map_or(T::zero(), |g| g[i]);
                // A lone vertex has no edge, so only a ground makes it a block.
                if vertices.len() == 1 && diagonal[0] + g(0) == T::zero() {
                    return Ok(None);
                }
                // `A`'s diagonal `d + g`, summed in `f64`, and each ground like any entry.
                let out_of_range =
                    |i: usize| !diagonal_in_range(narrow::<T>(widen(diagonal[i]) + widen(g(i))));
                if let Some(i) = (0..vertices.len()).find(|&i| out_of_range(i)) {
                    let vertex = vertices[i];
                    return Err(ReduceError::OutOfRange { vertex });
                }
                let m = Congruent::new(vertices, off_diagonal, Congruence::Identity);
                m.balanced(grounding).map(Some)
            })
            .map(into_sgdd)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteLaplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// `M`, a bipartite Laplacian, with the kept side negated folding `A` into it; a lone vertex is dead.
    fn try_from(m: BipartiteLaplacian<'a, T, I>) -> Result<Self, Self::Error> {
        assemble_bipartite(m)?
            .extract(|(vertices, off_diagonal)| {
                let OffDiagonal::Bipartite(cross) = &off_diagonal else {
                    return Ok(None);
                };
                let signs = Congruence::NegateFrom(cross.n_rows());
                if let Some(i) =
                    (off_diagonal.abs_row_sums().iter()).position(|&d| !diagonal_in_range(d))
                {
                    let vertex = vertices[i];
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

impl<'a, T: Real, I: PrimInt> TryFrom<SignedGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
    fn try_from(graph: SignedGraph<'a, T, I>) -> Result<Self, Self::Error> {
        let (a, surplus) = assemble_signed(graph)?;
        a.extract(|(vertices, off_diagonal, diagonal): Cells<T>| {
            if diagonal == [T::zero()] {
                return Ok(None);
            }
            if let Some(i) = diagonal.iter().position(|&d| !diagonal_in_range(d)) {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let grounding = Grounding::of(vertices.iter().map(|&v| surplus[v as usize]).collect());
            let block = match signature::folding_signs(&off_diagonal) {
                Some(signs) => Congruent::new(vertices, off_diagonal, signs).balanced(grounding),
                None => Congruent::new(vertices, off_diagonal, Congruence::Identity)
                    .frustrated(grounding),
            };
            block.map(Some)
        })
        .map(into_sgdd)
    }
}
