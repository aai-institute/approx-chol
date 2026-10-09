//! The leading independent block eliminated at once: its exact columns, and the kept block's complement.

use std::sync::Arc;

use crate::config::{CliqueRule, InnerParallelism};
use crate::laplacian::Sddm;
use crate::sampling::{CdfSampler, Streams};
use crate::sgdd::Bipartite;
use crate::storage::{CrossBlock, Fill};
use crate::Real;

/// Stored columns `C / d_k`, applied as two products because the block's steps commute.
#[derive(Clone, Debug)]
pub(crate) struct Round<T> {
    /// Shared with the part, signed for a frustrated one: the cover never doubles it.
    cross: Arc<CrossBlock<T>>,
    inv_pivot: Box<[T]>,
}

impl<T: Real> Round<T> {
    pub(crate) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// `y_e = D_e⁻¹ b_e`, then `b_k −= Cᵀ y_e`.
    pub(crate) fn forward(&self, x: &mut [T], inner: InnerParallelism) {
        todo!()
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`.
    pub(crate) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        todo!()
    }
}

/// A round before its stored form is taken, so one pivot vector serves every complement and the solve.
pub(crate) struct RoundBuilder<'a, T> {
    round: Round<T>,
    part: &'a Bipartite<T>,
}

impl<'a, T: Real> RoundBuilder<'a, T> {
    /// Pivots are `Σ|c_kj| + s_k`, the same on both sheets of a cover.
    pub(crate) fn new(part: &'a Bipartite<T>) -> Self {
        todo!()
    }

    /// What a Cholesky of the complement solves for, known before it is built.
    pub(crate) fn complement_eliminated(&self) -> usize {
        let rows = self.part.sheets().count() * self.part.cross().n_kept();
        self.part.ground().eliminated(rows)
    }

    /// The kept block's complement on the part's sheets; the stars share no edge, so they run in parallel.
    pub(crate) fn schur(&self, rule: CliqueRule, streams: Streams) -> Sddm<T> {
        todo!()
    }

    pub(crate) fn finish(self) -> Round<T> {
        self.round
    }
}

/// A clique tree over the star, the ground sampled as a final vertex.
fn sample_tree<T: Real>(
    star: &[(u32, T)],
    ground: T,
    sampler: &mut CdfSampler<T>,
    fill: &mut Fill<T>,
) {
    todo!()
}

/// The whole clique: `w_i w_j / d` per pair, and each neighbor's share of the ground.
fn exact_clique<T: Real>(star: &[(u32, T)], ground: T, fill: &mut Fill<T>) {
    todo!()
}
