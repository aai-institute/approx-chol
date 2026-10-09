//! The leading independent block eliminated at once: its exact columns, and the kept block's complement.

use std::sync::Arc;

use crate::clique_tree::{EdgeCount, Streams};
use crate::config::{CliqueRule, InnerParallelism};
use crate::sddm::{Grounding, Sddm, Sheets};
use crate::storage::{CrossBlock, Fill};
use crate::Real;

/// Stored columns `C / d_e`, applied as two products because the block's steps commute.
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
    grounding: &'a Grounding<T>,
    sheets: Sheets,
}

impl<'a, T: Real> RoundBuilder<'a, T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(crate) fn new(
        cross: &Arc<CrossBlock<T>>,
        grounding: &'a Grounding<T>,
        sheets: Sheets,
    ) -> Self {
        todo!()
    }

    pub(crate) fn n_eliminated(&self) -> usize {
        self.round.n_eliminated()
    }

    /// The complement is grounded iff the part is, so its rank is known before it is built.
    pub(crate) fn complement_rank(&self) -> usize {
        let rows = self.sheets.count() * self.round.cross.n_kept();
        self.grounding.rank(rows)
    }

    /// The kept block's Schur complement on the part's sheets; stars run in parallel.
    pub(crate) fn schur(&self, rule: CliqueRule, streams: Streams) -> Sddm<'static, T> {
        todo!()
    }

    pub(crate) fn finish(self) -> Round<T> {
        self.round
    }
}

/// Each eliminated star's clique tree, the ground a final neighbor, under the star's own stream.
fn sampled_fill<T: Real, C: EdgeCount>(
    builder: &RoundBuilder<'_, T>,
    split: C::Split,
    streams: Streams,
) -> Fill<T> {
    todo!()
}

/// The whole clique: `w_i w_j / d` per pair, and each neighbor's share of the ground.
fn exact_clique<T: Real>(star: &[(u32, T)], ground: T, fill: &mut Fill<T>) {
    todo!()
}
