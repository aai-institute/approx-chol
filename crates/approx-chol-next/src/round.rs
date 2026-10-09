//! The leading independent block eliminated at once: its exact columns, and the kept block's complement.

use std::sync::Arc;

use crate::clique_tree::{Copies, Streams};
use crate::config::{CliqueRule, InnerParallelism};
use crate::sddm::{self, Sddm};
use crate::sgdd::Signs;
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
    signs: &'a Signs<T>,
}

impl<'a, T: Real> RoundBuilder<'a, T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(crate) fn new(cross: &Arc<CrossBlock<T>>, signs: &'a Signs<T>) -> Self {
        todo!()
    }

    pub(crate) fn n_eliminated(&self) -> usize {
        self.round.n_eliminated()
    }

    /// Known before the complement is built: its surplus `s_k + |C|ᵀ D_e⁻¹ s_e` grounds iff the part's does.
    pub(crate) fn complement_rank(&self) -> usize {
        let rows = self.signs.sheets().count() * self.round.cross.n_kept();
        sddm::rank(rows, self.signs.grounded())
    }

    /// The kept block's Schur complement on the part's sheets; stars run in parallel.
    pub(crate) fn complement(&self, rule: CliqueRule, streams: Streams) -> Sddm<'static, T> {
        todo!()
    }

    pub(crate) fn finish(self) -> Round<T> {
        self.round
    }
}

/// Each eliminated star's clique tree under its own stream, appended to one task's fill.
fn sampled_fill<T: Real, C: Copies>(
    builder: &RoundBuilder<'_, T>,
    factor: C::Factor,
    streams: Streams,
) -> Fill<T> {
    todo!()
}

/// The whole clique: `w_i w_j / d` per pair, the ground a neighbor like any other.
fn exact_clique<T: Real>(star: &[(u32, T)], fill: &mut Fill<T>) {
    todo!()
}
