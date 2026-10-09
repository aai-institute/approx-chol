//! The leading independent block eliminated at once: its exact columns, and the kept block's complement.

use std::sync::Arc;

use crate::config::CliqueRule;
use crate::sampling::{CdfSampler, Streams};
use crate::sgdd::{Ground, Sddm, Sheets};
use crate::storage::CrossBlock;
use crate::Real;

/// Whether a solve may split its own work across threads; an outer parallel loop says no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerParallelism {
    /// Large rounds split their rows.
    Allow,
    /// Everything runs on the calling thread.
    Forbid,
}

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

/// A round before its stored form is taken, so one pivot vector serves the complement and the solve.
pub(crate) struct RoundBuilder<'a, T> {
    round: Round<T>,
    ground: &'a Ground<T>,
}

impl<'a, T: Real> RoundBuilder<'a, T> {
    /// Pivots are `Σ|c_kj| + s_k`, the same on both sheets of a cover.
    pub(crate) fn new(cross: &Arc<CrossBlock<T>>, ground: &'a Ground<T>) -> Self {
        todo!()
    }

    pub(crate) fn n_kept(&self) -> usize {
        self.round.cross.n_kept()
    }

    /// The kept block's complement; the stars share no edge, so they run in parallel.
    pub(crate) fn schur(
        &self,
        sheets: Sheets,
        rule: CliqueRule,
        streams: Streams,
    ) -> Sddm<'static, T> {
        todo!()
    }

    pub(crate) fn finish(self) -> Round<T> {
        self.round
    }
}

/// Fill edges `(lo, hi, weight)`, `lo < hi`.
type Fill<T> = Vec<(u32, u32, T)>;

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
