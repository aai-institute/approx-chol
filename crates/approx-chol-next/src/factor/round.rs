//! The leading independent block eliminated at once: its exact columns, and the kept block's complement.

use std::sync::Arc;

use super::clique_tree::{Copies, Fill, Streams};
use super::config::{CliqueRule, InnerParallelism};
use super::sddm::{self, Sddm, Sheets};
use crate::sgdd::Balance;
use crate::storage::{CrossBlock, UpperAdjacency};
use crate::Real;

/// `M`'s cross block and the pivots `d_e`, applied as two products because the block's steps commute.
#[derive(Clone, Debug)]
pub(super) struct Round<T> {
    /// `M`'s own `C`, signs kept and shared with the part: a frustrated part's cover never doubles it.
    cross: Arc<CrossBlock<T>>,
    inv_pivot: Box<[T]>,
}

impl<T: Real> Round<T> {
    pub(super) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// `y_e = D_e⁻¹ b_e`, then `b_k −= Cᵀ y_e`.
    pub(super) fn forward(&self, x: &mut [T], inner: InnerParallelism) {
        todo!()
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`.
    pub(super) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        todo!()
    }
}

/// A round before its stored form is taken, so one pivot vector serves every complement and the solve.
pub(super) struct RoundBuilder<'a, T> {
    round: Round<T>,
    balance: &'a Balance<T>,
}

impl<'a, T: Real> RoundBuilder<'a, T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(super) fn new(cross: &Arc<CrossBlock<T>>, balance: &'a Balance<T>) -> Self {
        todo!()
    }

    pub(super) fn n_eliminated(&self) -> usize {
        self.round.n_eliminated()
    }

    /// Known before the complement is built: its surplus `s_k + |C|ᵀ D_e⁻¹ s_e` grounds iff the part's does.
    pub(super) fn complement_rank(&self) -> usize {
        let rows = Sheets::of(self.balance).count() * self.round.cross.n_kept();
        sddm::rank(rows, self.balance.grounding().is_grounded())
    }

    /// The kept block's Schur complement on the part's sheets: exact by kept row, sampled by star in parallel.
    pub(super) fn complement(&self, rule: CliqueRule, streams: Streams) -> Sddm<'static, T> {
        todo!()
    }

    pub(super) fn finish(self) -> Round<T> {
        self.round
    }
}

/// Each eliminated star's clique tree under its own stream, the ground a final neighbor; one chunk per task.
fn sampled_fill<T: Real, C: Copies>(
    builder: &RoundBuilder<'_, T>,
    split: C::Split,
    streams: Streams,
) -> Vec<Fill<T>> {
    todo!()
}

/// Rows of the cover's complement: `|c_ke| (|c_je| / d_e)` to `j`'s same sheet when signs agree, else its other; none cancel.
fn exact_rows<T: Real>(builder: &RoundBuilder<'_, T>) -> UpperAdjacency<T> {
    todo!()
}

/// `s_k + |C|ᵀ D_e⁻¹ s_e` per sheet, exact because no product of the cover's complement cancels.
fn exact_surplus<T: Real>(builder: &RoundBuilder<'_, T>) -> Vec<T> {
    todo!()
}
