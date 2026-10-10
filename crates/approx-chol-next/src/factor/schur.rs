//! Block elimination of a bipartite part's larger, independent side, and the Schur complement it leaves.

use std::sync::Arc;

use super::clique_tree::{Copies, Fill, Streams};
use super::config::{CliqueRule, InnerParallelism};
use super::sddm::{Sddm, Sheets};
use crate::sgdd::Balance;
use crate::storage::{CrossBlock, UpperAdjacency};
use crate::Real;

/// `M`'s cross block and the pivots `d_e`, applied as two products because the block's steps commute.
#[derive(Clone, Debug)]
pub(super) struct BlockElimination<T> {
    /// `M`'s own `C`, signs kept and shared with the part: a frustrated part's cover never doubles it.
    cross: Arc<CrossBlock<T>>,
    inv_pivot: Box<[T]>,
}

impl<T: Real> BlockElimination<T> {
    pub(super) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// `y_e = D_e⁻¹ b_e`, then `b_k −= Cᵀ y_e`, gathered by kept row through `Cᵀ` under either mode so both give the same bits.
    pub(super) fn forward(&self, x: &mut [T], inner: InnerParallelism) {
        todo!()
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`.
    pub(super) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        todo!()
    }
}

/// `S = D_k − Cᵀ D_e⁻¹ C` before the elimination's stored form is taken, so one pivot vector serves every complement and the solve.
pub(super) struct SchurComplement<'a, T> {
    elimination: BlockElimination<T>,
    balance: &'a Balance<T>,
}

impl<'a, T: Real> SchurComplement<'a, T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(super) fn new(cross: &Arc<CrossBlock<T>>, balance: &'a Balance<T>) -> Self {
        todo!()
    }

    pub(super) fn n_eliminated(&self) -> usize {
        self.elimination.n_eliminated()
    }

    /// The Cholesky's dimension, a cover's two sheets counted; known before `S` is built, so routing never builds it twice.
    pub(super) fn dimension(&self) -> usize {
        Sheets::of(self.balance).count() * self.elimination.cross.n_kept()
    }

    /// `S` on the part's sheets: exact by kept row, sampled by star in parallel.
    pub(super) fn build(&self, rule: CliqueRule, streams: Streams) -> Sddm<'static, T> {
        todo!()
    }

    pub(super) fn finish(self) -> BlockElimination<T> {
        self.elimination
    }
}

/// Each eliminated star's clique tree under its own stream, the ground a final neighbor; chunks in star order, so merged duplicates sum alike at any thread count.
fn sampled_fill<T: Real, C: Copies>(
    schur: &SchurComplement<'_, T>,
    split: C::Split,
    streams: Streams,
) -> Vec<Fill<T>> {
    todo!()
}

/// Rows of the cover's complement: `|c_ke||c_je|/d_e` to `j`'s same sheet when signs agree, else its other, as the smaller times the larger's share, which underflows only below the floor.
fn exact_rows<T: Real>(schur: &SchurComplement<'_, T>) -> UpperAdjacency<T> {
    todo!()
}

/// `s_k + |C|ᵀ D_e⁻¹ s_e` per sheet, exact as no product cancels; each `|c_ek| s_e/d_e` multiplied as the complement's are.
fn exact_surplus<T: Real>(schur: &SchurComplement<'_, T>) -> Vec<T> {
    todo!()
}
