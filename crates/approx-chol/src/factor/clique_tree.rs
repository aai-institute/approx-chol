//! A pivot's star prepared by the engine, and the clique sample that draws its fill (GKS 2023, Algorithm 6; AC is its `k = l = 1`).

mod cdf;
mod copies;
mod star;

use rand::rngs::Xoshiro256PlusPlus;

use crate::storage::{narrow, widen, Index, Real};
use cdf::draw_after;
pub(super) use copies::{Copies, Multi, Single};
pub(super) use star::{Column, Star};

/// A sampled edge, `lo < hi`, its weight summed over its copies.
#[derive(Clone, Copy, Debug)]
pub(super) struct FillEdge<T, I, C> {
    pub(super) lo: I,
    pub(super) hi: I,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// Fill edges as an elimination emits them: one per entry and partner.
pub(super) type Fill<T, I, C> = Vec<FillEdge<T, I, C>>;

/// A clique sample: a prepared star in, fill edges out; the column is the engine's, read off the same star.
pub(super) trait FillRule<T, I, C: Copies> {
    /// Emits each fill edge once, with a weight of at least `MIN_POSITIVE`, dropping none.
    fn fill(
        &mut self,
        star: &Star<T, I, C>,
        rng: &mut Xoshiro256PlusPlus,
        emit: impl FnMut(FillEdge<T, I, C>),
    );
}

/// The sampled clique tree: each neighbor but the heaviest joins one drawn past it, per copy.
#[derive(Default)]
pub(super) struct CliqueTree {
    /// Scratch [`Copies::draw`] reuses across entries; one copy never touches it.
    tally: Vec<(usize, u32)>,
}

impl<T: Real, I: Index, C: Copies> FillRule<T, I, C> for CliqueTree {
    /// `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn fill(
        &mut self,
        star: &Star<T, I, C>,
        rng: &mut Xoshiro256PlusPlus,
        mut emit: impl FnMut(FillEdge<T, I, C>),
    ) {
        let (entries, cumsum) = (star.entries(), star.cumsum());
        let Some((_, rest)) = entries.split_last() else {
            return;
        };
        let d = star.total();
        for (i, entry) in rest.iter().enumerate() {
            let total: T = narrow(widen(entry.weight) * (star.past(i) / d));
            // Clamped, never dropped: the clique tree keeps the star's neighbors connected, and every weight's inverse finite.
            let weight = (entry.copies).per_copy(total).max(T::min_positive_value());
            let draw = || draw_after(cumsum, rng, i);
            for (at, copies, weight) in (entry.copies).draw(&mut self.tally, weight, draw) {
                let other = entries[at].neighbor;
                emit(FillEdge {
                    lo: entry.neighbor.min(other),
                    hi: entry.neighbor.max(other),
                    copies,
                    weight,
                });
            }
        }
    }
}
