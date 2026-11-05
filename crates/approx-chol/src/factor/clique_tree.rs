//! A pivot's star prepared by the engine, and the clique sample that draws its fill (GKS 2023, Algorithm 6; AC is its `k = l = 1`).

mod copies;
mod star;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::wdd::Sign;
use crate::storage::{narrow, widen, Index, Real};
pub(super) use copies::{Copies, Multi, Single};
pub(super) use star::{Column, Star};

/// What a rule draws: two positions of the star, the weight summed over the copies that drew them.
#[derive(Clone, Copy, Debug)]
pub(super) struct FillPair<T, C> {
    pub(super) first: usize,
    pub(super) second: usize,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// A fill edge as the engine emits it, `lo < hi`, its sign attached.
#[derive(Clone, Copy, Debug)]
pub(super) struct FillEdge<T, I, C, S> {
    pub(super) lo: I,
    pub(super) hi: I,
    pub(super) copies: C,
    pub(super) sign: S,
    pub(super) weight: T,
}

/// Fill edges as an elimination emits them: one per entry and partner.
pub(super) type Fill<T, I, C, S> = Vec<FillEdge<T, I, C, S>>;

/// A clique sample: a prepared star in, pairs of its positions out; it reads magnitudes only, and the engine attaches ends and signs.
pub(super) trait FillRule<T, I, C: Copies, S> {
    /// Emits each pair once, with a weight of at least `MIN_POSITIVE`, dropping none.
    fn fill(
        &mut self,
        star: &Star<T, I, C, S>,
        rng: &mut Xoshiro256PlusPlus,
        emit: impl FnMut(FillPair<T, C>),
    );
}

/// The sampled clique tree: each neighbor but the heaviest joins one drawn past it, per copy.
#[derive(Default)]
pub(super) struct CliqueTree {
    /// Scratch [`Copies::draw`] reuses across entries; one copy never touches it.
    tally: Vec<(usize, u32)>,
}

impl<T: Real, I: Index, C: Copies, S: Sign> FillRule<T, I, C, S> for CliqueTree {
    /// `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn fill(
        &mut self,
        star: &Star<T, I, C, S>,
        rng: &mut Xoshiro256PlusPlus,
        mut emit: impl FnMut(FillPair<T, C>),
    ) {
        let entries = star.entries();
        let Some((_, rest)) = entries.split_last() else {
            return;
        };
        let d = star.total();
        for (i, entry) in rest.iter().enumerate() {
            let total: T = narrow(widen(entry.weight) * (star.past(i) / d));
            // Clamped, never dropped: the clique tree keeps the star's neighbors connected, and every weight's inverse finite.
            let weight = (entry.copies).per_copy(total).max(T::min_positive_value());
            let draw = || star.draw_after(rng, i);
            for (at, copies, weight) in (entry.copies).draw(&mut self.tally, weight, draw) {
                emit(FillPair {
                    first: i,
                    second: at,
                    copies,
                    weight,
                });
            }
        }
    }
}
