use super::adjacency::AdjListGraph;
use crate::types::{count_as_scalar, Real};

/// AC is AC2 at one copy per edge, so the whole difference between them is this
/// trait. `Single` is a ZST, so an AC edge carries no count at all.
pub(crate) trait EdgeCount: Clone + Copy {
    /// `Single` cannot name a `k`, which makes an AC factorization over split
    /// multi-edges a type error rather than a mistake to avoid.
    type Split: Copy;

    /// Known statically, which lets the single-copy path sort on weights instead of
    /// quotients that are all division by one.
    const SINGLE_COPY: bool;

    fn one() -> Self;
    fn get(&self) -> u32;

    /// The identity for `Single`, so the AC path performs no division rather than
    /// dividing by one.
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// `Single` keeps one, so its discard count is the duplicates the merge
    /// collapsed. The cap is the split itself, which is why `Single` has none to name.
    fn cap(copies: u32, limit: Self::Split) -> (Self, u32);

    /// The degree-bucket scale and the per-neighbor sample count are one number because
    /// they are one return value; the merge cap is [`Self::Split`] instead.
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: Self::Split) -> u32;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// This edge's virtual copy count, only ever lowered from the [`SplitFactor`] by the
/// merge cap.
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    /// Only a test needs this: elimination makes a `Multi` from the validated split
    /// or from [`EdgeCount::cap`], never from a bare number.
    #[cfg(test)]
    pub(crate) fn new(count: u32) -> Self {
        Self(count)
    }
}

impl From<SplitFactor> for Multi {
    #[inline]
    fn from(split: SplitFactor) -> Self {
        Self(split.get())
    }
}

/// Distinct from the per-edge count [`Multi`] carries. Only the factors AC2 is
/// defined for exist, so `1/k` is never infinite and no split is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitFactor(u32);

impl SplitFactor {
    /// `None` when `k` selects standard AC instead.
    pub(crate) fn new(k: u32) -> Option<Self> {
        (k >= 2).then_some(Self(k))
    }

    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl EdgeCount for Single {
    type Split = ();
    const SINGLE_COPY: bool = true;

    #[inline]
    fn one() -> Self {
        Self
    }
    #[inline]
    fn get(&self) -> u32 {
        1
    }
    #[inline]
    fn per_copy<T: Real>(&self, total: T) -> T {
        total
    }
    #[inline]
    fn cap(copies: u32, _limit: ()) -> (Self, u32) {
        (Self, copies - 1)
    }
    /// A slim edge has nowhere to put a multiplicity.
    #[inline]
    fn split_edges<T: Real>(_graph: &mut AdjListGraph<Self, T>, _split: ()) -> u32 {
        1
    }
}

impl EdgeCount for Multi {
    type Split = SplitFactor;
    const SINGLE_COPY: bool = false;

    #[inline]
    fn one() -> Self {
        Self(1)
    }
    #[inline]
    fn get(&self) -> u32 {
        self.0
    }
    #[inline]
    fn per_copy<T: Real>(&self, total: T) -> T {
        total / count_as_scalar::<T, _>(self.0)
    }
    #[inline]
    fn cap(copies: u32, limit: SplitFactor) -> (Self, u32) {
        let kept = copies.min(limit.get());
        (Self(kept), copies - kept)
    }
    #[inline]
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: SplitFactor) -> u32 {
        graph.mark_split_edges(split);
        split.get()
    }
}
#[cfg(test)]
mod tests {
    use super::super::adjacency::{Edge, MultiEdgeGraph};
    use super::*;

    /// A cap that drifted from [`MultiEdgeGraph::mark_split_edges`] would bound every
    /// star at a multiplicity the graph does not carry, unnoticed elsewhere.
    #[test]
    fn the_reported_cap_is_the_count_written_on_the_edges() {
        let k = SplitFactor::new(3).expect("3 splits");
        let mut graph = MultiEdgeGraph::<f64>::from_adjacency(vec![
            vec![Edge::new(1.0, 1, 0)],
            vec![Edge::new(1.0, 0, 0)],
        ]);

        let cap = Multi::split_edges(&mut graph, k);

        let mut neighbors = Vec::new();
        graph.live_neighbors(0, &mut neighbors);
        assert_eq!(cap, k.get(), "the cap is the configured split");
        assert_eq!(
            neighbors[0].count.get(),
            cap,
            "the cap is the count the edges carry"
        );
    }
}
