use super::adjacency::AdjListGraph;
use crate::types::{count_as_scalar, Real};

/// AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(crate) trait EdgeCount: Clone + Copy {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Split: Copy;

    /// Static, so the single-copy path sorts on weights not division by one.
    const SINGLE_COPY: bool;

    fn one() -> Self;
    fn get(&self) -> u32;

    /// Identity for `Single`, so the AC path divides not at all.
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// `Single` keeps one, so its discard count is what the merge collapsed.
    fn cap(copies: u32, limit: Self::Split) -> (Self, u32);

    /// Bucket scale and per-neighbor sample count are one number; the cap is `Split`.
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: Self::Split) -> u32;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// Virtual copy count, only ever lowered from [`SplitFactor`] by the merge cap.
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    /// Only a test needs this; elimination never makes a `Multi` from a bare number.
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

/// Only factors AC2 is defined for exist, so `1/k` is finite and no split is a no-op.
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

    /// A cap drifting from `mark_split_edges` bounds every star at a multiplicity
    /// the graph does not carry.
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
