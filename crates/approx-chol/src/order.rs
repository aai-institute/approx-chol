//! `P`: every block's vertices of the factored matrix in local order, one set per coupled block, then the isolated ones.

use std::ops::Range;

use inflight::parallel::{Claim, Members, Partition, Set};

use crate::{Index, MemStack};

/// The coupled blocks' vertices, one set each, then the isolated vertices as one more: the lone ones, which the factor counts by their pivots, then the dead ones; no vertex in two sets.
#[derive(Clone, Debug)]
pub(crate) struct Order {
    pub(crate) sets: Sets,
}

/// In `u32` when `n` lets every vertex fit.
#[derive(Clone, Debug)]
pub(crate) enum Sets {
    U32(Partition<u32>),
    U64(Partition<u64>),
}

/// A coupled block's vertices as persisted: a run of consecutive ones storing no index, or the next `len` of the order's list.
#[cfg(feature = "serde")]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum Segment {
    Run { start: usize, len: usize },
    Listed { len: usize },
}

#[cfg(feature = "serde")]
impl Segment {
    pub(crate) fn len(self) -> usize {
        match self {
            Segment::Run { len, .. } | Segment::Listed { len } => len,
        }
    }
}

/// The listed vertices as persisted.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) enum Listed<U, W> {
    U32(U),
    U64(W),
}

/// One block's vertices of the factored matrix, in local order.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slots<'a> {
    Run { start: usize, len: usize },
    U32(&'a [u32]),
    U64(&'a [u64]),
}

/// Why packing cannot fail: the ranges are disjoint, of distinct vertices renumbered below `n`.
const DISJOINT: &str = "disjoint ranges of distinct vertices below n";

impl From<Partition<u32>> for Sets {
    fn from(sets: Partition<u32>) -> Self {
        Self::U32(sets)
    }
}

impl From<Partition<u64>> for Sets {
    fn from(sets: Partition<u64>) -> Self {
        Self::U64(sets)
    }
}

impl Order {
    /// The ranges `coupled` of `vertices`, ascending and disjoint, as the coupled blocks' sets, consecutive ones as runs; then the isolated vertices from the last range's end on; every vertex renumbered from `numbering`'s start, whose length is `n`, and moved in place unless that narrows it to `u32`.
    pub(crate) fn pack<I: Index>(
        mut vertices: Vec<I>,
        coupled: impl ExactSizeIterator<Item = Range<usize>> + Clone,
        numbering: Range<usize>,
    ) -> Self
    where
        Sets: From<Partition<I>>,
    {
        let local = |v: I| v.index() - numbering.start;
        let mut sets = Vec::with_capacity(coupled.len() + 1);
        let mut isolated = 0;
        for range in coupled.clone() {
            isolated = range.end;
            let block = &vertices[range.clone()];
            sets.push(
                match block.windows(2).all(|w| w[1].index() == w[0].index() + 1) {
                    true => Set::Run {
                        start: local(block[0]),
                        len: range.len(),
                    },
                    false => Set::Listed { len: range.len() },
                },
            );
        }
        sets.push(Set::Listed {
            len: vertices.len() - isolated,
        });
        let listed = (coupled.zip(&sets))
            .filter(|(_, set)| matches!(set, Set::Listed { .. }))
            .flat_map(|(range, _)| range)
            .chain(isolated..vertices.len());
        let len = (sets.iter())
            .map(|set| match set {
                Set::Listed { len } => *len,
                Set::Run { .. } => 0,
            })
            .sum();
        let n = numbering.len();
        let sets = match size_of::<I>() > size_of::<u32>() && n as u64 <= 1 << 32 {
            true => {
                let mut narrow = Vec::with_capacity(len);
                narrow.extend(listed.map(|k| local(vertices[k]) as u32));
                Sets::U32(Partition::new(n, narrow.into_boxed_slice(), sets).expect(DISJOINT))
            }
            false => {
                for (to, k) in listed.enumerate() {
                    vertices[to] = I::from_index(local(vertices[k]));
                }
                vertices.truncate(len);
                Sets::from(Partition::new(n, vertices.into_boxed_slice(), sets).expect(DISJOINT))
            }
        };
        Self { sets }
    }

    /// From `listed` and the coupled blocks' `segments`, with `n` their vertices and the isolated ones, the rest of `listed`; `None` unless every vertex is below `n` and in one set.
    #[cfg(feature = "serde")]
    pub(crate) fn decoded(
        listed: Listed<Box<[u32]>, Box<[u64]>>,
        segments: &[Segment],
    ) -> Option<Self> {
        let (mut covered, mut in_list) = (0usize, 0usize);
        for &segment in segments {
            covered = covered.checked_add(segment.len())?;
            if let Segment::Listed { len } = segment {
                in_list = in_list.checked_add(len)?;
            }
        }
        let listed_len = match &listed {
            Listed::U32(listed) => listed.len(),
            Listed::U64(listed) => listed.len(),
        };
        let isolated = listed_len.checked_sub(in_list)?;
        let n = covered.checked_add(isolated)?;
        let sets = (segments.iter())
            .map(|&segment| match segment {
                Segment::Run { start, len } => Set::Run { start, len },
                Segment::Listed { len } => Set::Listed { len },
            })
            .chain([Set::Listed { len: isolated }]);
        let sets = match listed {
            Listed::U32(listed) => Sets::U32(Partition::new(n, listed, sets).ok()?),
            Listed::U64(listed) => Sets::U64(Partition::new(n, listed, sets).ok()?),
        };
        Some(Self { sets })
    }

    /// What persists of it: the listed vertices, and each coupled block's segment.
    #[cfg(feature = "serde")]
    pub(crate) fn persisted(&self) -> (Listed<&[u32], &[u64]>, Vec<Segment>) {
        let (listed, sets) = match &self.sets {
            Sets::U32(sets) => (Listed::U32(sets.listed()), sets.len()),
            Sets::U64(sets) => (Listed::U64(sets.listed()), sets.len()),
        };
        // The isolated vertices' set, the last, follows from the list's length.
        let segments = (0..sets - 1)
            .map(|k| match self.slots(k) {
                Slots::Run { start, len } => Segment::Run { start, len },
                listed => Segment::Listed { len: listed.len() },
            })
            .collect();
        (listed, segments)
    }

    /// The vertices in a block or dead.
    pub(crate) fn n(&self) -> usize {
        match &self.sets {
            Sets::U32(sets) => sets.n(),
            Sets::U64(sets) => sets.n(),
        }
    }

    /// Coupled block `k`'s vertices; the isolated ones past the last.
    pub(crate) fn slots(&self, k: usize) -> Slots<'_> {
        match &self.sets {
            Sets::U32(sets) => Slots::of(sets.members(k), Slots::U32),
            Sets::U64(sets) => Slots::of(sets.members(k), Slots::U64),
        }
    }

    /// [`on_claimed`] on coupled block `k`'s vertices of `x`.
    pub(crate) fn on_local<V: Copy>(
        &self,
        k: usize,
        x: &mut [V],
        stack: &mut MemStack,
        solve: impl FnOnce(&mut [V], &mut MemStack),
    ) {
        match &self.sets {
            Sets::U32(sets) => on_claimed(sets.claim(x, k), stack, solve),
            Sets::U64(sets) => on_claimed(sets.claim(x, k), stack, solve),
        }
    }

    /// Lone vertex `k`'s block.
    pub(crate) fn lone(&self, k: usize) -> Slots<'_> {
        let start = self.isolated().vertex(k);
        Slots::Run { start, len: 1 }
    }

    /// Vertices no entry couples: the lone ones, each its own block, then the dead ones, in no block.
    pub(crate) fn isolated(&self) -> Slots<'_> {
        let last = match &self.sets {
            Sets::U32(sets) => sets.len(),
            Sets::U64(sets) => sets.len(),
        };
        self.slots(last - 1)
    }

    /// The first `lone` isolated vertices, and the dead ones after them.
    pub(crate) fn lone_and_dead(&self, lone: usize) -> (Slots<'_>, Slots<'_>) {
        match self.isolated() {
            Slots::Run { start, len } => (
                Slots::Run { start, len: lone },
                Slots::Run {
                    start: start + lone,
                    len: len - lone,
                },
            ),
            Slots::U32(isolated) => (Slots::U32(&isolated[..lone]), Slots::U32(&isolated[lone..])),
            Slots::U64(isolated) => (Slots::U64(&isolated[..lone]), Slots::U64(&isolated[lone..])),
        }
    }
}

impl<'a> Slots<'a> {
    fn of<I>(members: Members<'a, I>, listed: impl FnOnce(&'a [I]) -> Self) -> Self {
        match members {
            Members::Run { start, len } => Slots::Run { start, len },
            Members::Listed(members) => listed(members),
        }
    }

    /// Slots [`on_local`](Self::on_local) takes from its stack before passing the rest on: room to gather, none for a run.
    pub(crate) fn gather_slots(self) -> usize {
        match self {
            Slots::Run { .. } => 0,
            listed => listed.len(),
        }
    }

    pub(crate) fn len(self) -> usize {
        match self {
            Slots::Run { len, .. } => len,
            Slots::U32(listed) => listed.len(),
            Slots::U64(listed) => listed.len(),
        }
    }

    /// Local vertex `k` in the factored matrix's numbering.
    pub(crate) fn vertex(self, k: usize) -> usize {
        match self {
            Slots::Run { start, .. } => start + k,
            Slots::U32(listed) => listed[k].index(),
            Slots::U64(listed) => listed[k].index(),
        }
    }

    /// Each local vertex in the factored matrix's numbering, in local order.
    pub(crate) fn vertices(self) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
        (0..self.len()).map(move |k| self.vertex(k))
    }
}

/// Runs `solve` on a block's claim of `x`, `Pᵀ x` gathered unless its vertices are consecutive, and the rest of `stack`, then sets its vertices of `x` to the result.
pub(crate) fn on_claimed<V: Copy, P: inflight::parallel::ListedIndex>(
    claim: Claim<'_, P, V>,
    stack: &mut MemStack,
    solve: impl FnOnce(&mut [V], &mut MemStack),
) {
    match claim {
        Claim::Run(x) => solve(x, stack),
        Claim::Scattered(mut x) => {
            let (mut local, rest) = stack.make_with(x.len(), |k| x.get(k));
            solve(&mut local, rest);
            local.iter().enumerate().for_each(|(k, &v)| x.set(k, v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(slots: Slots<'_>) -> Vec<usize> {
        slots.vertices().collect()
    }

    fn is_run(slots: Slots<'_>) -> bool {
        matches!(slots, Slots::Run { .. })
    }

    /// Coupled ranges keep their local order, a consecutive one as a run; the isolated vertices follow.
    #[test]
    fn packing_lists_coupled_blocks_ahead_of_isolated_vertices() {
        let vertices = vec![0u32, 2, 4, 5, 3, 6, 7, 1, 8];
        let order = Order::pack(vertices, [0..2, 2..5, 5..7].into_iter(), 0..9);
        assert_eq!(order.n(), 9);
        let coupled = (0..3).map(|k| order.slots(k));
        let blocks: Vec<_> = coupled.chain([order.lone(0)]).collect();
        assert!(!is_run(blocks[0]) && !is_run(blocks[1]) && is_run(blocks[2]) && is_run(blocks[3]));
        let blocks: Vec<_> = blocks.into_iter().map(listed).collect();
        assert_eq!(blocks, [vec![0, 2], vec![4, 5, 3], vec![6, 7], vec![1]]);
        assert_eq!(listed(order.isolated()), [1, 8]);
    }

    /// Tails renumbered from the kept side's start, a consecutive one as a run.
    #[test]
    fn packing_renumbers_onto_a_kept_side() {
        let vertices = vec![0u32, 4, 6, 1, 5, 8, 7];
        let order = Order::pack(vertices, [1..3, 4..5].into_iter(), 4..9);
        assert_eq!(order.n(), 5);
        let coupled = (0..2).map(|k| order.slots(k));
        let blocks: Vec<_> = coupled.chain([order.lone(0)]).collect();
        assert!(!is_run(blocks[0]) && is_run(blocks[1]));
        let blocks: Vec<_> = blocks.into_iter().map(listed).collect();
        assert_eq!(blocks, [vec![0, 2], vec![1], vec![4]]);
        assert_eq!(listed(order.isolated()), [4, 3]);
    }
}
