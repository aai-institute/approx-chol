//! `P`: every block's vertices of the factored matrix in local order, coupled blocks by their segments, then the isolated ones.

use std::ops::{Range, RangeBounds};

use crate::{Index, MemStack};

/// The vertices of `P` a coupled block's [`Segment`] lists, then the isolated vertices: the lone ones, which the factor counts by their pivots, then the dead ones; persisted without what follows from those.
#[derive(Clone, Debug)]
pub(crate) struct Order {
    n: usize,
    /// The listed segments' vertices in segment order, then the isolated ones from `isolated`.
    listed: Listed,
    isolated: usize,
}

/// A coupled block's vertices: a run of consecutive ones storing no index, or listed in its [`Order`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Segment {
    Run {
        start: usize,
        len: usize,
    },
    Listed {
        #[cfg_attr(feature = "serde", serde(skip))]
        start: usize,
        len: usize,
    },
}

impl Segment {
    pub(crate) fn len(self) -> usize {
        match self {
            Segment::Run { len, .. } | Segment::Listed { len, .. } => len,
        }
    }

    /// Slots [`Slots::on_local`] takes from its stack before passing the rest on: room to gather, none for a run.
    pub(crate) fn gather_slots(self) -> usize {
        match self {
            Segment::Run { .. } => 0,
            Segment::Listed { len, .. } => len,
        }
    }
}

/// In `u32` when `n` lets every vertex fit.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) enum Listed {
    U32(Box<[u32]>),
    U64(Box<[u64]>),
}

/// One block's vertices of the factored matrix, in local order.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slots<'a> {
    Run { start: usize, len: usize },
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl From<Box<[u32]>> for Listed {
    fn from(listed: Box<[u32]>) -> Self {
        Self::U32(listed)
    }
}

impl From<Box<[u64]>> for Listed {
    fn from(listed: Box<[u64]>) -> Self {
        Self::U64(listed)
    }
}

impl Order {
    /// The ranges `coupled` of `vertices`, ascending and disjoint, as the coupled blocks' segments, consecutive ones as runs; then the isolated vertices from `isolated` on; every vertex renumbered from `numbering`'s start, whose length is `n`, and moved in place unless that narrows it to `u32`.
    pub(crate) fn pack<I: Index>(
        mut vertices: Vec<I>,
        coupled: &[Range<usize>],
        isolated: usize,
        numbering: Range<usize>,
    ) -> (Self, Vec<Segment>)
    where
        Listed: From<Box<[I]>>,
    {
        let local = |v: I| v.index() - numbering.start;
        let (mut segments, mut end) = (Vec::with_capacity(coupled.len()), 0);
        for range in coupled {
            let (block, len) = (&vertices[range.clone()], range.len());
            if block.windows(2).all(|w| w[1].index() == w[0].index() + 1) {
                let start = local(block[0]);
                segments.push(Segment::Run { start, len });
            } else {
                segments.push(Segment::Listed { start: end, len });
                end += len;
            }
        }
        let listed = (coupled.iter().zip(&segments))
            .filter(|(_, segment)| matches!(segment, Segment::Listed { .. }))
            .flat_map(|(range, _)| range.clone())
            .chain(isolated..vertices.len());
        let (len, n) = (end + vertices.len() - isolated, numbering.len());
        let listed = match size_of::<I>() > size_of::<u32>() && n as u64 <= 1 << 32 {
            true => {
                let mut narrow = Vec::with_capacity(len);
                narrow.extend(listed.map(|k| local(vertices[k]) as u32));
                Listed::U32(narrow.into_boxed_slice())
            }
            false => {
                for (to, k) in listed.enumerate() {
                    vertices[to] = I::from_index(local(vertices[k]));
                }
                vertices.truncate(len);
                Listed::from(vertices.into_boxed_slice())
            }
        };
        let order = Self {
            n,
            listed,
            isolated: end,
        };
        (order, segments)
    }

    /// From `listed` and the coupled blocks' `segments`, whose list offsets it rebuilds, with `n`; `None` unless every offset is in bounds and every vertex below `n`, allocating nothing `n` sizes.
    #[cfg(feature = "serde")]
    pub(crate) fn decoded<'s>(
        listed: Listed,
        segments: impl Iterator<Item = &'s mut Segment>,
    ) -> Option<Self> {
        let (mut end, mut covered, mut runs_end) = (0usize, 0usize, 0usize);
        for segment in segments {
            let len = match segment {
                Segment::Run { start, len } => {
                    runs_end = runs_end.max(start.checked_add(*len)?);
                    *len
                }
                Segment::Listed { start, len } => {
                    *start = end;
                    end = end.checked_add(*len)?;
                    *len
                }
            };
            covered = covered.checked_add(len)?;
        }
        let isolated = listed.slots(..).len().checked_sub(end)?;
        let n = covered.checked_add(isolated)?;
        let below = match &listed {
            Listed::U32(listed) => listed.iter().all(|&v| v.index() < n),
            Listed::U64(listed) => listed.iter().all(|&v| v < n as u64),
        };
        let order = Self {
            n,
            listed,
            isolated: end,
        };
        (runs_end <= n && below).then_some(order)
    }

    /// What persists of it: all else follows from the blocks' segments and the lone vertices' count.
    #[cfg(feature = "serde")]
    pub(crate) fn listed(&self) -> &Listed {
        &self.listed
    }

    /// The vertices in a block or dead.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// A coupled block's vertices, by its segment.
    pub(crate) fn slots(&self, segment: Segment) -> Slots<'_> {
        match segment {
            Segment::Run { start, len } => Slots::Run { start, len },
            Segment::Listed { start, len } => self.listed.slots(start..start + len),
        }
    }

    /// Lone vertex `k`'s block.
    pub(crate) fn lone(&self, k: usize) -> Slots<'_> {
        let start = self.isolated().vertex(k);
        Slots::Run { start, len: 1 }
    }

    /// Vertices no entry couples: the lone ones, each its own block, then the dead ones, in no block.
    pub(crate) fn isolated(&self) -> Slots<'_> {
        self.listed.slots(self.isolated..)
    }

    /// The first `lone` isolated vertices, and the dead ones after them.
    pub(crate) fn lone_and_dead(&self, lone: usize) -> (Slots<'_>, Slots<'_>) {
        let dead = self.isolated + lone;
        (
            self.listed.slots(self.isolated..dead),
            self.listed.slots(dead..),
        )
    }

    /// Whether no vertex is in two places, the coupled blocks' on `segments`; with every one below `n` and `n` places, each is in one.
    #[cfg(feature = "serde")]
    pub(crate) fn once(&self, mut segments: impl Iterator<Item = Segment>) -> bool {
        let mut seen = vec![false; self.n];
        let mut once = |v: usize| !std::mem::replace(&mut seen[v], true);
        segments.all(|segment| self.slots(segment).vertices().all(&mut once))
            && self.isolated().vertices().all(once)
    }
}

impl Listed {
    fn slots(&self, range: impl RangeBounds<usize>) -> Slots<'_> {
        let range = (range.start_bound().cloned(), range.end_bound().cloned());
        match self {
            Self::U32(listed) => Slots::U32(&listed[range]),
            Self::U64(listed) => Slots::U64(&listed[range]),
        }
    }
}

impl<'a> Slots<'a> {
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

    /// Runs `solve` on `Pᵀ x` and the rest of `stack`, then sets the block's vertices of `x` to the result.
    pub(crate) fn on_local<V: Copy>(
        self,
        x: &mut [V],
        stack: &mut MemStack,
        solve: impl FnOnce(&mut [V], &mut MemStack),
    ) {
        match self {
            Slots::Run { start, len } => solve(&mut x[start..start + len], stack),
            Slots::U32(listed) => gathered(listed, x, stack, solve),
            Slots::U64(listed) => gathered(listed, x, stack, solve),
        }
    }
}

fn gathered<V: Copy, I: Index>(
    listed: &[I],
    x: &mut [V],
    stack: &mut MemStack,
    solve: impl FnOnce(&mut [V], &mut MemStack),
) {
    let (mut local, rest) = stack.make_with(listed.len(), |k| x[listed[k].index()]);
    solve(&mut local, rest);
    (local.iter().zip(listed)).for_each(|(&v, &i)| x[i.index()] = v);
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
        let (order, segments) = Order::pack(vertices, &[0..2, 2..5, 5..7], 7, 0..9);
        assert_eq!(order.n(), 9);
        let coupled = segments.into_iter().map(|segment| order.slots(segment));
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
        let (order, segments) = Order::pack(vertices, &[1..3, 4..5], 5, 4..9);
        assert_eq!(order.n(), 5);
        let coupled = segments.into_iter().map(|segment| order.slots(segment));
        let blocks: Vec<_> = coupled.chain([order.lone(0)]).collect();
        assert!(!is_run(blocks[0]) && is_run(blocks[1]));
        let blocks: Vec<_> = blocks.into_iter().map(listed).collect();
        assert_eq!(blocks, [vec![0, 2], vec![1], vec![4]]);
        assert_eq!(listed(order.isolated()), [4, 3]);
    }
}
