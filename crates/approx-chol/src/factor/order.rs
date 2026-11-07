//! `P`: every block's vertices of the factored matrix in local order, coupled blocks first, then the lone and the dead ones.

use crate::sgdd::Selection;
use crate::{Index, MemStack};

/// `P` as one segment per coupled block, a run of consecutive vertices storing no index, then the lone and the dead vertices as lists.
#[derive(Clone, Debug)]
pub(super) struct Order {
    n: usize,
    segments: Box<[Segment]>,
    /// The coupled blocks' vertices outside runs, then the lone ones from `lone`, then the dead ones from `dead`.
    listed: Listed,
    lone: usize,
    dead: usize,
}

#[derive(Clone, Copy, Debug)]
enum Segment {
    Run { start: usize, len: usize },
    Listed { start: usize, len: usize },
}

/// In `u32` when `n` lets every vertex fit.
#[derive(Clone, Debug)]
enum Listed {
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

/// An index type whose vertex lists [`Slots`] borrows as stored.
pub(crate) trait Width: Index {
    fn slots(listed: &[Self]) -> Slots<'_>;
}

impl Width for u32 {
    fn slots(listed: &[u32]) -> Slots<'_> {
        Slots::U32(listed)
    }
}

impl Width for u64 {
    fn slots(listed: &[u64]) -> Slots<'_> {
        Slots::U64(listed)
    }
}

impl Order {
    /// Each coupled block's vertices in turn, then the lone and the dead ones; every vertex is below their total count.
    pub(super) fn new(
        coupled: &[Slots<'_>],
        lone: impl ExactSizeIterator<Item = usize>,
        dead: impl ExactSizeIterator<Item = usize>,
    ) -> Self {
        let mut end = 0;
        let segments: Box<[Segment]> = (coupled.iter())
            .map(|&slots| match slots {
                Slots::Run { start, len } => Segment::Run { start, len },
                listed => {
                    let start = end;
                    end += listed.len();
                    Segment::Listed {
                        start,
                        len: listed.len(),
                    }
                }
            })
            .collect();
        let (lone_len, dead_len) = (lone.len(), dead.len());
        let n = coupled.iter().map(|s| s.len()).sum::<usize>() + lone_len + dead_len;
        let len = end + lone_len + dead_len;
        let listed = match n as u64 <= 1 << 32 {
            true => Listed::U32(collect(len, coupled, lone, dead)),
            false => Listed::U64(collect(len, coupled, lone, dead)),
        };
        Self {
            n,
            segments,
            listed,
            lone: end,
            dead: end + lone_len,
        }
    }

    /// From decoded parts: `None` unless they cover `0..n` once each.
    #[cfg(feature = "serde")]
    pub(super) fn checked(
        n: usize,
        coupled: &[Slots<'_>],
        lone: &[usize],
        dead: &[usize],
    ) -> Option<Self> {
        let below = |&v: &usize| v < n;
        let within = coupled.iter().all(|s| s.within(n)) && lone.iter().chain(dead).all(below);
        let order =
            within.then(|| Self::new(coupled, lone.iter().copied(), dead.iter().copied()))?;
        (order.n == n && order.once()).then_some(order)
    }

    /// The vertices in a block or dead.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Blocks: the coupled ones, then one per lone vertex.
    pub(super) fn len(&self) -> usize {
        self.segments.len() + self.dead - self.lone
    }

    /// Block `k`'s vertices.
    pub(super) fn block(&self, k: usize) -> Slots<'_> {
        match self.segments.get(k) {
            Some(&Segment::Run { start, len }) => Slots::Run { start, len },
            Some(&Segment::Listed { start, len }) => self.listed.slots(start, len),
            None => Slots::Run {
                start: self.listed.vertex(self.lone + k - self.segments.len()),
                len: 1,
            },
        }
    }

    /// Each block's vertices, in block order.
    pub(super) fn blocks(&self) -> impl ExactSizeIterator<Item = Slots<'_>> + '_ {
        (0..self.len()).map(|k| self.block(k))
    }

    /// Vertices in no block, which solve to `0`.
    pub(super) fn dead(&self) -> Slots<'_> {
        self.listed.slots(self.dead, self.listed.len() - self.dead)
    }

    /// Whether no vertex is in two places; with every one below `n` and `n` places, each is in one.
    #[cfg(feature = "serde")]
    fn once(&self) -> bool {
        let mut seen = vec![false; self.n];
        let mut once = |v: usize| !std::mem::replace(&mut seen[v], true);
        self.blocks().all(|s| s.vertices().all(&mut once)) && self.dead().vertices().all(once)
    }
}

fn collect<W: Index>(
    len: usize,
    coupled: &[Slots<'_>],
    lone: impl Iterator<Item = usize>,
    dead: impl Iterator<Item = usize>,
) -> Box<[W]> {
    let mut all = Vec::with_capacity(len);
    for &slots in coupled {
        match slots {
            Slots::Run { .. } => {}
            Slots::U32(listed) => all.extend(listed.iter().map(|v| W::from_index(v.index()))),
            Slots::U64(listed) => all.extend(listed.iter().map(|v| W::from_index(v.index()))),
        }
    }
    all.extend(lone.chain(dead).map(W::from_index));
    all.into_boxed_slice()
}

impl Listed {
    fn len(&self) -> usize {
        match self {
            Self::U32(listed) => listed.len(),
            Self::U64(listed) => listed.len(),
        }
    }

    fn vertex(&self, k: usize) -> usize {
        match self {
            Self::U32(listed) => listed[k].index(),
            Self::U64(listed) => listed[k].index(),
        }
    }

    fn slots(&self, start: usize, len: usize) -> Slots<'_> {
        match self {
            Self::U32(listed) => Slots::U32(&listed[start..start + len]),
            Self::U64(listed) => Slots::U64(&listed[start..start + len]),
        }
    }
}

impl<'a> Slots<'a> {
    pub(super) fn of<I: Width>(selection: &'a Selection<I>) -> Self {
        match selection {
            Selection::Range(range) => Slots::Run {
                start: range.start,
                len: range.len(),
            },
            Selection::Listed(listed) => I::slots(listed),
        }
    }

    pub(super) fn len(self) -> usize {
        match self {
            Slots::Run { len, .. } => len,
            Slots::U32(listed) => listed.len(),
            Slots::U64(listed) => listed.len(),
        }
    }

    /// Local vertex `k` in the factored matrix's numbering.
    pub(super) fn vertex(self, k: usize) -> usize {
        match self {
            Slots::Run { start, .. } => start + k,
            Slots::U32(listed) => listed[k].index(),
            Slots::U64(listed) => listed[k].index(),
        }
    }

    /// Each local vertex in the factored matrix's numbering, in local order.
    pub(super) fn vertices(self) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
        (0..self.len()).map(move |k| self.vertex(k))
    }

    /// Slots [`on_local`](Self::on_local) takes from its stack before passing the rest on: room to gather, none for a run.
    pub(super) fn gather_slots(self) -> usize {
        match self {
            Slots::Run { .. } => 0,
            listed => listed.len(),
        }
    }

    /// Runs `solve` on `Pᵀ x` and the rest of `stack`, then sets the block's vertices of `x` to the result.
    pub(super) fn on_local<V: Copy>(
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

    /// Whether every vertex is below `n`, a run's end included, so none overflows or narrows.
    #[cfg(feature = "serde")]
    fn within(self, n: usize) -> bool {
        match self {
            Slots::Run { start, len } => start.checked_add(len).is_some_and(|end| end <= n),
            Slots::U32(listed) => listed.iter().all(|&v| v.index() < n),
            Slots::U64(listed) => listed.iter().all(|&v| v < n as u64),
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
