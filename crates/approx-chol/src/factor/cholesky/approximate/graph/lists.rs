//! Many growable lists in one slab: a buffer per list, growing with fill, left macOS malloc holding 3.6 GB for 0.7 GB in use (dense worker-firm, 1M).

use std::num::NonZeroUsize;

use crate::factor::cholesky::approximate::REUSED_UP_TO;
use crate::Index;

/// A list's `len` elements from `start`, in room for `cap`, which is run `run` of the slab; a roomless list has none.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: usize,
    run: Option<NonZeroUsize>,
    len: u32,
    cap: u32,
}

// Every list touches its span, so a wider one costs a word per vertex: +7 MiB counted peak at 1M.
const _: () = assert!(size_of::<Span>() == 2 * size_of::<usize>() + 8);

/// Run `r`, one up so that `Option` stores its absence in the zero.
fn run(r: usize) -> Option<NonZeroUsize> {
    NonZeroUsize::new(r + 1)
}

/// At 0, as `get` slices from `start` and compaction may truncate the slab below any other.
const EMPTY: Span = Span {
    start: 0,
    run: None,
    len: 0,
    cap: 0,
};

/// Lists `0..n`, each a run of slab slots; a full list moves whole to the slab's end, so an element's position changes only by its own list's removals.
pub(super) struct Lists<E, I> {
    slab: Vec<E>,
    spans: Vec<Span>,
    /// The slab's runs in order, each with the list that took it and its room; a run is live while that list's span names it.
    runs: Vec<(I, u32)>,
    dead: usize,
}

/// A release compacts once more than `1/RELEASE_DEAD` of the slab is dead: at ¼ or ⅛, dense worker-firm 1M built 4% and 11% slower.
const RELEASE_DEAD: usize = 2;

/// Growth past capacity compacts first if at least `1/GROWTH_DEAD` of the slab is dead, else reserves.
const GROWTH_DEAD: usize = 4;

/// The slab grows, and shrinks to, `1/HEADROOM` past its length: doubling it measured +70% counted peak on grids.
const HEADROOM: usize = 4;

impl<E: Copy, I: Index> Lists<E, I> {
    pub(super) fn new() -> Self {
        Self {
            slab: Vec::new(),
            spans: Vec::new(),
            runs: Vec::new(),
            dead: 0,
        }
    }

    /// Empty lists, list `l` with room for `room[l]`; `unused` fills slots no list has written.
    pub(super) fn load(&mut self, room: &[u32], unused: E) {
        self.spans.clear();
        self.runs.clear();
        self.spans.reserve_exact(room.len());
        self.runs.reserve_exact(room.len());
        let mut end = 0;
        for (l, &cap) in room.iter().enumerate() {
            if cap == 0 {
                self.spans.push(EMPTY);
                continue;
            }
            self.spans.push(Span {
                start: end,
                run: run(self.runs.len()),
                len: 0,
                cap,
            });
            self.runs.push((I::from_index(l), cap));
            end += cap as usize;
        }
        self.slab.clear();
        self.slab.reserve_exact(end);
        self.slab.resize(end, unused);
        self.dead = 0;
    }

    pub(super) fn n(&self) -> usize {
        self.spans.len()
    }

    pub(super) fn len(&self, l: usize) -> u32 {
        self.spans[l].len
    }

    pub(super) fn get(&self, l: usize) -> &[E] {
        let Span { start, len, .. } = self.spans[l];
        &self.slab[start..start + len as usize]
    }

    pub(super) fn get_mut(&mut self, l: usize) -> &mut [E] {
        let Span { start, len, .. } = self.spans[l];
        &mut self.slab[start..start + len as usize]
    }

    pub(super) fn push(&mut self, l: usize, e: E) {
        let span = &mut self.spans[l];
        if span.len == span.cap {
            return self.grow(l, e);
        }
        self.slab[span.start + span.len as usize] = e;
        span.len += 1;
    }

    pub(super) fn pop(&mut self, l: usize) -> Option<E> {
        let span = &mut self.spans[l];
        span.len = span.len.checked_sub(1)?;
        Some(self.slab[span.start + span.len as usize])
    }

    /// As [`Vec::swap_remove`], but yields the element moved into `at`, if one was.
    pub(super) fn swap_remove(&mut self, l: usize, at: usize) -> Option<E> {
        let span = &mut self.spans[l];
        let len = span.len as usize;
        assert!(at < len, "swap_remove at {at} of a list of {len}");
        span.len -= 1;
        let (at, last) = (span.start + at, span.start + len - 1);
        (at != last).then(|| {
            let moved = self.slab[last];
            self.slab[at] = moved;
            moved
        })
    }

    /// Frees `l`'s room, emptying it; past [`RELEASE_DEAD`], the live runs move down and the slab shrinks.
    pub(super) fn release(&mut self, l: usize) {
        self.dead += self.spans[l].cap as usize;
        self.spans[l] = EMPTY;
        // A block that keeps its buffers keeps its slab whole too; a larger one's late fill lands on a shrunken slab.
        if RELEASE_DEAD * self.dead > self.slab.len() && self.n() > REUSED_UP_TO {
            self.compact();
            self.slab
                .shrink_to(self.slab.len() + self.slab.len() / HEADROOM);
            self.runs
                .shrink_to(self.runs.len() + self.runs.len() / HEADROOM);
        }
    }

    /// Pushes `e` onto full `l` in twice the room, as `Vec` grows; `e` also fills the new room past it.
    #[cold]
    fn grow(&mut self, l: usize, e: E) {
        let cap = self.spans[l].cap;
        let room = cap.saturating_mul(2).max(4);
        assert!(room > cap, "a list exceeds u32 slots");
        let past = self.slab.len() + room as usize > self.slab.capacity();
        if past && GROWTH_DEAD * self.dead >= self.slab.len() {
            self.compact();
        }
        let Span {
            start,
            run: own,
            len,
            ..
        } = self.spans[l];
        let at = self.slab.len();
        // The slab's last run grows where it is: no copy, and no dead run left behind.
        let last = own.is_some_and(|r| r.get() == self.runs.len());
        let more = if last { room - cap } else { room } as usize;
        if self.slab.capacity() - at < more {
            self.slab.reserve_exact((at / HEADROOM).max(more));
        }
        if last {
            let r = self.runs.len() - 1;
            self.runs[r].1 = room;
            self.spans[l].cap = room;
        } else {
            self.slab.extend_from_within(start..start + len as usize);
            self.dead += cap as usize;
            self.spans[l] = Span {
                start: at,
                run: run(self.runs.len()),
                len,
                cap: room,
            };
            self.runs.push((I::from_index(l), room));
        }
        let span = &mut self.spans[l];
        self.slab.resize(span.start + room as usize, e);
        span.len += 1;
    }

    /// Slides the live runs down in slab order, each to at or below where it was.
    fn compact(&mut self) {
        let (mut from, mut to, mut kept) = (0, 0, 0);
        for r in 0..self.runs.len() {
            let (l, room) = self.runs[r];
            let span = &mut self.spans[l.index()];
            if span.run == run(r) {
                if to != from {
                    self.slab.copy_within(from..from + span.len as usize, to);
                }
                span.start = to;
                span.run = run(kept);
                self.runs[kept] = (l, room);
                kept += 1;
                to += room as usize;
            }
            from += room as usize;
        }
        self.runs.truncate(kept);
        self.slab.truncate(to);
        self.dead = 0;
    }

    /// Runs tile the slab, each live one where its list's span says, and `dead` sums the rest.
    #[cfg(test)]
    pub(super) fn assert_consistent(&self) {
        let (mut from, mut dead) = (0, 0);
        for (r, &(l, room)) in self.runs.iter().enumerate() {
            let span = self.spans[l.index()];
            if span.run == run(r) {
                assert_eq!((span.start, span.cap), (from, room), "run {r}");
            } else {
                dead += room as usize;
            }
            from += room as usize;
        }
        assert_eq!((from, dead), (self.slab.len(), self.dead));
        for (l, span) in self.spans.iter().enumerate() {
            assert!(span.len <= span.cap, "list {l}");
            assert!(
                span.start + span.cap as usize <= self.slab.len(),
                "list {l}"
            );
            assert_eq!(span.run.is_none(), span.cap == 0, "list {l}");
            assert!(
                (span.run).is_none_or(|r| self.runs[r.get() - 1].0.index() == l),
                "list {l}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::{Lists, REUSED_UP_TO};

    #[derive(Clone, Debug)]
    enum Op {
        Push(usize, u32),
        Pop(usize),
        SwapRemove(usize, usize),
        Release(usize),
        Reload(Vec<u32>),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            40 => (any::<usize>(), any::<u32>()).prop_map(|(l, e)| Op::Push(l, e)),
            10 => any::<usize>().prop_map(Op::Pop),
            20 => (any::<usize>(), any::<usize>()).prop_map(|(l, at)| Op::SwapRemove(l, at)),
            10 => any::<usize>().prop_map(Op::Release),
            1 => vec(0u32..4, 1..8).prop_map(Op::Reload),
        ]
    }

    /// Lists past [`REUSED_UP_TO`], so a release compacts; only the first `room.len()` are used.
    fn load(lists: &mut Lists<u32, u32>, room: &[u32]) -> Vec<Vec<u32>> {
        let mut all = room.to_vec();
        all.resize(REUSED_UP_TO + 1, 0);
        lists.load(&all, u32::MAX);
        vec![Vec::new(); room.len()]
    }

    /// Proptest's first find: compaction truncated the slab past a released list's old start.
    #[test]
    fn a_released_list_reads_empty_once_compaction_truncates_past_it() {
        let mut lists = Lists::<u32, u32>::new();
        load(&mut lists, &[0]);
        (0..5).for_each(|e| lists.push(0, e));
        lists.release(0);
        assert_eq!(lists.get(0), &[] as &[u32]);
        lists.assert_consistent();
    }

    proptest! {
        /// Any sequence leaves every list as `Vec<Vec<_>>` would, through growth, compaction, shrinking and reloads.
        #[test]
        fn lists_agree_with_vecs(room in vec(0u32..4, 1..8), ops in vec(op(), 0..400)) {
            let mut lists = Lists::<u32, u32>::new();
            let mut vecs = load(&mut lists, &room);
            for op in ops {
                let n = vecs.len();
                match op {
                    Op::Push(l, e) => {
                        lists.push(l % n, e);
                        vecs[l % n].push(e);
                    }
                    Op::Pop(l) => prop_assert_eq!(lists.pop(l % n), vecs[l % n].pop()),
                    Op::SwapRemove(l, at) => {
                        let l = l % n;
                        if !vecs[l].is_empty() {
                            let at = at % vecs[l].len();
                            vecs[l].swap_remove(at);
                            prop_assert_eq!(lists.swap_remove(l, at), vecs[l].get(at).copied());
                        }
                    }
                    Op::Release(l) => {
                        lists.release(l % n);
                        vecs[l % n].clear();
                    }
                    Op::Reload(room) => vecs = load(&mut lists, &room),
                }
                lists.assert_consistent();
                for (l, v) in vecs.iter().enumerate() {
                    prop_assert_eq!(lists.get(l), &v[..]);
                    prop_assert_eq!(lists.len(l) as usize, v.len());
                }
            }
        }
    }
}
