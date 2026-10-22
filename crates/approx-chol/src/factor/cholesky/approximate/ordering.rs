use super::graph::EliminationGraph;
use crate::factor::clique_tree::{Copies, Fill};
use crate::{Index, Real};

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`).
pub(super) struct MinDegreeOrder<I> {
    elems: Vec<QueueElem<I>>,
    lists: Vec<I>,
    min_list: usize,
    scale: KeyScale,
    /// Net degree change per vertex, zero at rest, so each moves once per pivot: a list would sort per pivot (measured 1.5-1.7x the star sort).
    net: Vec<i64>,
    touched: Vec<I>,
}

/// 12 bytes at `u32`: `Option<u32>` keys measured +2-3% on grids.
struct QueueElem<I> {
    prev: I,
    next: I,
    /// `u32::MAX` once removed; a live key saturates below it, as only the order matters.
    key: u32,
}

impl<I: Index> MinDegreeOrder<I> {
    pub(super) fn new() -> Self {
        Self {
            elems: Vec::new(),
            lists: Vec::new(),
            min_list: 0,
            scale: KeyScale {
                base: 1,
                divisor: 1,
            },
            net: Vec::new(),
            touched: Vec::new(),
        }
    }

    /// `pin` is never queued.
    pub(super) fn load<T: Real, C: Copies>(
        &mut self,
        graph: &EliminationGraph<T, I, C>,
        split: C,
        pin: Option<usize>,
        edge_ends: usize,
    ) {
        let n = graph.n();
        let scale = KeyScale::new(n, edge_ends, split.get());
        let Self {
            elems, lists, net, ..
        } = self;
        lists.clear();
        lists.resize(scale.n_buckets(), sentinel());
        elems.clear();
        elems.reserve(n);
        let mut min_list = lists.len();
        for v in 0..n {
            if Some(v) == pin {
                elems.push(QueueElem {
                    prev: sentinel(),
                    next: sentinel(),
                    key: REMOVED,
                });
                continue;
            }
            let key = graph.degree(v).min(u64::from(REMOVED - 1)) as u32;
            let list = scale.bucket(key);
            let head = lists[list];
            elems.push(QueueElem {
                prev: sentinel(),
                next: head,
                key,
            });
            if head != sentinel() {
                elems[head.index()].prev = I::from_index(v);
            }
            lists[list] = I::from_index(v);
            min_list = min_list.min(list);
        }
        net.clear();
        net.resize(n, 0);
        self.min_list = min_list;
        self.scale = scale;
    }

    /// Panics on an empty queue: every step leaves at least one vertex behind.
    pub(super) fn next_vertex(&mut self) -> usize {
        while self.lists[self.min_list] == sentinel() {
            self.min_list += 1;
        }
        let v = self.lists[self.min_list].index();
        let next = self.elems[v].next;
        self.lists[self.min_list] = next;
        if next != sentinel() {
            self.elems[next.index()].prev = sentinel();
        }
        self.elems[v].key = REMOVED;
        v
    }

    /// The star's degree losses, then fill's gains.
    pub(super) fn update<T: Real, C: Copies>(
        &mut self,
        losses: impl Iterator<Item = (I, u32)>,
        fill: &Fill<T, I, C>,
    ) {
        for (u, copies) in losses {
            self.add(u, -i64::from(copies));
        }
        for &(u, w, copies, _) in fill {
            self.add(u, i64::from(copies.get()));
            self.add(w, i64::from(copies.get()));
        }
        for i in 0..self.touched.len() {
            let v = self.touched[i].index();
            let delta = std::mem::take(&mut self.net[v]);
            self.apply_delta(v, delta);
        }
        self.touched.clear();
    }

    fn add(&mut self, v: I, delta: i64) {
        let slot = &mut self.net[v.index()];
        if *slot == 0 {
            self.touched.push(v);
        }
        *slot += delta;
    }

    /// `i64` so negating a full-range `u32` count cannot sign-flip as an `i32` would.
    fn apply_delta(&mut self, v: usize, delta: i64) {
        let key = self.elems[v].key;
        // A saturated key no longer counts copies, so it stays saturated rather than undercount.
        if key >= REMOVED - 1 || delta == 0 {
            return;
        }
        let moved = i64::from(key) + delta;
        debug_assert!(moved >= 0, "a degree estimate counts every copy it loses");
        let moved = moved.clamp(0, i64::from(REMOVED - 1)) as u32;
        let (old, new) = (self.scale.bucket(key), self.scale.bucket(moved));
        self.elems[v].key = moved;
        if old == new {
            return;
        }
        let QueueElem { prev, next, .. } = self.elems[v];
        match prev == sentinel() {
            true => self.lists[old] = next,
            false => self.elems[prev.index()].next = next,
        }
        if next != sentinel() {
            self.elems[next.index()].prev = prev;
        }
        let head = self.lists[new];
        self.elems[v].prev = sentinel();
        self.elems[v].next = head;
        if head != sentinel() {
            self.elems[head.index()].prev = I::from_index(v);
        }
        self.lists[new] = I::from_index(v);
        self.min_list = self.min_list.min(new);
    }
}

/// Linked-list terminator.
fn sentinel<I: Index>() -> I {
    I::max_value()
}

/// A popped vertex's key, which [`MinDegreeOrder::apply_delta`] skips.
const REMOVED: u32 = u32::MAX;

/// Translated from Laplacians.jl's `keyMap` and `ApproxCholPQ` (MIT, see NOTICE): keys exact up to `base`, then `divisor` copies per bucket, in `2 · base + 1` buckets.
#[derive(Clone, Copy)]
struct KeyScale {
    base: usize,
    divisor: usize,
}

impl KeyScale {
    /// Laplacians.jl's `base = k · n`, capped at `8 (n + edge_ends)`; past the cap `divisor` grows so no key, at most `k · edge_ends / 2`, passes the last bucket.
    fn new(n: usize, edge_ends: usize, copies: u32) -> Self {
        let copies = copies as usize;
        let budget = 8usize.saturating_mul(n.saturating_add(edge_ends));
        let base = copies.saturating_mul(n).min(budget).max(1);
        let reach = copies
            .saturating_mul(edge_ends / 2)
            .min(REMOVED as usize - 1);
        Self {
            base,
            divisor: base.max(reach.div_ceil(base)),
        }
    }

    /// Sole derivation of the bucket count, so `bucket`'s cap and `load` agree on the last bucket.
    fn n_buckets(self) -> usize {
        self.base.saturating_mul(2).saturating_add(1)
    }

    // Called out of line, its two fields spill around each call in `apply_delta` (+0.3-0.6% build).
    #[inline(always)]
    fn bucket(self, key: u32) -> usize {
        let degree = key as usize;
        match degree <= self.base {
            true => degree,
            false => (self.base + degree / self.divisor).min(self.n_buckets() - 1),
        }
    }
}
