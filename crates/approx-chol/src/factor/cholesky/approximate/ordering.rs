use super::graph::EliminationGraph;
use crate::factor::clique_tree::{Copies, Fill};
use crate::{Index, Real};

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`).
pub(super) struct MinDegreeOrder<I> {
    elems: Vec<QueueElem<I>>,
    lists: Vec<I>,
    min_list: usize,
    bucket_base: usize,
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
    /// Buckets span `split · n`, Laplacians.jl's AC2 scale; `last` is never queued.
    pub(super) fn new<T: Real, C: Copies>(
        graph: &EliminationGraph<T, I, C>,
        split: C::Split,
        last: Option<usize>,
    ) -> Self {
        let n = graph.n();
        let bucket_base = (C::initial(split).get() as usize).saturating_mul(n).max(1);
        let mut lists: Vec<I> = vec![sentinel(); n_buckets(bucket_base)];
        let mut elems = Vec::with_capacity(n);
        let mut min_list = lists.len();
        for v in 0..n {
            if Some(v) == last {
                elems.push(QueueElem {
                    prev: sentinel(),
                    next: sentinel(),
                    key: REMOVED,
                });
                continue;
            }
            let key = graph.degree(v).min(u64::from(u32::MAX - 1)) as u32;
            let list = key_map(key, bucket_base);
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
        Self {
            elems,
            lists,
            min_list,
            bucket_base,
            net: vec![0; n],
            touched: Vec::new(),
        }
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
        if key == REMOVED || delta == 0 {
            return;
        }
        let moved = i64::from(key) + delta;
        debug_assert!(moved >= 0, "a degree estimate counts every copy it loses");
        let moved = moved.clamp(0, i64::from(REMOVED - 1)) as u32;
        let (old, new) = (
            key_map(key, self.bucket_base),
            key_map(moved, self.bucket_base),
        );
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

/// Sole derivation of the bucket count, so `key_map`'s cap and `new` agree on the last bucket.
fn n_buckets(bucket_base: usize) -> usize {
    bucket_base.saturating_mul(2).saturating_add(1)
}

/// Exact below `bucket_base`, coarser above it.
fn key_map(key: u32, bucket_base: usize) -> usize {
    let degree = key as usize;
    match degree <= bucket_base {
        true => degree,
        false => (bucket_base + degree / bucket_base).min(n_buckets(bucket_base) - 1),
    }
}
