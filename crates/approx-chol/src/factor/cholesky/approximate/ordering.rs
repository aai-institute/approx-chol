use super::graph::EliminationGraph;
use crate::factor::clique_tree::Copies;
use crate::{Index, Real};

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`).
pub(super) struct MinDegreeOrder<I> {
    elems: Vec<QueueElem<I>>,
    lists: Vec<I>,
    min_list: usize,
    scale: KeyScale,
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
        let Self { elems, lists, .. } = self;
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
            let key = key(graph.degree(v));
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

    /// Re-keys `vertices`, in their order, from `graph`'s degrees: a pivot moves only its star.
    pub(super) fn update<T: Real, C: Copies>(
        &mut self,
        graph: &EliminationGraph<T, I, C>,
        vertices: impl Iterator<Item = I>,
    ) {
        for v in vertices.map(I::index) {
            if self.elems[v].key != REMOVED {
                self.rekey(v, key(graph.degree(v)));
            }
        }
    }

    fn rekey(&mut self, v: usize, moved: u32) {
        let key = self.elems[v].key;
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

/// A popped or pinned vertex's key, which [`MinDegreeOrder::update`] skips.
const REMOVED: u32 = u32::MAX;

/// Saturated below [`REMOVED`], as only the order matters.
fn key(degree: u64) -> u32 {
    degree.min(u64::from(REMOVED - 1)) as u32
}

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

    // Called out of line, its two fields spill around each call in `rekey` (+0.3-0.6% build).
    #[inline(always)]
    fn bucket(self, key: u32) -> usize {
        let degree = key as usize;
        match degree <= self.base {
            true => degree,
            false => (self.base + degree / self.divisor).min(self.n_buckets() - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph::EliminationGraph;
    use super::MinDegreeOrder;
    use crate::factor::clique_tree::Multi;

    /// Path 0-1-2-3 at 2^31 copies an edge: inner keys saturate, and 2's follows its degree down once 3 goes, so it pops before 0.
    #[test]
    fn a_saturated_key_follows_its_degree_down() {
        let split = Multi::split(1 << 31).expect("AC2");
        let mut graph = EliminationGraph::<f64, u32, Multi>::new();
        graph.reset(&[2; 4], split);
        for v in 0..3 {
            graph.push_pair(v, v + 1, split, 1.0);
        }
        let mut order = MinDegreeOrder::new();
        order.load(&graph, split, None, graph.edge_ends());
        let mut popped = Vec::new();
        for _ in 0..3 {
            let v = order.next_vertex();
            popped.push(v);
            let neighbors: Vec<u32> = graph.neighbors(v).map(|(u, _, _)| u).collect();
            graph.eliminate_vertex(v);
            order.update(&graph, neighbors.into_iter());
        }
        assert_eq!(popped, [3, 2, 1]);
    }
}
