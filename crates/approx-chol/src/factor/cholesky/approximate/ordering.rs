use super::graph::EliminationGraph;
use crate::factor::clique_tree::Copies;
use crate::{Index, Real};

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`), first in first out within a bucket: last in first out, under degrees that merge duplicates, measured 2x the PCG iterations on power-law bipartite graphs.
pub(super) struct MinDegreeOrder<I> {
    elems: Vec<QueueElem<I>>,
    lists: Vec<I>,
    min_list: usize,
    scale: KeyScale,
}

/// 12 bytes at `u32`: `Option<u32>` keys measured +2-3% on grids.
struct QueueElem<I> {
    /// A bucket's head holds its tail here, so appending needs no array of tails.
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
        self.scale = scale;
        self.lists.clear();
        self.lists.resize(scale.n_buckets(), sentinel());
        self.elems.clear();
        self.elems.reserve(n);
        self.min_list = self.lists.len();
        for v in 0..n {
            self.elems.push(QueueElem {
                prev: sentinel(),
                next: sentinel(),
                key: REMOVED,
            });
            if Some(v) != pin {
                let key = key(graph.degree(v));
                self.elems[v].key = key;
                self.append(v, scale.bucket(key));
            }
        }
    }

    /// Panics on an empty queue: every step leaves at least one vertex behind.
    pub(super) fn next_vertex(&mut self) -> usize {
        while self.lists[self.min_list] == sentinel() {
            self.min_list += 1;
        }
        let v = self.lists[self.min_list].index();
        self.unlink(v, self.min_list);
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
        self.unlink(v, old);
        self.append(v, new);
    }

    fn append(&mut self, v: usize, list: usize) {
        let head = self.lists[list];
        let tail = match head == sentinel() {
            true => {
                self.lists[list] = I::from_index(v);
                I::from_index(v)
            }
            false => {
                let tail = self.elems[head.index()].prev;
                self.elems[tail.index()].next = I::from_index(v);
                self.elems[head.index()].prev = I::from_index(v);
                tail
            }
        };
        self.elems[v].prev = tail;
        self.elems[v].next = sentinel();
        self.min_list = self.min_list.min(list);
    }

    fn unlink(&mut self, v: usize, list: usize) {
        let QueueElem { prev, next, .. } = self.elems[v];
        let head = self.lists[list].index();
        if v == head {
            self.lists[list] = next;
        } else {
            self.elems[prev.index()].next = next;
        }
        match (next == sentinel(), v == head) {
            (false, _) => self.elems[next.index()].prev = prev,
            (true, false) => self.elems[head].prev = prev,
            (true, true) => {}
        }
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
    use crate::factor::clique_tree::{Multi, Single};

    /// Path 0-1-2-3 at 2^31 copies an edge: inner keys saturate, and an end's neighbor's key follows its degree down once the end goes.
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
        let end = order.next_vertex();
        let inner = graph
            .neighbors(end)
            .map(|(u, _, _)| u)
            .next()
            .expect("a path's end has a neighbor");
        assert_eq!(order.elems[inner as usize].key, super::REMOVED - 1);
        graph.eliminate_vertex(end);
        order.update(&graph, std::iter::once(inner));
        assert_eq!(order.elems[inner as usize].key, 1 << 31);
    }

    /// Path 0-1-2 and edge 3-4: 1, rekeyed to degree 1 once 0 goes, queues behind 2, 3 and 4, then drops below them once 2 goes.
    #[test]
    fn equal_keys_pop_first_in_first_out() {
        let mut graph = EliminationGraph::<f64, u32, Single>::new();
        graph.reset(&[2; 5], Single);
        for (u, v) in [(0, 1), (1, 2), (3, 4)] {
            graph.push_pair(u, v, Single, 1.0);
        }
        let mut order = MinDegreeOrder::new();
        order.load(&graph, Single, None, graph.edge_ends());
        let mut popped = Vec::new();
        for _ in 0..4 {
            let v = order.next_vertex();
            popped.push(v);
            let neighbors: Vec<u32> = graph.neighbors(v).map(|(u, _, _)| u).collect();
            graph.eliminate_vertex(v);
            order.update(&graph, neighbors.into_iter());
        }
        assert_eq!(popped, [0, 2, 1, 3]);
    }
}
