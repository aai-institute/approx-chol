//! Elimination graph for approximate Cholesky factorization.
//!
//! Building one from CSR input lives in [`ingest`]; this module owns only the
//! graph itself and the elimination operations on it.

mod ingest;

use crate::types::{count_as_scalar, Real};
use crate::{CsrRef, Error};

/// Named return type for [`AdjListGraph::from_sddm`].
pub(crate) struct GraphBuild<G, T: Real> {
    pub graph: G,
    pub diagonal: Vec<T>,
    /// `None` when the graph is connected, which is the one block case.
    pub layout: Option<BlockLayout>,
    /// The Gremban ground vertex, when ingestion closed any row deficit with one.
    /// Appended last, so it is the highest-numbered vertex in the graph and the
    /// last vertex of whichever block ends up holding it.
    pub ground: Option<u32>,
}

/// Every vertex once, its connected components back to back: the order the factor
/// stores blocks in.
///
/// One array rather than one per component, because the same sequence answers all
/// three questions asked of it — where each block ends, which global vertices a
/// block owns, and what permutation maps the two coordinate systems.
pub(crate) struct BlockLayout {
    order: Vec<u32>,
    /// Exclusive end of each block in `order`; the next block starts where this
    /// one stops, so no block can claim a vertex twice or leave a gap.
    ends: Vec<u32>,
}

impl BlockLayout {
    pub(crate) fn block_count(&self) -> usize {
        self.ends.len()
    }

    /// Each block's global vertex names, in storage order.
    pub(crate) fn blocks(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.ends.iter().scan(0usize, |start, &end| {
            let end = end as usize;
            let block = &self.order[*start..end];
            *start = end;
            Some(block)
        })
    }

    /// The same sequence read as a permutation: position `i` holds the input
    /// vertex the factor keeps at `i`.
    pub(crate) fn into_order(self) -> Vec<u32> {
        self.order
    }

    /// Ascending within each block, which puts the ground vertex — the highest
    /// numbered — last in whichever block holds it.
    fn sort_blocks(&mut self) {
        let mut start = 0usize;
        for &end in &self.ends {
            let end = end as usize;
            self.order[start..end].sort_unstable();
            start = end;
        }
    }
}

/// A neighbor entry produced by star elimination. Carries the edge's
/// multiplicity storage, so the AC path has no multiplicity field to fill in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor<T, C> {
    pub to: u32,
    /// Accumulated weight the neighbor's copies carry between them.
    pub fill_weight: T,
    pub count: C,
}

struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0u64; n.div_ceil(64)],
        }
    }

    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }
}

/// How an edge stores its multiplicity, and everything that follows from it: AC is
/// AC2 at one copy per edge, so the whole difference between them is this trait.
///
/// `Single` is a ZST, so the AC edge stays exactly as wide as an edge carrying no
/// count at all (asserted in the tests below).
pub(crate) trait EdgeCount: Clone + Copy {
    /// What choosing this layout carries beyond the type: nothing for AC, the
    /// multiplicity for AC2. `Single` cannot name a `k`, which is what makes an AC
    /// factorization over split multi-edges a type error rather than a mistake the
    /// caller has to avoid.
    type Split: Copy;

    /// Whether every edge of this layout is exactly one copy. `Single` knows it
    /// statically, which is what lets the single-copy path sort on weights instead
    /// of on quotients that are all division by one.
    const SINGLE_COPY: bool;

    fn one() -> Self;
    fn get(&self) -> u32;

    /// Share of `total` carried by one copy. The star's sort key and its sampled
    /// fill weight are the same division, and for `Single` it is the identity — so
    /// the AC path performs no division rather than dividing by one.
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// Keep at most `limit` copies of a coalesced neighbor, and report how many the
    /// cap discarded. `Single` stores nothing and keeps one, so its discard count is
    /// the duplicates the merge collapsed.
    fn cap(copies: u32, limit: u32) -> (Self, u32);

    /// Split every edge into copies of itself, before the degrees are read, and
    /// report how many that is. The degree-bucket scale, the star's merge cap and
    /// the per-neighbor sample count are one number because they are one return
    /// value; a cap of one is what keeps AC's star single-copy throughout.
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: Self::Split) -> u32;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// AC2: this edge's virtual copy count, set to the [`SplitFactor`] by
/// [`MultiEdgeGraph::mark_split_edges`] and only lowered from there by the merge cap.
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    /// Any count at all, which only a test needs: elimination makes a `Multi` from
    /// the validated split or from [`EdgeCount::cap`], never from a bare number.
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

/// The factor AC2 splits each edge by, distinct from the per-edge count [`Multi`]
/// carries. Splitting fewer than twice is standard AC, so the factors that exist are
/// exactly the ones AC2 is defined for: `1/k` cannot be the infinity a zero would
/// give, and no split can be a no-op.
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
    fn cap(copies: u32, _limit: u32) -> (Self, u32) {
        (Self, copies - 1)
    }
    /// A slim edge has nowhere to put a multiplicity, so AC eliminates the graph
    /// exactly as ingested.
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
    fn cap(copies: u32, limit: u32) -> (Self, u32) {
        let kept = copies.min(limit);
        (Self(kept), copies - kept)
    }
    #[inline]
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: SplitFactor) -> u32 {
        graph.mark_split_edges(split);
        split.get()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real, C> {
    weight: T,
    to: u32,
    /// Index of this edge's mirror in `adj[to]`; whatever moves an edge must
    /// preserve it.
    rev: u32,
    count: C,
}

impl<T: Real, C: EdgeCount> Edge<T, C> {
    #[inline]
    fn new(weight: T, to: u32, rev: u32) -> Self {
        Self {
            weight,
            to,
            rev,
            count: C::one(),
        }
    }

    /// The weight all copies carry between them, which is what the edge stores:
    /// splitting sets the count and leaves the weight alone.
    #[inline]
    fn fill_weight(&self) -> T {
        self.weight
    }
}

/// Adjacency-list elimination graph, generic over edge multiplicity storage.
pub(crate) struct AdjListGraph<C, T: Real> {
    /// Per-vertex adjacency list.
    adj: Vec<Vec<Edge<T, C>>>,
    /// `eliminated[v]` is `true` after `eliminate_vertex(v)` has been called.
    eliminated: BitVec,
}

/// AC path: no multi-edge tracking.
pub(crate) type SlimGraph<T> = AdjListGraph<Single, T>;

/// AC2 path: edges with virtual multi-edge counts.
pub(crate) type MultiEdgeGraph<T> = AdjListGraph<Multi, T>;

/// Keep capacity of tiny adjacency lists to reduce allocator churn, but release
/// large vectors to avoid retaining fill-heavy buffers across eliminations.
const RETAIN_ADJ_CAPACITY_MAX: usize = 64;

impl<C: EdgeCount, T: Real> AdjListGraph<C, T> {
    /// Construct from a CSR SDDM matrix.
    pub(crate) fn from_sddm(csr: CsrRef<'_, T, u32>) -> Result<GraphBuild<Self, T>, Error> {
        ingest::from_sddm(csr)
    }

    fn from_adjacency(adj: Vec<Vec<Edge<T, C>>>) -> Self {
        Self {
            eliminated: BitVec::new(adj.len()),
            adj,
        }
    }

    /// Number of vertices (fixed at construction time).
    pub(crate) fn n(&self) -> usize {
        self.adj.len()
    }

    /// Current degree of vertex `v` (sum of multi-edge counts; includes stale entries).
    pub(crate) fn degree(&self, v: usize) -> usize {
        self.adj[v].iter().map(|e| e.count.get() as usize).sum()
    }

    /// Collect live (non-eliminated, positive-weight) neighbors of `v` into `scratch`.
    pub(crate) fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T, C>>) {
        scratch.clear();
        scratch.extend(self.adj[v].iter().filter_map(|e| {
            // Positive predicate: a NaN weight is dead (`!(w > 0)` differs from
            // `w <= 0` at NaN). if/else (not `bool::then`) keeps `fill_weight()`
            // lazy for dead edges and avoids `clippy::filter_map_bool_then`.
            if e.weight > T::zero() && !self.eliminated.get(e.to as usize) {
                Some(Neighbor {
                    to: e.to,
                    fill_weight: e.fill_weight(),
                    count: e.count,
                })
            } else {
                None
            }
        }));
    }

    /// Mark `v` as eliminated and release its adjacency storage.
    pub(crate) fn eliminate_vertex(&mut self, v: usize) {
        self.eliminated.set(v);
        while let Some(edge) = self.adj[v].pop() {
            let u = edge.to as usize;
            if self.eliminated.get(u) {
                continue;
            }
            debug_assert!(
                (edge.rev as usize) < self.adj[u].len(),
                "reverse pointer out of bounds: rev={} but adj[{}].len()={}",
                edge.rev,
                u,
                self.adj[u].len()
            );
            remove_edge_at(&mut self.adj, u, edge.rev as usize);
        }
        if self.adj[v].capacity() > RETAIN_ADJ_CAPACITY_MAX {
            self.adj[v] = Vec::new();
        }
    }

    /// Insert a symmetric fill edge between `u` and `v` with the given weight.
    pub(crate) fn add_fill_edge(&mut self, u: u32, v: u32, weight: T) {
        if u == v {
            return;
        }
        add_edge_pair(&mut self.adj, u as usize, v as usize, weight);
    }

    /// Moves the subgraph over `vertices` out of `self`, renumbered to
    /// `0..vertices.len()` in the order given.
    ///
    /// A component is closed under edges, so each list moves intact and only its
    /// endpoints need relabeling: every `rev` still addresses the position it did
    /// in the parent.
    pub(crate) fn take_component(&mut self, vertices: &[u32], local_of: &mut [u32]) -> Self {
        debug_assert_eq!(local_of.len(), self.adj.len());
        for (local, &global) in vertices.iter().enumerate() {
            local_of[global as usize] = local as u32;
        }
        let adjacency = vertices
            .iter()
            .map(|&global| {
                let mut edges = core::mem::take(&mut self.adj[global as usize]);
                for edge in &mut edges {
                    edge.to = local_of[edge.to as usize];
                }
                edges
            })
            .collect();
        Self::from_adjacency(adjacency)
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// Mark each edge as `k` virtual copies of it. The weight stays the total across
    /// the copies, so this cannot underflow one away and [`EdgeCount::per_copy`]
    /// divides where a single copy is what's wanted.
    pub(crate) fn mark_split_edges(&mut self, k: SplitFactor) {
        for adj_list in &mut self.adj {
            for edge in adj_list.iter_mut() {
                edge.count = k.into();
            }
        }
    }
}

#[inline]
fn add_edge_pair<T: Real, C: EdgeCount>(
    adj: &mut [Vec<Edge<T, C>>],
    u: usize,
    v: usize,
    weight: T,
) {
    // u32 reverse pointers; overflow is unreachable for tractable inputs,
    // so assert (release too) rather than truncate and corrupt removal.
    assert!(
        adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
        "adjacency list exceeds u32 edge capacity"
    );
    let rev_u = adj[v].len() as u32;
    let rev_v = adj[u].len() as u32;
    adj[u].push(Edge::new(weight, v as u32, rev_u));
    adj[v].push(Edge::new(weight, u as u32, rev_v));
}

/// Remove `adj[u][idx]` in O(1) via swap-remove and repair the moved edge's
/// reverse pointer in its opposite adjacency list.
fn remove_edge_at<T: Real, C: EdgeCount>(adj: &mut [Vec<Edge<T, C>>], u: usize, idx: usize) {
    let last_idx = adj[u].len() - 1;
    adj[u].swap_remove(idx);
    if idx < last_idx {
        let moved = adj[u][idx];
        let w = moved.to as usize;
        let rev = moved.rev as usize;
        adj[w][rev].rev = idx as u32;
    }
}

/// Connected components among the first `n_real` vertices laid out back to back,
/// or `None` when the graph is connected. Traversal follows every edge, so a
/// ground vertex (index `>= n_real`) links the blocks it touches without being
/// counted as its own component.
fn block_layout<T: Real, C: EdgeCount>(
    adj: &[Vec<Edge<T, C>>],
    n_real: usize,
) -> Option<BlockLayout> {
    let n = adj.len();
    let mut visited = BitVec::new(n);
    let mut stack: Vec<usize> = Vec::new();
    let mut layout = BlockLayout {
        order: Vec::with_capacity(n),
        ends: Vec::new(),
    };
    for start in 0..n_real {
        if visited.get(start) {
            continue;
        }
        visited.set(start);
        stack.push(start);
        while let Some(v) = stack.pop() {
            layout.order.push(v as u32);
            for e in &adj[v] {
                let u = e.to as usize;
                if !visited.get(u) {
                    visited.set(u);
                    stack.push(u);
                }
            }
        }
        layout.ends.push(layout.order.len() as u32);
    }
    // The first traversal reaching every vertex means the graph is connected,
    // so the common case never sorts or returns a layout.
    if layout.ends.len() <= 1 && layout.order.len() == n {
        return None;
    }
    layout.sort_blocks();
    Some(layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AC edge must not pay for the multiplicity it does not store: `Single`
    /// is a ZST, so both layouts are what the two hand-written structs were.
    #[test]
    fn edge_layout_is_unchanged_by_the_shared_definition() {
        assert_eq!(
            size_of::<Edge<f64, Single>>(),
            size_of::<f64>() + 2 * size_of::<u32>()
        );
        assert_eq!(
            size_of::<Edge<f64, Multi>>(),
            size_of::<Edge<f64, Single>>() + size_of::<f64>(),
            "one u32 plus its alignment padding"
        );
        assert_eq!(size_of::<Single>(), 0);
    }

    /// The cap the split reports is the count it wrote on the edges. A return value
    /// that drifted from [`MultiEdgeGraph::mark_split_edges`] would cap every star
    /// at a multiplicity the graph does not carry, and no other test would notice.
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
