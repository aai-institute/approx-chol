//! Elimination graph for approximate Cholesky factorization.
//!
//! Building one from CSR input lives in [`ingest`]; this module owns only the
//! graph itself and the elimination operations on it.

mod ingest;

use crate::types::count_as_scalar;
use crate::{CsrRef, Error, Real};

/// Named return type for [`AdjListGraph::from_sddm`].
pub(crate) struct GraphBuild<G, T: Real> {
    pub graph: G,
    pub diagonal: Vec<T>,
    pub components: Option<Vec<Vec<u32>>>,
}

/// A neighbor entry produced by star elimination. Carries the edge's
/// multiplicity storage, so the AC path has no multiplicity field to fill in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor<T, C> {
    pub to: u32,
    /// Accumulated fill weight (weight × count for AC2, just weight for AC).
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

/// How an edge stores its multiplicity: the only thing that differs between the
/// AC and AC2 edge layouts, so it is the only thing either one defines.
///
/// `Single` is a ZST, which keeps the AC edge exactly as wide as it was before
/// the count existed (asserted in the tests below).
pub(crate) trait EdgeCount: Clone + Copy {
    fn one() -> Self;
    fn get(&self) -> u32;
    /// Fill weight this multiplicity contributes at edge weight `weight`.
    fn scale<T: Real>(&self, weight: T) -> T;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// AC2: a virtual multiplicity set by [`MultiEdgeGraph::mark_split_edges`].
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    pub(crate) fn new(count: u32) -> Self {
        Self(count)
    }
}

impl EdgeCount for Single {
    #[inline]
    fn one() -> Self {
        Self
    }
    #[inline]
    fn get(&self) -> u32 {
        1
    }
    #[inline]
    fn scale<T: Real>(&self, weight: T) -> T {
        weight
    }
}

impl EdgeCount for Multi {
    #[inline]
    fn one() -> Self {
        Self(1)
    }
    #[inline]
    fn get(&self) -> u32 {
        self.0
    }
    #[inline]
    fn scale<T: Real>(&self, weight: T) -> T {
        weight * count_as_scalar::<T, _>(self.0)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real, C> {
    weight: T,
    to: u32,
    rev: u32,
    count: C,
}

impl<T: Real, C: EdgeCount> Edge<T, C> {
    /// A single edge of the given weight; [`link_pair`] assigns its endpoint and
    /// reverse pointer, the only place either is set.
    #[inline]
    fn new(weight: T) -> Self {
        Self {
            weight,
            to: 0,
            rev: 0,
            count: C::one(),
        }
    }

    /// The same edge pointing at a different endpoint.
    #[inline]
    fn reindex(self, to: u32, rev: u32) -> Self {
        Self { to, rev, ..self }
    }

    #[inline]
    fn fill_weight(&self) -> T {
        self.count.scale(self.weight)
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

    /// The subgraph over `vertices`, renumbered to `0..vertices.len()` in the
    /// order given. `local_of` is a scratch map of `usize::MAX`, left as found.
    pub(crate) fn extract_component(
        &self,
        diagonal: &[T],
        vertices: &[u32],
        local_of: &mut [usize],
    ) -> (Self, Vec<T>) {
        debug_assert_eq!(local_of.len(), self.adj.len());
        for (local, &global) in vertices.iter().enumerate() {
            local_of[global as usize] = local;
        }
        // The parent degree bounds the local one, so no list has to grow.
        let mut adjacency: Vec<Vec<Edge<T, C>>> = vertices
            .iter()
            .map(|&global| Vec::with_capacity(self.adj[global as usize].len()))
            .collect();
        for (local_u, &global_u) in vertices.iter().enumerate() {
            for &edge in &self.adj[global_u as usize] {
                let local_v = local_of[edge.to as usize];
                if local_v != usize::MAX && local_u < local_v {
                    link_pair(&mut adjacency, local_u, local_v, edge);
                }
            }
        }
        for &global in vertices {
            local_of[global as usize] = usize::MAX;
        }
        let local_diagonal = vertices
            .iter()
            .map(|&vertex| diagonal[vertex as usize])
            .collect();
        (Self::from_adjacency(adjacency), local_diagonal)
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// Mark each edge as `k` virtual copies at `weight / k`.
    pub(crate) fn mark_split_edges(&mut self, k: u32) {
        if k <= 1 {
            return;
        }
        let inv_k = T::one() / count_as_scalar::<T, _>(k);
        for adj_list in &mut self.adj {
            for edge in adj_list.iter_mut() {
                edge.weight = edge.weight * inv_k;
                edge.count = Multi::new(k);
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
    link_pair(adj, u, v, Edge::new(weight));
}

/// Store `edge` in both endpoints' adjacency lists, each copy pointing at the
/// other and carrying the index it sits at there. The one place a reverse pointer
/// is produced, so it is also the one place their `u32` range is checked.
#[inline]
fn link_pair<T: Real, C: EdgeCount>(
    adj: &mut [Vec<Edge<T, C>>],
    u: usize,
    v: usize,
    edge: Edge<T, C>,
) {
    // Overflow is unreachable for tractable inputs, so assert (release too)
    // rather than truncate and corrupt removal.
    assert!(
        adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
        "adjacency list exceeds u32 edge capacity"
    );
    let rev_u = adj[v].len() as u32;
    let rev_v = adj[u].len() as u32;
    adj[u].push(edge.reindex(v as u32, rev_u));
    adj[v].push(edge.reindex(u as u32, rev_v));
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

/// Connected components among the first `n_real` vertices, or `None` when the
/// graph is connected. Traversal follows every edge, so a ground vertex (index
/// `>= n_real`) links the blocks it touches without being counted as its own
/// component.
fn components<T: Real, C: EdgeCount>(
    adj: &[Vec<Edge<T, C>>],
    n_real: usize,
) -> Option<Vec<Vec<u32>>> {
    let n = adj.len();
    let mut visited = BitVec::new(n);
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<u32>> = Vec::new();
    for start in 0..n_real {
        if visited.get(start) {
            continue;
        }
        let mut component = Vec::new();
        visited.set(start);
        stack.push(start);
        while let Some(v) = stack.pop() {
            component.push(v as u32);
            for e in &adj[v] {
                let u = e.to as usize;
                if !visited.get(u) {
                    visited.set(u);
                    stack.push(u);
                }
            }
        }
        components.push(component);
    }
    // The first traversal reaching every vertex means the graph is connected,
    // so the common case never sorts or returns a component list.
    if components.len() <= 1 && components.first().map_or(0, Vec::len) == n {
        return None;
    }
    for component in &mut components {
        component.sort_unstable();
    }
    Some(components)
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
}
