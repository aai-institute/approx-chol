use super::multiplicity::{EdgeCount, Multi, SplitFactor};
use crate::types::Real;

/// Carries the edge's multiplicity storage, so the AC path has no field to fill in.
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

#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real, C> {
    weight: T,
    to: u32,
    /// Index of this edge's mirror in `adj[to]`; whatever moves an edge preserves it.
    rev: u32,
    count: C,
}

impl<T: Real, C: EdgeCount> Edge<T, C> {
    #[inline]
    pub(super) fn new(weight: T, to: u32, rev: u32) -> Self {
        Self {
            weight,
            to,
            rev,
            count: C::one(),
        }
    }

    /// Padding, and never below a region's `len`: a read of one indexes out of bounds
    /// rather than counting as an edge to vertex zero.
    #[inline]
    fn blank() -> Self {
        Self::new(T::zero(), u32::MAX, u32::MAX)
    }

    /// Splitting sets the count and leaves the weight alone.
    #[inline]
    fn fill_weight(&self) -> T {
        self.weight
    }
}

/// Where one vertex's edges sit in the flat buffer, and how far they can grow before
/// the list has to move.
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    len: u32,
    cap: u32,
}

impl Region {
    #[inline]
    fn end(&self) -> usize {
        self.start + self.len as usize
    }

    #[inline]
    fn limit(&self) -> usize {
        self.start + self.cap as usize
    }
}

/// Every adjacency list in one allocation. Reverse pointers index *within* a list, so a
/// list that outgrows its region moves elsewhere in the buffer without invalidating one.
pub(super) struct EdgeBuffer<T: Real, C> {
    edges: Vec<Edge<T, C>>,
    regions: Vec<Region>,
}

impl<T: Real, C: EdgeCount> EdgeBuffer<T, C> {
    fn with_capacity(vertices: usize, edges: usize) -> Self {
        Self {
            edges: Vec::with_capacity(edges),
            regions: Vec::with_capacity(vertices),
        }
    }

    #[inline]
    pub(super) fn vertices(&self) -> usize {
        self.regions.len()
    }

    #[inline]
    pub(super) fn list(&self, v: usize) -> &[Edge<T, C>] {
        let region = self.regions[v];
        &self.edges[region.start..region.end()]
    }

    /// `None` past the list's end, which is how the fill tells a claimed slot from a
    /// mirror that the upper triangle stored as zero.
    #[inline]
    fn get(&self, v: usize, index: u32) -> Option<Edge<T, C>> {
        let region = self.regions[v];
        (index < region.len).then(|| self.edges[region.start + index as usize])
    }

    pub(super) fn lists_mut(&mut self, mut visit: impl FnMut(&mut [Edge<T, C>])) {
        for v in 0..self.regions.len() {
            let region = self.regions[v];
            visit(&mut self.edges[region.start..region.end()]);
        }
    }

    /// Appends to the region opened last, which is the only one the buffer's end can
    /// still grow into.
    #[inline]
    fn append(&mut self, edge: Edge<T, C>) {
        let region = self
            .regions
            .last_mut()
            .expect("a region is open before any edge is appended");
        debug_assert_eq!(
            region.end(),
            self.edges.len(),
            "the open region is the tail"
        );
        // Truncating a u32 reverse pointer would corrupt removal, so assert in release too.
        assert!(
            region.len < u32::MAX,
            "adjacency list exceeds u32 edge capacity"
        );
        region.len += 1;
        region.cap = region.cap.max(region.len);
        self.edges.push(edge);
    }

    /// Each end's reverse pointer is where the other end's edge lands, so the pair is
    /// placed here rather than by a caller reading two lengths off the buffer first.
    #[inline]
    pub(super) fn push_pair(&mut self, u: usize, v: usize, weight: T) {
        let (rev_u, rev_v) = (self.regions[v].len, self.regions[u].len);
        self.push(u, Edge::new(weight, v as u32, rev_u));
        self.push(v, Edge::new(weight, u as u32, rev_v));
    }

    /// One look at the region, not one to read it and another to lengthen it: this and
    /// the removal below are 8% of a dense build.
    #[inline]
    fn push(&mut self, v: usize, edge: Edge<T, C>) {
        let region = &mut self.regions[v];
        if region.len == region.cap {
            return self.grow_and_push(v, edge);
        }
        let slot = region.end();
        region.len += 1;
        self.edges[slot] = edge;
    }

    /// A full region at the buffer's end grows into it; any other is copied to the end
    /// with slack of its own, so the copy amortizes. What a copy leaves behind is dead.
    #[cold]
    fn grow_and_push(&mut self, v: usize, edge: Edge<T, C>) {
        let region = self.regions[v];
        assert!(
            region.len < u32::MAX / 2,
            "adjacency list exceeds u32 edge capacity"
        );
        let len = region.len + 1;
        if region.limit() == self.edges.len() {
            self.edges.push(edge);
            self.regions[v] = Region {
                start: region.start,
                len,
                cap: len,
            };
            return;
        }
        let want = region_cap(len as usize);
        // Growing a `Vec` doubles it, and doubling the whole buffer to lengthen one
        // list costs more than every list in it.
        if self.edges.len() + want > self.edges.capacity() {
            self.edges.reserve_exact(want + self.edges.capacity() / 8);
        }
        let start = self.edges.len();
        self.edges.extend_from_within(region.start..region.end());
        self.edges.resize(start + want, Edge::blank());
        self.edges[start + region.len as usize] = edge;
        self.regions[v] = Region {
            start,
            len,
            cap: want as u32,
        };
    }

    #[inline]
    fn pop(&mut self, v: usize) -> Option<Edge<T, C>> {
        let region = &mut self.regions[v];
        let last = region.len.checked_sub(1)?;
        region.len = last;
        let slot = region.start + last as usize;
        Some(self.edges[slot])
    }

    /// Swap-remove, repairing the moved edge's reverse pointer. Both writes go through
    /// one slice of the list, so the removal is bounds-checked once rather than per access.
    fn remove_at(&mut self, u: usize, index: usize) {
        let region = &mut self.regions[u];
        let last = region.len as usize - 1;
        let start = region.start;
        region.len = last as u32;

        let list = &mut self.edges[start..start + last + 1];
        let moved = list[last];
        if index < last {
            list[index] = moved;
            let mirror = self.regions[moved.to as usize].start + moved.rev as usize;
            self.edges[mirror].rev = index as u32;
        }
    }

    /// Only the buffer's end can be given back; pooling the rest measured 3 reuses in 176.
    /// The vertex keeps the end, so a push it should never get appends rather than overwrites.
    fn release(&mut self, v: usize) {
        let region = self.regions[v];
        debug_assert_eq!(region.len, 0, "a released list has been emptied");
        if region.limit() == self.edges.len() {
            self.edges.truncate(region.start);
        }
        self.regions[v] = Region {
            start: self.edges.len(),
            len: 0,
            cap: 0,
        };
    }
}

/// Adjacency-list elimination graph, generic over edge multiplicity storage.
pub(crate) struct AdjListGraph<C, T: Real> {
    edges: EdgeBuffer<T, C>,
    /// `eliminated[v]` is `true` after `eliminate_vertex(v)` has been called.
    eliminated: BitVec,
}

/// AC2 path: edges with virtual multi-edge counts.
pub(crate) type MultiEdgeGraph<T> = AdjListGraph<Multi, T>;

impl<C: EdgeCount, T: Real> AdjListGraph<C, T> {
    pub(super) fn from_edges(edges: EdgeBuffer<T, C>) -> Self {
        Self {
            eliminated: BitVec::new(edges.vertices()),
            edges,
        }
    }

    /// Fixture route: the lists carry their own reverse pointers, and no region keeps
    /// slack, so any growth exercises a migration.
    #[cfg(test)]
    pub(super) fn from_adjacency(adj: Vec<Vec<Edge<T, C>>>) -> Self {
        let mut buffer = EdgeBuffer::with_capacity(adj.len(), adj.iter().map(Vec::len).sum());
        for list in adj {
            buffer.regions.push(Region {
                start: buffer.edges.len(),
                len: list.len() as u32,
                cap: list.len() as u32,
            });
            buffer.edges.extend(list);
        }
        Self::from_edges(buffer)
    }

    /// Number of vertices (fixed at construction time).
    pub(crate) fn n(&self) -> usize {
        self.edges.vertices()
    }

    /// Current degree of vertex `v` (sum of multi-edge counts; includes stale entries).
    pub(crate) fn degree(&self, v: usize) -> usize {
        self.edges
            .list(v)
            .iter()
            .map(|e| e.count.get() as usize)
            .sum()
    }

    /// Collect live (non-eliminated, positive-weight) neighbors of `v` into `scratch`.
    pub(crate) fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T, C>>) {
        scratch.clear();
        scratch.extend(self.edges.list(v).iter().filter_map(|e| {
            // Positive predicate, so a NaN weight is dead: `!(w > 0)` differs from
            // `w <= 0` there. if/else keeps `fill_weight()` lazy for dead edges.
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
        while let Some(edge) = self.edges.pop(v) {
            let u = edge.to as usize;
            if self.eliminated.get(u) {
                continue;
            }
            debug_assert!(
                (edge.rev as usize) < self.edges.list(u).len(),
                "reverse pointer out of bounds: rev={} but adj[{}].len()={}",
                edge.rev,
                u,
                self.edges.list(u).len()
            );
            self.edges.remove_at(u, edge.rev as usize);
        }
        self.edges.release(v);
    }

    /// Insert a symmetric fill edge between `u` and `v` with the given weight.
    pub(crate) fn add_fill_edge(&mut self, u: u32, v: u32, weight: T) {
        if u == v {
            return;
        }
        self.edges.push_pair(u as usize, v as usize, weight);
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// The weight stays the total, so no copy underflows; `per_copy` divides.
    pub(crate) fn mark_split_edges(&mut self, k: SplitFactor) {
        self.edges.lists_mut(|list| {
            for edge in list {
                edge.count = k.into();
            }
        });
    }
}

/// Fills the buffer front to back, so it needs no initializing. Ascending order is what
/// makes the slot the next edge naming a vertex will occupy a running count.
pub(super) struct SequentialFill<T: Real, C> {
    buffer: EdgeBuffer<T, C>,
    /// `claimed[v]` is the slot of region `v` that the next edge naming `v` occupies.
    claimed: Vec<u32>,
}

/// Room a region keeps for fill edges. Measured: an eighth of a row instead of a half
/// costs 7.0% of the build at degree 256, and a constant of 4 instead of 2 costs 1.7%.
const fn region_cap(edges: usize) -> usize {
    edges + edges / 2 + 2
}

impl<T: Real, C: EdgeCount> SequentialFill<T, C> {
    /// `entries` is how many edges the regions will hold between them. Reserving the
    /// slack too means the buffer is sized once: growing it would copy the whole graph.
    pub(super) fn new(vertices: usize, entries: usize) -> Self {
        Self {
            buffer: EdgeBuffer::with_capacity(vertices, region_cap(entries) + 2 * vertices),
            claimed: vec![0; vertices],
        }
    }

    /// Opens the next vertex's region, wide enough for `edges` and their slack, padding
    /// the one before it out to its own cap.
    pub(super) fn open(&mut self, edges: usize) {
        if let Some(previous) = self.buffer.regions.last() {
            self.buffer.edges.resize(previous.limit(), Edge::blank());
        }
        let start = self.buffer.edges.len();
        self.buffer.regions.push(Region {
            start,
            len: 0,
            cap: region_cap(edges).min(u32::MAX as usize) as u32,
        });
    }

    /// The edge to a vertex whose own region is not written yet. `weight` is the upper
    /// mirror's, the one both copies of the edge carry.
    #[inline]
    pub(super) fn upper(&mut self, to: usize, weight: T) {
        let rev = self.claim(to);
        self.buffer.append(Edge::new(weight, to as u32, rev));
    }

    /// The edge to a vertex already written, which is where its weight comes from. A
    /// mirror that is not there is an upper triangle entry stored as zero: no edge.
    #[inline]
    pub(super) fn lower(&mut self, to: usize) {
        let open = self.buffer.regions.len() - 1;
        let Some(mirror) = self
            .buffer
            .get(to, self.claimed[to])
            .filter(|edge| edge.to as usize == open)
        else {
            return;
        };
        let rev = self.claim(to);
        self.buffer.append(Edge::new(mirror.weight, to as u32, rev));
    }

    #[inline]
    fn claim(&mut self, to: usize) -> u32 {
        let rev = self.claimed[to];
        self.claimed[to] = rev + 1;
        rev
    }

    /// The last region keeps no slack: it already sits at the buffer's end, which it
    /// can grow into without moving.
    pub(super) fn finish(mut self) -> EdgeBuffer<T, C> {
        if let Some(last) = self.buffer.regions.last_mut() {
            last.cap = last.len;
        }
        self.buffer
    }
}

#[cfg(test)]
mod tests;
