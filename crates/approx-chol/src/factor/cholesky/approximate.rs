//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (Algorithm 8).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{CliqueSampler, Copies, Fill};
use crate::factor::sddm::Sddm;
#[cfg(feature = "serde")]
use crate::factor::within_one;
use crate::{Index, Real};
use graph::EliminationGraph;
use ordering::MinDegreeOrder;

/// A task's AC or AC2 elimination; small blocks reuse its buffers (malloc was 17% of 20-vertex covers).
pub(in crate::factor) struct CliqueTreeElimination<T, I, C: Copies> {
    split: C,
    graph: EliminationGraph<T, I, C>,
    clique: CliqueSampler<T, I, C>,
    fill: Fill<T, I, C>,
    order: MinDegreeOrder<I>,
}

impl<T: Real, I: Index, C: Copies> CliqueTreeElimination<T, I, C> {
    pub(in crate::factor) fn new(split: C) -> Self {
        Self {
            split,
            graph: EliminationGraph::new(),
            clique: CliqueSampler::new(split),
            fill: Fill::new(),
            order: MinDegreeOrder::new(),
        }
    }

    /// Every vertex but one in minimum-degree order, each star replaced by a sampled clique tree (Algorithm 8); the last, `pin` when given, is the slot the solve zeroes.
    pub(super) fn eliminate(
        &mut self,
        sddm: Sddm<T, I>,
        mut draws: Xoshiro256PlusPlus,
        pin: Option<usize>,
    ) -> (ApproximateCholesky<T, I>, usize) {
        self.graph.load(&sddm, self.split);
        drop(sddm);
        let n = self.graph.n();
        self.order.load(&self.graph, self.split, pin);
        let mut steps = ApproximateCholesky::with_capacity(n, self.graph.edge_ends());
        self.pivot_all(&mut draws, &mut steps);
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        steps.neighbors.shrink_to_fit();
        steps.coefficients.shrink_to_fit();
        (steps, pin.unwrap_or_else(|| self.order.next_vertex()))
    }

    #[inline(never)] // Inlined, its codegen moves with the buffer handling: grids +1-3%.
    fn pivot_all(&mut self, draws: &mut Xoshiro256PlusPlus, steps: &mut ApproximateCholesky<T, I>) {
        let Self {
            graph,
            clique,
            fill,
            order: ordering,
            ..
        } = self;
        for _ in 1..graph.n() {
            let v = ordering.next_vertex();
            fill.clear();
            let column = clique.eliminate(draws, graph.neighbors(v), fill);
            steps.push(v, column);
            graph.eliminate_vertex(v);
            for &(u, w, copies, weight) in fill.iter() {
                graph.push_pair(u, w, copies, weight);
            }
            ordering.update(clique.degree_losses(), fill);
        }
    }
}

/// A step's neighbors follow the previous step's, so the solve's running `usize` offset finds them at any size.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
struct StepHeader<T, I> {
    vertex: I,
    /// Below `n`, so `I` always holds it and a `u32` header stays 16 bytes.
    len: I,
    /// `D⁻¹` premultiplied by the retained fraction, so no pass rebuilds it.
    pivot_scale: T,
}

/// `LDLᵀ` with sampled columns, appended in pivot order: a serial recurrence, so shares are composed against the original pivot.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct ApproximateCholesky<T, I> {
    headers: Vec<StepHeader<T, I>>,
    neighbors: Vec<I>,
    coefficients: Vec<T>,
}

impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Growing to `entries` by doubling measured +1.5-2.5% on grids.
    fn with_capacity(n: usize, entries: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(entries),
            coefficients: Vec::with_capacity(entries),
        }
    }

    /// The one site deriving the remainder's share.
    fn push(
        &mut self,
        vertex: usize,
        (weight, shares, remainder): (T, impl Iterator<Item = (I, T)>, I),
    ) {
        let start = self.neighbors.len();
        let mut retained = T::one();
        for (neighbor, share) in shares {
            self.neighbors.push(neighbor);
            self.coefficients.push(share);
            retained = retained - share;
        }
        self.neighbors.push(remainder);
        self.coefficients.push(retained);
        // `retained / (w²/d)` is `1/w` for the remainder's `w`, finite as fill and `S` never weigh below `MIN_POSITIVE`.
        self.headers.push(StepHeader {
            vertex: I::from_index(vertex),
            len: I::from_index(self.neighbors.len() - start),
            pivot_scale: T::one() / weight,
        });
    }

    /// Forward, zero the `pinned` entry, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(super) fn substitute(&self, y: &mut [T], pinned: usize) {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        let mut end = 0;
        for header in headers {
            let span = end..end + header.len.index();
            end = span.end;
            let pivot = y[header.vertex.index()];
            match coefficients[span.clone()] {
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    let j = neighbors[span.start].index();
                    y[j] = y[j] + pivot;
                }
                _ => {
                    for (&j, &c) in neighbors[span.clone()].iter().zip(&coefficients[span]) {
                        y[j.index()] = y[j.index()] + c * pivot;
                    }
                }
            }
            y[header.vertex.index()] = pivot * header.pivot_scale;
        }
        y[pinned] = T::zero();
        for header in headers.iter().rev() {
            let span = end - header.len.index()..end;
            end = span.start;
            let v = header.vertex.index();
            match coefficients[span.clone()] {
                [] => {}
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    y[v] = y[v] + y[neighbors[span.start].index()];
                }
                [.., retained] => {
                    let mut lanes = [T::zero(); LANES];
                    lanes[0] = retained * y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(LANES);
                    let mut cs = coefficients[span].chunks_exact(LANES);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..LANES {
                            lanes[lane] = lanes[lane] + cs[lane] * y[js[lane].index()];
                        }
                    }
                    let mut total = lanes.iter().fold(T::zero(), |sum, &lane| sum + lane);
                    for (&j, &c) in js.remainder().iter().zip(cs.remainder()) {
                        total = total + c * y[j.index()];
                    }
                    y[v] = total;
                }
            }
        }
    }
}

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const LANES: usize = 4;

#[cfg(feature = "serde")]
impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Each slot but `pin` pivots once, giving only to slots not yet pivoted, shares of at most its whole.
    pub(super) fn fits(&self, slots: usize, pin: usize) -> bool {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        if headers.len() + 1 != slots || neighbors.len() != coefficients.len() {
            return false;
        }
        let mut pivoted = vec![false; slots];
        let mut end = 0usize;
        for header in headers {
            let (v, start) = (header.vertex.index(), end);
            end = end.saturating_add(header.len.index());
            if v == pin
                || v >= slots
                || std::mem::replace(&mut pivoted[v], true)
                || end > neighbors.len()
            {
                return false;
            }
            let later = neighbors[start..end]
                .iter()
                .all(|&j| j.index() < slots && !pivoted[j.index()]);
            let shares = &coefficients[start..end];
            let pivot = header.pivot_scale.is_finite() && header.pivot_scale > T::zero();
            let bounded = pivot && within_one(shares.iter().copied());
            if !later || !bounded || matches!(shares, [c] if *c != T::one()) {
                return false;
            }
        }
        end == neighbors.len()
    }
}
