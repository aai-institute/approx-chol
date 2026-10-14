//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (Algorithm 8).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single};
use crate::factor::config::Multiplicity;
use crate::factor::sddm::Sddm;
use crate::{Index, Real};
use graph::EliminationGraph;
use ordering::MinDegreeOrder;

/// AC or AC2 by `multiplicity`, each its own monomorphized elimination.
pub(super) fn factor<T: Real, I: Index>(
    sddm: Sddm<'_, T, I>,
    multiplicity: Multiplicity,
    draws: Xoshiro256PlusPlus,
    last: Option<usize>,
) -> ApproximateCholesky<T, I> {
    match multiplicity {
        Multiplicity::Single => eliminate::<T, I, Single>(sddm, draws, (), last),
        Multiplicity::Split(k) => eliminate::<T, I, Multi>(sddm, draws, k, last),
    }
}

/// Every vertex but one in minimum-degree order, each star replaced by a sampled clique tree (Algorithm 8); the last, `last` when given, is the slot the solve zeroes.
fn eliminate<T: Real, I: Index, C: Copies>(
    sddm: Sddm<'_, T, I>,
    draws: Xoshiro256PlusPlus,
    split: C::Split,
    last: Option<usize>,
) -> ApproximateCholesky<T, I> {
    let mut graph = EliminationGraph::<T, I, C>::of(&sddm, split);
    drop(sddm);
    let n = graph.n();
    let mut ordering = MinDegreeOrder::<I>::new(&graph, split, last);
    let mut draws = draws;
    let mut clique = CliqueSampler::<T, I, C>::new(split);
    let mut fill = Fill::new();
    let mut steps = Steps::with_capacity(n, graph.edge_ends());
    for _ in 1..n {
        let v = ordering.next_vertex();
        fill.clear();
        let pivot = clique.eliminate(&mut draws, graph.neighbors(v), &mut fill);
        steps.push(v, pivot, clique.column());
        graph.eliminate_vertex(v);
        for &(u, w, weight) in &fill {
            graph.push_pair(u, w, C::one(), weight);
        }
        ordering.update(clique.degree_losses(), &fill);
    }
    steps.finish(last.unwrap_or_else(|| ordering.next_vertex()))
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

/// The eliminated columns, appended in pivot order.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
struct Steps<T, I> {
    headers: Vec<StepHeader<T, I>>,
    neighbors: Vec<I>,
    coefficients: Vec<T>,
}

impl<T: Real, I: Index> Steps<T, I> {
    /// Growing to `entries` by doubling measured +1.5-2.5% on grids.
    fn with_capacity(n: usize, entries: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(entries),
            coefficients: Vec::with_capacity(entries),
        }
    }

    /// The one site deriving the remainder's share.
    fn push(&mut self, vertex: usize, pivot: T, column: Option<(&[(I, T)], I)>) {
        let start = self.neighbors.len();
        let retained = match column {
            None => T::one(),
            Some((shares, remainder)) => {
                let mut left = T::one();
                for &(neighbor, share) in shares {
                    self.neighbors.push(neighbor);
                    self.coefficients.push(share);
                    left = left - share;
                }
                self.neighbors.push(remainder);
                self.coefficients.push(left);
                left
            }
        };
        // Admitted input reaches this only past ~2^1074 range in one star (#168); `one` drops the scale.
        debug_assert!(
            (T::one() / pivot).is_finite(),
            "pivot has no finite inverse"
        );
        let inverse = match T::one() / pivot {
            inverse if inverse.is_finite() => inverse,
            _ => T::one(),
        };
        self.headers.push(StepHeader {
            vertex: I::from_index(vertex),
            len: I::from_index(self.neighbors.len() - start),
            pivot_scale: retained * inverse,
        });
    }

    /// A factor cannot exist without naming the vertex its steps left behind.
    fn finish(mut self, uneliminated: usize) -> ApproximateCholesky<T, I> {
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        self.neighbors.shrink_to_fit();
        self.coefficients.shrink_to_fit();
        ApproximateCholesky {
            steps: self,
            uneliminated: I::from_index(uneliminated),
        }
    }
}

/// `LDLᵀ` with sampled columns: a serial recurrence, so shares are composed against the original pivot.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct ApproximateCholesky<T, I> {
    steps: Steps<T, I>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    uneliminated: I,
}

impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Forward, zero the uneliminated entry, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(super) fn substitute(&self, y: &mut [T]) {
        let Steps {
            headers,
            neighbors,
            coefficients,
        } = &self.steps;
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
        y[self.uneliminated.index()] = T::zero();
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
    /// Each slot but the uneliminated one pivots once, giving only to slots not yet pivoted.
    pub(super) fn fits(&self, slots: usize) -> bool {
        let Steps {
            headers,
            neighbors,
            coefficients,
        } = &self.steps;
        let last = self.uneliminated.index();
        if headers.len() + 1 != slots || last >= slots || neighbors.len() != coefficients.len() {
            return false;
        }
        let mut pivoted = vec![false; slots];
        let mut end = 0;
        for header in headers {
            let (v, start) = (header.vertex.index(), end);
            end += header.len.index();
            if v == last
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
            let finite = header.pivot_scale.is_finite() && shares.iter().all(|c| c.is_finite());
            if !later || !finite || matches!(shares, [c] if *c != T::one()) {
                return false;
            }
        }
        end == neighbors.len()
    }
}
