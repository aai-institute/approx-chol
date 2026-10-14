//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (Algorithm 8).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single};
use crate::factor::config::Multiplicity;
use crate::factor::sddm::Sddm;
use crate::Real;
use graph::EliminationGraph;
use ordering::MinDegreeOrder;

/// AC or AC2 by `multiplicity`, each its own monomorphized elimination.
pub(super) fn factor<T: Real>(
    sddm: Sddm<'_, T>,
    multiplicity: Multiplicity,
    draws: Xoshiro256PlusPlus,
) -> ApproximateCholesky<T> {
    match multiplicity {
        Multiplicity::Single => eliminate::<T, Single>(sddm, draws, ()),
        Multiplicity::Split(k) => eliminate::<T, Multi>(sddm, draws, k),
    }
}

/// Every vertex but one in minimum-degree order, each star replaced by a sampled clique tree (Algorithm 8); the last is the slot the solve zeroes.
fn eliminate<T: Real, C: Copies>(
    sddm: Sddm<'_, T>,
    draws: Xoshiro256PlusPlus,
    split: C::Split,
) -> ApproximateCholesky<T> {
    let mut graph = EliminationGraph::<T, C>::of(&sddm, split);
    drop(sddm);
    let n = graph.n();
    let mut ordering = MinDegreeOrder::new(&graph, split);
    let mut draws = draws;
    let mut clique = CliqueSampler::<T, C>::new(split);
    let mut fill = Fill::new();
    let mut steps = Steps::with_capacity(n, graph.edge_ends());
    for _ in 1..n {
        let v = ordering.next_vertex();
        fill.clear();
        let pivot = clique.eliminate(&mut draws, graph.neighbors(v), &mut fill);
        steps.push(v, pivot, clique.column());
        graph.eliminate_vertex(v);
        for &(u, w, weight) in &fill {
            graph.add_fill(u, w, weight);
        }
        ordering.update(clique.degree_losses(), &fill);
    }
    steps.finish(ordering.next_vertex())
}

/// A step's neighbors follow the previous step's, so the solve's running `usize` offset finds them at any size.
#[derive(Clone, Copy, Debug)]
struct StepHeader<T> {
    vertex: u32,
    /// Below `n`, so `u32` always holds it and the header stays 16 bytes.
    len: u32,
    /// `D⁻¹` premultiplied by the retained fraction, so no pass rebuilds it.
    pivot_scale: T,
}

/// The eliminated columns, appended in pivot order.
#[derive(Clone, Debug)]
struct Steps<T> {
    headers: Vec<StepHeader<T>>,
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
}

impl<T: Real> Steps<T> {
    /// Growing to `entries` by doubling measured +1.5-2.5% on grids.
    fn with_capacity(n: usize, entries: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(entries),
            coefficients: Vec::with_capacity(entries),
        }
    }

    /// The one site deriving the remainder's share.
    fn push(&mut self, vertex: usize, pivot: T, column: Option<(&[(u32, T)], u32)>) {
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
            vertex: vertex as u32,
            len: (self.neighbors.len() - start) as u32,
            pivot_scale: retained * inverse,
        });
    }

    /// A factor cannot exist without naming the vertex its steps left behind.
    fn finish(mut self, uneliminated: usize) -> ApproximateCholesky<T> {
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        self.neighbors.shrink_to_fit();
        self.coefficients.shrink_to_fit();
        ApproximateCholesky {
            steps: self,
            uneliminated: uneliminated as u32,
        }
    }
}

/// `LDLᵀ` with sampled columns: a serial recurrence, so shares are composed against the original pivot.
#[derive(Clone, Debug)]
pub(super) struct ApproximateCholesky<T> {
    steps: Steps<T>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    uneliminated: u32,
}

impl<T: Real> ApproximateCholesky<T> {
    /// Forward, zero the uneliminated entry, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(super) fn substitute(&self, y: &mut [T]) {
        let Steps {
            headers,
            neighbors,
            coefficients,
        } = &self.steps;
        let mut end = 0;
        for header in headers {
            let span = end..end + header.len as usize;
            end = span.end;
            let pivot = y[header.vertex as usize];
            match coefficients[span.clone()] {
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    let j = neighbors[span.start] as usize;
                    y[j] = y[j] + pivot;
                }
                _ => {
                    for (&j, &c) in neighbors[span.clone()].iter().zip(&coefficients[span]) {
                        y[j as usize] = y[j as usize] + c * pivot;
                    }
                }
            }
            y[header.vertex as usize] = pivot * header.pivot_scale;
        }
        y[self.uneliminated as usize] = T::zero();
        for header in headers.iter().rev() {
            let span = end - header.len as usize..end;
            end = span.start;
            let v = header.vertex as usize;
            match coefficients[span.clone()] {
                [] => {}
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    y[v] = y[v] + y[neighbors[span.start] as usize];
                }
                [.., retained] => {
                    let mut lanes = [T::zero(); LANES];
                    lanes[0] = retained * y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(LANES);
                    let mut cs = coefficients[span].chunks_exact(LANES);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..LANES {
                            lanes[lane] = lanes[lane] + cs[lane] * y[js[lane] as usize];
                        }
                    }
                    let mut total = lanes.iter().fold(T::zero(), |sum, &lane| sum + lane);
                    for (&j, &c) in js.remainder().iter().zip(cs.remainder()) {
                        total = total + c * y[j as usize];
                    }
                    y[v] = total;
                }
            }
        }
    }
}

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const LANES: usize = 4;
