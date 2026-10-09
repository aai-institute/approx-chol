//! Sampled elimination of a whole SDDM in minimum-degree order (Algorithm 8), and its stored columns.

mod cholesky;
mod graph;
mod ordering;

pub(crate) use cholesky::SampledCholesky;

use crate::clique_tree::{CliqueSampler, EdgeCount, Multi, Single, Streams};
use crate::config::Multiplicity;
use crate::sddm::Sddm;
use crate::Real;
use cholesky::Steps;
use graph::EliminationGraph;
use ordering::DynamicOrdering;

pub(crate) fn eliminate<T: Real>(
    sddm: &Sddm<'_, T>,
    multiplicity: Multiplicity,
    streams: Streams,
) -> SampledCholesky<T> {
    match multiplicity {
        Multiplicity::Single => run::<T, Single>(sddm, streams, ()),
        Multiplicity::Split(k) => run::<T, Multi>(sddm, streams, k),
    }
}

/// One step short of `n`, leaving the vertex the solve zeroes.
fn run<T: Real, C: EdgeCount>(
    sddm: &Sddm<'_, T>,
    streams: Streams,
    split: C::Split,
) -> SampledCholesky<T> {
    let mut graph = EliminationGraph::<C, T>::of(sddm, split);
    let n = graph.n();
    let mut ordering = DynamicOrdering::new(&graph);
    let mut clique = CliqueSampler::<T, C>::new(streams.min_degree());
    let mut steps = Steps::with_capacity(n);
    for _ in 1..n {
        let v = ordering.next_vertex();
        let pivot = clique.eliminate(graph.neighbors(v), split);
        steps.push(v, pivot, clique.column());
        graph.eliminate_vertex(v);
        for &(u, w, weight) in clique.fill() {
            graph.add_fill(u, w, weight);
        }
        ordering.update(clique.star(), clique.fill());
    }
    steps.finish(ordering.next_vertex())
}
