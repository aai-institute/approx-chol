//! Sampled elimination of a whole SDDM in minimum-degree order (Algorithm 8), and its stored columns.

mod cholesky;
mod graph;
mod ordering;

pub(super) use cholesky::SampledCholesky;

use crate::factor::clique_tree::{CliqueSampler, Copies, Multi, Single, Streams};
use crate::factor::config::Multiplicity;
use crate::factor::sddm::Sddm;
use crate::storage::Fill;
use crate::Real;
use cholesky::Steps;
use graph::EliminationGraph;
use ordering::DynamicOrdering;

pub(super) fn eliminate<T: Real>(
    sddm: Sddm<'_, T>,
    multiplicity: Multiplicity,
    streams: Streams,
) -> SampledCholesky<T> {
    match multiplicity {
        Multiplicity::Single => run::<T, Single>(sddm, streams, ()),
        Multiplicity::Split(k) => run::<T, Multi>(sddm, streams, k),
    }
}

/// One step short of `n`, leaving the vertex the solve zeroes.
fn run<T: Real, C: Copies>(
    sddm: Sddm<'_, T>,
    streams: Streams,
    split: C::Split,
) -> SampledCholesky<T> {
    let mut graph = EliminationGraph::<T, C>::of(&sddm, split);
    drop(sddm);
    let n = graph.n();
    let mut ordering = DynamicOrdering::new(&graph, split);
    let mut clique = CliqueSampler::<T, C>::new(streams.remainder(), split);
    let mut fill = Fill::new();
    let mut steps = Steps::with_capacity(n);
    for _ in 1..n {
        let v = ordering.next_vertex();
        fill.clear();
        let pivot = clique.eliminate(graph.neighbors(v), &mut fill);
        steps.push(v, pivot, clique.column());
        graph.eliminate_vertex(v);
        for &(u, w, weight) in &fill {
            graph.add_fill(u, w, weight);
        }
        ordering.update(clique.star(), &fill);
    }
    steps.finish(ordering.next_vertex())
}
