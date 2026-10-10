//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (Algorithm 8).

mod graph;
mod ordering;
mod steps;

pub(super) use steps::ApproximateCholesky;

use crate::factor::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single, Stream};
use crate::factor::config::Multiplicity;
use crate::factor::sddm::Sddm;
use crate::Real;
use graph::EliminationGraph;
use ordering::MinDegreeOrder;
use steps::Steps;

pub(super) fn eliminate<T: Real>(
    sddm: Sddm<'_, T>,
    multiplicity: Multiplicity,
    stream: Stream,
) -> ApproximateCholesky<T> {
    match multiplicity {
        Multiplicity::Single => run::<T, Single>(sddm, stream, ()),
        Multiplicity::Split(k) => run::<T, Multi>(sddm, stream, k),
    }
}

/// One step short of `n`, leaving the vertex the solve zeroes.
fn run<T: Real, C: Copies>(
    sddm: Sddm<'_, T>,
    stream: Stream,
    split: C::Split,
) -> ApproximateCholesky<T> {
    let mut graph = EliminationGraph::<T, C>::of(&sddm, split);
    drop(sddm);
    let n = graph.n();
    let mut ordering = MinDegreeOrder::new(&graph, split);
    let mut clique = CliqueSampler::<T, C>::new(stream, split);
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
        ordering.update(clique.degree_losses(), &fill);
    }
    steps.finish(ordering.next_vertex())
}
