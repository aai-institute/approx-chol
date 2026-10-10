//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (Algorithm 8).

mod graph;
mod ordering;
mod steps;

pub(super) use steps::ApproximateCholesky;

use crate::factor::clique_tree::{CliqueSampler, Copies, Fill, Multi, SequentialDraws, Single};
use crate::factor::config::Multiplicity;
use crate::factor::sddm::Sddm;
use crate::Real;
use graph::EliminationGraph;
use ordering::MinDegreeOrder;
use steps::Steps;

/// AC or AC2 by `multiplicity`, each its own monomorphized elimination.
pub(super) fn factor<T: Real>(
    sddm: Sddm<'_, T>,
    multiplicity: Multiplicity,
    draws: SequentialDraws,
) -> ApproximateCholesky<T> {
    match multiplicity {
        Multiplicity::Single => eliminate::<T, Single>(sddm, draws, ()),
        Multiplicity::Split(k) => eliminate::<T, Multi>(sddm, draws, k),
    }
}

/// Every vertex but one in minimum-degree order, each star replaced by a sampled clique tree (Algorithm 8); the last is the slot the solve zeroes.
fn eliminate<T: Real, C: Copies>(
    sddm: Sddm<'_, T>,
    draws: SequentialDraws,
    split: C::Split,
) -> ApproximateCholesky<T> {
    let mut graph = EliminationGraph::<T, C>::of(&sddm, split);
    drop(sddm);
    let n = graph.n();
    let mut ordering = MinDegreeOrder::new(&graph, split);
    let mut draws = draws;
    let mut clique = CliqueSampler::<T, C>::new(split);
    let mut fill = Fill::new();
    let mut steps = Steps::with_capacity(n);
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
