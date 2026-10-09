//! Sampled elimination of a whole SDDM in minimum-degree order (Algorithm 8), and its stored columns.

mod clique;
mod graph;
mod multiplicity;
mod ordering;
mod sequence;
mod star;

pub(crate) use sequence::Sequence;

use crate::config::Multiplicity;
use crate::sampling::Streams;
use crate::sgdd::Sddm;
use crate::Real;
use clique::{sample_column, Column, Fill};
use graph::EliminationGraph;
use multiplicity::{EdgeCount, Multi, Single};
use ordering::DynamicOrdering;
use sequence::SequenceBuilder;
use star::Star;

pub(crate) fn eliminate<T: Real>(
    sddm: &Sddm<'_, T>,
    multiplicity: Multiplicity,
    streams: Streams,
) -> Sequence<T> {
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
) -> Sequence<T> {
    let mut graph = EliminationGraph::<C, T>::of(sddm, split);
    let n = graph.n();
    let mut ordering = DynamicOrdering::new(&graph);
    let mut sampler = streams.kernel();
    let (mut star, mut column, mut fill) = (Star::new(), Column::new(), Fill::new());
    let mut sequence = SequenceBuilder::with_capacity(n);
    for _ in 1..n {
        let v = ordering.next_vertex();
        star.refill(&graph, v, split);
        fill.clear();
        let pivot = sample_column(&star, &mut sampler, &mut column, &mut fill);
        sequence.push(v, pivot, &column);
        graph.eliminate_vertex(v);
        for &(u, w, weight) in &fill {
            graph.add_fill(u, w, weight);
        }
        ordering.update(&star, &fill);
    }
    sequence.finish(ordering.next_vertex())
}
