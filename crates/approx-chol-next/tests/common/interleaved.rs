use proptest::prelude::*;

use crate::graph::{Edge, Graph};

/// Components are the residue classes mod the returned `parts`, so they interleave and no block is contiguous; each class is complete, or complete bipartite between its alternate members.
pub fn interleaved_components() -> impl Strategy<Value = (Graph, usize)> {
    (2usize..=3, 5usize..=9, any::<bool>()).prop_flat_map(|(parts, n, bipartite)| {
        let pairs = n * (n - 1) / 2;
        prop::collection::vec(1u8..=4, pairs).prop_map(move |weights| {
            let mut edges = Vec::new();
            let pairs = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j)));
            for ((i, j), w) in pairs.zip(weights) {
                let alternate = (i / parts) % 2 != (j / parts) % 2;
                if i % parts == j % parts && (alternate || !bipartite) {
                    edges.push(Edge::negative(i, j, f64::from(w)));
                }
            }
            (Graph::new(n, edges), parts)
        })
    })
}
