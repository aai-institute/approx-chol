use super::*;
use crate::graph::Single;

/// `Single` is a ZST, so the AC edge pays nothing for multiplicity it never stores.
#[test]
fn edge_layout_is_unchanged_by_the_shared_definition() {
    assert_eq!(
        size_of::<Edge<f64, Single>>(),
        size_of::<f64>() + 2 * size_of::<u32>()
    );
    assert_eq!(
        size_of::<Edge<f64, Multi>>(),
        size_of::<Edge<f64, Single>>() + size_of::<f64>(),
        "one u32 plus its alignment padding"
    );
    assert_eq!(size_of::<Single>(), 0);
}

/// Zero and NaN weights are dead though the neighbor lives; reading either as
/// live puts a phantom edge in the star.
#[test]
fn only_positively_weighted_edges_are_live() {
    let graph = MultiEdgeGraph::<f64>::from_adjacency(vec![
        vec![
            Edge::new(2.0, 1, 0),
            Edge::new(0.0, 2, 0),
            Edge::new(f64::NAN, 3, 0),
        ],
        vec![Edge::new(2.0, 0, 0)],
        vec![Edge::new(0.0, 0, 1)],
        vec![Edge::new(f64::NAN, 0, 2)],
    ]);

    let mut neighbors = Vec::new();
    graph.live_neighbors(0, &mut neighbors);

    let live: Vec<u32> = neighbors.iter().map(|n| n.to).collect();
    assert_eq!(
        live,
        vec![1],
        "only the positively weighted neighbor is live"
    );
}

/// Every edge's mirror names it back at the index the edge stores, which is the
/// invariant removal reads and the one a migration could silently break.
fn assert_mirrors_pair<T: Real + core::fmt::Debug, C: EdgeCount>(graph: &AdjListGraph<C, T>) {
    for v in 0..graph.n() {
        for (index, edge) in graph.edges.list(v).iter().enumerate() {
            let mirror = graph.edges.list(edge.to as usize)[edge.rev as usize];
            assert_eq!(
                mirror.to as usize, v,
                "adj[{v}][{index}] -> {} whose mirror at {} names {}",
                edge.to, edge.rev, mirror.to
            );
            assert_eq!(mirror.weight, edge.weight, "mirrors carry one weight");
        }
    }
}

/// A path 0-1-2-3, every list packed to its length so the first fill edge migrates.
fn packed_path() -> MultiEdgeGraph<f64> {
    MultiEdgeGraph::from_adjacency(vec![
        vec![Edge::new(1.0, 1, 0)],
        vec![Edge::new(1.0, 0, 0), Edge::new(1.0, 2, 0)],
        vec![Edge::new(1.0, 1, 1), Edge::new(1.0, 3, 0)],
        vec![Edge::new(1.0, 2, 1)],
    ])
}

/// The migrated list leaves its old region behind, so an unrepaired reverse pointer
/// would still index a plausible edge rather than fault.
#[test]
fn a_migrating_list_keeps_every_reverse_pointer() {
    let mut graph = packed_path();
    graph.add_fill_edge(0, 2, 5.0);
    assert_mirrors_pair(&graph);
    graph.add_fill_edge(0, 3, 7.0);
    graph.add_fill_edge(1, 3, 9.0);
    assert_mirrors_pair(&graph);

    let mut neighbors = Vec::new();
    graph.live_neighbors(0, &mut neighbors);
    let mut live: Vec<u32> = neighbors.iter().map(|n| n.to).collect();
    live.sort_unstable();
    assert_eq!(live, vec![1, 2, 3], "every fill edge is reachable from 0");
}

/// Removal moves an edge within the list it removes from, and the moved edge's own
/// mirror has to follow it — across a migration as well as within a region.
#[test]
fn eliminating_a_migrated_vertex_repairs_the_edges_it_moves() {
    let mut graph = packed_path();
    graph.add_fill_edge(0, 2, 5.0);
    graph.add_fill_edge(0, 3, 7.0);

    graph.eliminate_vertex(0);
    assert_mirrors_pair(&graph);
    for v in 1..4 {
        let mut neighbors = Vec::new();
        graph.live_neighbors(v, &mut neighbors);
        assert!(
            neighbors.iter().all(|n| n.to != 0),
            "adj[{v}] still names the eliminated vertex"
        );
    }
}

/// Growth alternating between two full lists migrates each one repeatedly; a region
/// that kept no room after moving would migrate on every single push.
#[test]
fn alternating_growth_stays_consistent() {
    let mut graph = packed_path();
    for round in 0..8 {
        graph.add_fill_edge(0, 2, 1.0 + round as f64);
        graph.add_fill_edge(1, 3, 2.0 + round as f64);
        assert_mirrors_pair(&graph);
    }
    assert_eq!(graph.edges.list(0).len(), 9);
    assert_eq!(graph.edges.list(1).len(), 10);
}

/// An emptied list at the buffer's end is given back, so a long elimination does not
/// keep every list it has already finished with.
#[test]
fn eliminating_the_last_vertex_gives_its_region_back() {
    let mut graph = packed_path();
    let before = graph.edges.edges.len();
    graph.eliminate_vertex(3);
    assert!(
        graph.edges.edges.len() < before,
        "the tail region was kept: {before} slots before, {} after",
        graph.edges.edges.len()
    );
    assert_mirrors_pair(&graph);
}
