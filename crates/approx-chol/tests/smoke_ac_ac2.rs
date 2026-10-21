#[path = "common/cases.rs"]
mod cases;
#[path = "common/entry.rs"]
mod entry;
#[path = "common/floats.rs"]
mod floats;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/sparse.rs"]
mod sparse;

use cases::cases;
use entry::{Entry, Types};
use graph::Graph;
use grid::grid;
use sparse::Sparse;

/// Bucket layout and fill bookkeeping carry load only past the property graphs' size.
fn smoke(entry: Entry, sparse: Sparse, types: Types, graph: &Graph) {
    let mut rhs = vec![0.0; graph.n];
    (rhs[0], rhs[graph.n - 1]) = (1.0, -1.0);
    let reduced = entry.reduce(graph, types);
    assert_eq!(
        reduced.floats(),
        [graph.ground.iter().all(|&g| g == 0.0)],
        "{types:?}: a connected graph floats unless grounded"
    );
    let config = sparse.config();
    let x = reduced.factorize(&config).solve(&rhs);
    let label = format!("{types:?} {config:?}");
    assert!(x.iter().all(|x| x.is_finite()), "{label}");
    assert!(x.iter().any(|x| x.abs() > types.epsilon()), "{label}");
}

cases! {
    fn smoke_medium_grid(route in sparse_routes(floating), types in types(all)) {
        let (entry, sparse) = route;
        smoke(entry, sparse, types, &grid(100, 100));
    }
}

cases! {
    /// Its first row tied to ground, so the grid is an SDDM.
    fn smoke_medium_grounded_grid(route in sparse_routes(grounded), types in types(all)) {
        let (entry, sparse) = route;
        let mut graph = grid(100, 100);
        graph.ground[..100].fill(1.0);
        smoke(entry, sparse, types, &graph);
    }
}
