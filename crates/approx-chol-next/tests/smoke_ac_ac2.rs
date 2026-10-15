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
use grid::grid;
use sparse::Sparse;

cases! {
    /// The scale at which bucket layout and fill-in bookkeeping carry load the property suite's eight-vertex graphs never reach.
    fn smoke_medium_grid(entry in entries(all), types in types(all), sparse in sparse) {
        let graph = grid(100, 100);
        let mut rhs = vec![0.0; graph.n];
        (rhs[0], rhs[graph.n - 1]) = (1.0, -1.0);
        let reduced = entry.reduce(&graph, types);
        assert_eq!(
            reduced.floats(),
            [true],
            "{types:?}: a connected Laplacian floats"
        );
        let config = sparse.config();
        let x = reduced.factorize(&config).solve(&rhs);
        let label = format!("{types:?} {config:?}");
        assert!(x.iter().all(|x| x.is_finite()), "{label}");
        assert!(x.iter().any(|x| x.abs() > types.epsilon()), "{label}");
    }
}
