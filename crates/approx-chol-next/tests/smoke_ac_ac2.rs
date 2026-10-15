#[path = "common/entry.rs"]
mod entry;
#[path = "common/floats.rs"]
mod floats;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/modes.rs"]
mod modes;

use entry::{Entry, Types};
use grid::grid;
use modes::modes_for;
use rstest::rstest;

/// The scale at which bucket layout and fill-in bookkeeping carry load the property suite's eight-vertex graphs never reach.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn smoke_medium_grid(#[case] entry: Entry) {
    let graph = grid(100, 100);
    let mut rhs = vec![0.0; graph.n];
    (rhs[0], rhs[graph.n - 1]) = (1.0, -1.0);
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states a grid");
        assert_eq!(
            reduced.floats(),
            [true],
            "{types:?}: a connected Laplacian floats"
        );
        for config in modes_for(&graph) {
            let x = reduced.factorize(&config).solve(&rhs);
            let label = format!("{types:?} {config:?}");
            assert!(x.iter().all(|x| x.is_finite()), "{label}");
            assert!(x.iter().any(|x| x.abs() > types.epsilon()), "{label}");
        }
    }
}
