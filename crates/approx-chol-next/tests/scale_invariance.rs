#[path = "common/entry.rs"]
mod entry;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/modes.rs"]
mod modes;

use entry::{Entry, Types};
use graph::{Edge, Graph};
use modes::modes_for;
use rstest::rstest;

const RHS: [f64; 6] = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5];

/// Uniform conductance `w`, so scaling `w` scales the whole matrix and nothing else.
fn complete(w: f64) -> Graph {
    let pairs = (0..6).flat_map(|a| (a + 1..6).map(move |b| (a, b)));
    Graph::new(6, pairs.map(|(a, b)| Edge::negative(a, b, w)).collect())
}

/// As [`complete`] on two sides of three, which a bipartite Gram states too.
fn complete_bipartite(w: f64) -> Graph {
    let pairs = (0..3).flat_map(|a| (3..6).map(move |b| (a, b)));
    Graph::new(6, pairs.map(|(a, b)| Edge::negative(a, b, w)).collect())
}

/// `M x = b` scaled by `w` has solution `x / w`, so `w · x(w)` is invariant for any scale-free method at a fixed seed: an absolute floor shows as a deviation of order one.
fn assert_invariant_under_scaling(
    entry: Entry,
    fixture: fn(f64) -> Graph,
    types: Types,
    exponents: &[i32],
) {
    // Measured at most 290ε, flat across exponents; a floor that ignores the scale deviates by order one.
    let tolerance = 1024.0 * types.epsilon();
    let solve = |w: f64| {
        let graph = fixture(w);
        let reduced = entry.reduce(&graph, types).expect("states the fixture");
        let solutions = modes_for(&graph).into_iter().map(move |config| {
            let x = reduced.factorize(&config).solve(&RHS);
            (config, x.iter().map(|&x| x * w).collect::<Vec<f64>>())
        });
        solutions.collect::<Vec<_>>()
    };
    let reference = solve(1.0);
    for &exponent in exponents {
        for ((config, scaled), (_, want)) in solve(10f64.powi(exponent)).iter().zip(&reference) {
            for (index, (&scaled, &want)) in scaled.iter().zip(want).enumerate() {
                let deviation = ((scaled - want) / want).abs();
                assert!(
                    deviation < tolerance,
                    "{types:?} {config:?} w=1e{exponent}: x[{index}] deviates by {deviation:e}"
                );
            }
        }
    }
}

const F64_EXPONENTS: [i32; 20] = [
    -290, -200, -100, -30, -16, -15, -14, -5, -1, 21, 23, 31, 32, 37, 43, 50, 100, 152, 200, 300,
];
const F32_EXPONENTS: [i32; 14] = [-30, -20, -12, -8, -7, -6, -5, -2, -1, 7, 10, 12, 20, 30];

/// Each scalar over the exponents its solution is representable at.
fn assert_invariant_on(entry: Entry, fixture: fn(f64) -> Graph) {
    for types in Types::ALL {
        let exponents = match types {
            Types::F32U32 | Types::F32U64 => &F32_EXPONENTS[..],
            Types::F64U32 | Types::F64U64 => &F64_EXPONENTS[..],
        };
        assert_invariant_under_scaling(entry, fixture, types, exponents);
    }
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn factorization_is_invariant_under_uniform_scaling(#[case] entry: Entry) {
    assert_invariant_on(entry, complete);
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn bipartite_factorization_is_invariant_under_uniform_scaling(#[case] entry: Entry) {
    assert_invariant_on(entry, complete_bipartite);
}
