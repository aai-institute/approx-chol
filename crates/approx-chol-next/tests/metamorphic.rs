//! Equivariance holds on the exact mode alone: relabeling redraws the sampler, which moved sampled solutions 6-37%.

#[path = "common/entry.rs"]
mod entry;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/interleaved.rs"]
mod interleaved;
#[path = "common/modes.rs"]
mod modes;
#[path = "common/range.rs"]
mod range;
#[path = "common/residual.rs"]
mod residual;
#[path = "common/runner.rs"]
mod runner;

use entry::{Entry, Types};
use graph::{Edge, Graph};
use grid::grid;
use interleaved::interleaved_components;
use modes::{exact, modes_for};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use rstest::rstest;
use runner::check;

/// `A'[p[i]][p[j]] = A[i][j]`.
fn permuted(graph: &Graph, p: &[usize]) -> Graph {
    let edges = (graph.edges.iter())
        .map(|edge| Edge {
            a: p[edge.a],
            b: p[edge.b],
            ..*edge
        })
        .collect();
    let mut ground = vec![0.0; graph.n];
    (graph.ground.iter().enumerate()).for_each(|(v, &g)| ground[p[v]] = g);
    Graph {
        ground,
        ..Graph::new(graph.n, edges)
    }
}

fn interleaved_case() -> impl Strategy<Value = (Graph, Vec<f64>, Vec<usize>)> {
    interleaved_components().prop_flat_map(|(graph, _parts)| {
        let n = graph.n;
        (
            Just(graph),
            prop::collection::vec(-10.0f64..10.0, n),
            Just((0..n).collect::<Vec<_>>()).prop_shuffle(),
        )
    })
}

fn not_stated() -> TestCaseError {
    TestCaseError::reject("the entry point cannot state this graph")
}

/// `(P A Pᵀ)(P x) = P b`; interleaved components drive a non-identity block order through the solve.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn permuting_interleaved_components_permutes_the_solution(#[case] entry: Entry) {
    check(interleaved_case(), |(graph, rhs, p)| {
        let mut permuted_rhs = vec![0.0; rhs.len()];
        (rhs.iter().enumerate()).for_each(|(v, &value)| permuted_rhs[p[v]] = value);
        let relabeled = permuted(&graph, &p);
        for types in Types::ALL {
            let base = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let got = entry
                .reduce(&relabeled, types)
                .expect("a relabeling states alike");
            let base = base.factorize(&exact()).solve(&rhs);
            let got = got.factorize(&exact()).solve(&permuted_rhs);
            let scale = base.iter().fold(1.0f64, |scale, x| scale.max(x.abs()));
            for (v, &want) in base.iter().enumerate() {
                let got = got[p[v]];
                // Measured at most 6.3ε·‖x‖∞; a broken permutation is off by order one.
                prop_assert!(
                    (got - want).abs() <= 64.0 * types.epsilon() * scale,
                    "{types:?} x[{v}] -> x'[{}]: {got:e} vs {want:e} (p={p:?})",
                    p[v]
                );
            }
        }
        Ok(())
    });
}

/// Equivariance survives anything wrong the same way both times; this is the direct claim that the components are solved.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn interleaved_components_are_solved_not_just_relabelled_consistently(#[case] entry: Entry) {
    check(interleaved_case(), |(graph, rhs, _p)| {
        let rhs = graph.project_to_range(&rhs);
        prop_assume!(rhs.iter().map(|value| value * value).sum::<f64>().sqrt() > 1e-9);
        for types in Types::ALL {
            let reduced = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let x = reduced.factorize(&exact()).solve(&rhs);
            let relative = graph.relative_residual(&x, &rhs);
            // Measured at most 9ε.
            prop_assert!(
                relative < 64.0 * types.epsilon(),
                "{types:?}: components left residual {relative:e}"
            );
        }
        Ok(())
    });
}

/// Dyadic, so `value + shift` is exact while the shift stays below the scalar's spare mantissa bits.
fn assert_a_constant_leaves_the_solution(entry: Entry, types: [Types; 2], shift: f64) {
    let graph = grid(100, 100);
    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs: Vec<f64> = (0..graph.n)
        .map(|i| (i * 37 % 2001) as f64 / 1024.0 - 1.0)
        .collect();
    let shifted: Vec<f64> = rhs.iter().map(|value| value + shift).collect();
    for types in types {
        let reduced = entry.reduce(&graph, types).expect("states a grid");
        for config in modes_for(&graph) {
            let factor = reduced.factorize(&config);
            let (base, got) = (factor.solve(&rhs), factor.solve(&shifted));
            let difference: Vec<f64> = got.iter().zip(&base).map(|(g, b)| g - b).collect();
            let moved = norm(&difference) / norm(&base);
            assert!(
                moved < 1e-3,
                "{types:?} {config:?}: the constant moved the solution by {moved:e}"
            );
        }
    }
}

/// A floating block solves `b` with its null component projected out, so adding a constant to `b` must leave `x`; only the solve's own sums can lose it.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn a_constant_added_to_a_floating_rhs_leaves_the_solution(#[case] entry: Entry) {
    assert_a_constant_leaves_the_solution(entry, [Types::F64U32, Types::F64U64], 2f64.powi(30));
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn a_constant_added_to_a_floating_f32_rhs_leaves_the_solution(#[case] entry: Entry) {
    assert_a_constant_leaves_the_solution(entry, [Types::F32U32, Types::F32U64], 2f64.powi(12));
}
