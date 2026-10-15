#[path = "common/entry.rs"]
mod entry;
#[path = "common/floats.rs"]
mod floats;
#[path = "common/graph.rs"]
mod graph;
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

use approx_chol_next::Config;
use entry::{by_type, Entry, Types};
use graph::{Edge, Graph};
use interleaved::interleaved_components;
use modes::{exact, modes_for};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use rstest::rstest;
use runner::check;

/// `x = 0` scores exactly `1`: the weakest bound that demands beating no answer; measured max 0.45.
const RESIDUAL_LIMIT: f64 = 1.0;

/// Up to eight vertices, each pair joined at weight `0..=4`; a bipartite shape keeps only pairs across a cut, and a signed one is `S A S`, balanced: `S` flips one side of a bipartite cut (the incidence Gram), else random vertices.
fn laplacian() -> impl Strategy<Value = Graph> {
    (1usize..=8).prop_flat_map(|n| {
        let pairs = n * (n - 1) / 2;
        let shape = (0..=n, any::<bool>(), prop::bool::weighted(0.25));
        let weights = prop::collection::vec(0u8..=4, pairs);
        let flips = prop::collection::vec(any::<bool>(), n);
        (weights, shape, flips).prop_map(move |(weights, (cut, bipartite, signed), flips)| {
            let flips: Vec<bool> = match bipartite {
                true => (0..n).map(|v| v < cut).collect(),
                false => flips,
            };
            let pairs = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b)));
            let edges = (pairs.zip(weights))
                .filter(|&((a, b), w)| w > 0 && (!bipartite || (a < cut) != (b < cut)))
                .map(|((a, b), w)| Edge {
                    positive: signed && flips[a] != flips[b],
                    ..Edge::negative(a, b, f64::from(w))
                })
                .collect();
            Graph::new(n, edges)
        })
    })
}

/// As [`laplacian`], every row grounded at `1..=5`.
fn sddm() -> impl Strategy<Value = Graph> {
    laplacian().prop_flat_map(|graph| {
        let n = graph.n;
        prop::collection::vec(1u8..=5, n).prop_map(move |ground| Graph {
            ground: ground.into_iter().map(f64::from).collect(),
            ..graph.clone()
        })
    })
}

fn not_stated() -> TestCaseError {
    TestCaseError::reject("the entry point cannot state this graph")
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn n(factor: &entry::Factored) -> usize {
    by_type!(&factor.factor, factor => factor.n())
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn residual_is_bounded(#[case] entry: Entry) {
    let graphs = prop_oneof![3 => laplacian(), 1 => sddm()];
    let rhs = prop::collection::vec(-10.0f64..10.0, 8);
    check((graphs, rhs), |(graph, rhs)| {
        let rhs = graph.project_to_range(&rhs[..graph.n]);
        // `relative_residual` divides by `‖b‖`.
        prop_assume!(norm(&rhs) > 1e-9);
        for types in Types::ALL {
            let reduced = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            for config in modes_for(&graph) {
                let x = reduced.factorize(&config).solve(&rhs);
                // A non-finite solve gives a non-finite ratio, so this subsumes a finiteness check.
                let relative = graph.relative_residual(&x, &rhs);
                prop_assert!(
                    relative < RESIDUAL_LIMIT,
                    "{types:?} {config:?}: relative residual {relative:e}"
                );
            }
        }
        Ok(())
    });
}

/// A balanced graph without surplus floats on every entry point, signed or not.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn laplacian_factor_has_input_dimension_and_floats(#[case] entry: Entry) {
    check(laplacian(), |graph| {
        for types in Types::ALL {
            let reduced = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let floats = reduced.floats();
            prop_assert!(floats.iter().all(|&f| f), "{types:?}: {floats:?}");
            for config in modes_for(&graph) {
                prop_assert_eq!(n(&reduced.factorize(&config)), graph.n);
            }
        }
        Ok(())
    });
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
fn sddm_factor_is_grounded_and_solves_finitely(#[case] entry: Entry) {
    check(sddm(), |graph| {
        let rhs: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin()).collect();
        for types in Types::ALL {
            let reduced = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let floats = reduced.floats();
            prop_assert!(floats.iter().all(|&f| !f), "{types:?}: {floats:?}");
            for config in modes_for(&graph) {
                let factor = reduced.factorize(&config);
                prop_assert_eq!(n(&factor), graph.n);
                let x = factor.solve(&rhs);
                prop_assert!(
                    x.iter().all(|x| x.is_finite()),
                    "{types:?} {config:?}: {x:?}"
                );
            }
        }
        Ok(())
    });
}

/// Two reductions and factorizations of one input at one seed agree bit for bit.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn deterministic_with_fixed_seed(#[case] entry: Entry) {
    let graphs = prop_oneof![3 => laplacian(), 1 => sddm()];
    check(graphs, |graph| {
        let rhs: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin()).collect();
        for types in Types::ALL {
            let first = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let second = entry.reduce(&graph, types).expect("states it again");
            for config in modes_for(&graph) {
                let config = Config { seed: 42, ..config };
                let x1 = first.factorize(&config).solve(&rhs);
                let x2 = second.factorize(&config).solve(&rhs);
                let bits = |x: &[f64]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                prop_assert_eq!(bits(&x1), bits(&x2), "{:?} {:?}", types, config);
            }
        }
        Ok(())
    });
}

/// Class `0` alone carries surplus, so a grounded block sits among floating ones; reading it as floating solves the wrong system on its rows.
fn one_grounded_component() -> impl Strategy<Value = Graph> {
    interleaved_components().prop_flat_map(|(graph, parts)| {
        let n = graph.n;
        prop::collection::vec(1u8..=5, n).prop_map(move |surplus| {
            let ground = (surplus.iter().enumerate())
                .map(|(v, &s)| if v % parts == 0 { f64::from(s) } else { 0.0 })
                .collect();
            Graph {
                ground,
                ..graph.clone()
            }
        })
    })
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
fn a_grounded_block_beside_floating_ones_solves_its_own_rows(#[case] entry: Entry) {
    check(one_grounded_component(), |graph| {
        let raw: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin() * 3.0).collect();
        let rhs = graph.project_to_range(&raw);
        for types in Types::ALL {
            let reduced = entry.reduce(&graph, types).ok_or_else(not_stated)?;
            let x = reduced.factorize(&exact()).solve(&rhs);
            let residual = graph.relative_residual(&x, &rhs);
            // Measured at most 3.4ε; a grounded block solved as floating misses by order one.
            prop_assert!(
                residual < 64.0 * types.epsilon(),
                "{types:?}: grounded block solved as floating: residual {residual:e}"
            );
        }
        Ok(())
    });
}
