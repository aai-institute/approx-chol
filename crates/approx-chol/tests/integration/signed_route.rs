use approx_chol::input::{CsrRef, Side};
use approx_chol::{factorize_with, ApproxInverse, Config, CoverSurplus, FrustratedRoute};
use approx_chol::{ScalingPolicy, SchurMode};
use rstest::rstest;

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::cycle::Cycle;
use crate::common::dense_residual::{apply, max_abs_diff};
use crate::common::frustrated::{factors, is_signed};
use crate::common::graph::{Edge, Graph};
use crate::common::grid::light_row;
use crate::common::lcg::Lcg;
use crate::common::pcg::pcg_iterations;
use crate::common::solve::Solve;
use crate::common::sparse::clique_tree;

/// Every block approximate, frustrated ones signed.
fn signed() -> Config {
    Config {
        frustrated: FrustratedRoute::Signed,
        exact_max_dim: 0,
        ..Config::default()
    }
}

fn factor(graph: &Graph, config: Config) -> ApproxInverse {
    let (ptrs, columns, values) = graph.csr(false);
    let ptrs: Vec<u32> = ptrs.into_iter().map(|p| p as u32).collect();
    let columns: Vec<u32> = columns.into_iter().map(|c| c as u32).collect();
    let csr = CsrRef::new(graph.n, &ptrs, &columns, &values).unwrap();
    factorize_with(csr, config, ScalingPolicy::default()).unwrap()
}

/// A cycle of weights `1 + (i mod 3) / 4`, its first edge positive and no surplus: frustrated, so nonsingular.
fn frustrated_cycle(n: usize) -> Graph {
    let edges = (0..n).map(|i| Edge {
        a: i,
        b: (i + 1) % n,
        weight: 1.0 + (i % 3) as f64 / 4.0,
        positive: i == 0,
    });
    Graph::new(n, edges.collect())
}

/// A spanning path and `3n` random edges, a third of them positive, log-uniform over `2^±6`, every tenth vertex grounded.
fn frustrated_random(n: usize, seed: u64) -> Graph {
    let mut lcg = Lcg(seed);
    let weight = |lcg: &mut Lcg| ((lcg.next_u64() % 1200) as f64 / 100.0 - 6.0).exp2();
    let mut edges: Vec<Edge> = (1..n)
        .map(|v| Edge::negative(v - 1, v, weight(&mut lcg)))
        .collect();
    while edges.len() < 4 * n {
        let (a, b) = (
            (lcg.next_u64() % n as u64) as usize,
            (lcg.next_u64() % n as u64) as usize,
        );
        if a != b {
            let w = weight(&mut lcg);
            edges.push(Edge {
                positive: lcg.next_u64() % 3 == 0,
                ..Edge::negative(a, b, w)
            });
        }
    }
    let mut graph = Graph::new(n, edges);
    (0..n).step_by(10).for_each(|v| graph.ground[v] = 0.5);
    graph
}

/// A frustrated cycle's stars have at most two neighbors, so its signed elimination samples nothing: it solves as the dense arm does.
#[test]
fn a_frustrated_cycle_without_surplus_solves_exactly() {
    for n in [3, 4, 9, 30] {
        let graph = frustrated_cycle(n);
        let want: Vec<f64> = (0..n).map(|i| ((3 * i + 1) as f64).sin()).collect();
        let b = graph.apply(&want);
        let signed = factor(&graph, signed()).solve(&b).unwrap();
        let dense = factor(
            &graph,
            Config {
                exact_max_dim: n,
                ..Config::default()
            },
        )
        .solve(&b)
        .unwrap();
        let scale = dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let off = max_abs_diff(&signed, &dense);
        assert!(off <= 1e-13 * scale, "n = {n}: off by {off:e} of {scale:e}");
        assert!(graph.relative_residual(&signed, &b) <= 1e-13, "n = {n}");
    }
}

/// `g`'s [`factors`] under `config`, each with its `A` or `S` dense.
fn bipartite_factors(g: &Bipartite, config: Config) -> Vec<(String, ApproxInverse, Vec<Vec<f64>>)> {
    let matrices = [
        g.dense(),
        g.complement(Side::Left),
        g.complement(Side::Right),
    ];
    (factors(g, config).into_iter().zip(matrices))
        .map(|((built, factor), a)| (built, factor, a))
        .collect()
}

/// A frustrated bipartite cycle without surplus: each eliminated star holds two neighbors, so its complement, exact or sampled, is a frustrated cycle, which the signed elimination solves exactly, whole or as a complement; at `m = 2` its two stars cancel to a diagonal.
#[test]
fn a_frustrated_bipartite_cycle_without_surplus_solves_exactly() {
    for m in [2, 3, 5, 16] {
        let cycle = Cycle {
            cycle: m,
            pendants: 0,
            wide: 1,
            right_pendants: 0,
        };
        let g = cycle.bipartite();
        for complement in [SchurMode::Exact, SchurMode::Same] {
            let config = Config {
                complement,
                ..signed()
            };
            for (built, factor, a) in bipartite_factors(&g, config) {
                let name = format!("m = {m}, {complement:?}, {built}");
                assert!(is_signed(&factor), "{name}: signed");
                let want: Vec<f64> = (0..a.len()).map(|i| ((3 * i + 1) as f64).sin()).collect();
                let x = factor.solve(&apply(&a, &want)).unwrap();
                let off = max_abs_diff(&x, &want);
                assert!(off <= 1e-13, "{name}: off by {off:e}");
            }
        }
    }
}

/// Both routes precondition a frustrated bipartite input, whole and onto each side, its complement exact or sampled.
#[rstest]
fn both_routes_converge_on_a_frustrated_bipartite_input(
    #[values(SchurMode::Exact, SchurMode::Same)] complement: SchurMode,
    #[values(1, 2)] copies: u32,
    #[values(FrustratedRoute::Signed, FrustratedRoute::Cover(CoverSurplus::Auto))]
    route: FrustratedRoute,
) {
    let config = Config {
        frustrated: route,
        sampling: clique_tree(copies),
        complement,
        exact_max_dim: 0,
        ..Config::default()
    };
    let signed = matches!(route, FrustratedRoute::Signed);
    for (built, factor, a) in bipartite_factors(&Bipartite::frustrated(), config) {
        assert_eq!(is_signed(&factor), signed, "{built}");
        let want: Vec<f64> = (0..a.len()).map(|i| ((7 * i + 3) as f64).sin()).collect();
        let b = apply(&a, &want);
        let iterations = pcg_iterations(
            |x| apply(&a, x),
            |r| factor.solve(r).unwrap(),
            &b,
            1e-8,
            200,
        );
        assert!(
            iterations.is_some(),
            "{route:?}, {complement:?} at {copies} copies, {built}: did not converge"
        );
    }
}

#[rstest]
#[case::grid(light_row(24, 1.0))]
#[case::cycle(frustrated_cycle(500))]
#[case::random(frustrated_random(800, 7))]
fn both_routes_converge_on_frustrated_inputs(
    #[case] graph: Graph,
    #[values(1, 2)] copies: u32,
    #[values(FrustratedRoute::Signed, FrustratedRoute::Cover(CoverSurplus::Auto))]
    route: FrustratedRoute,
) {
    let config = Config {
        frustrated: route,
        sampling: clique_tree(copies),
        exact_max_dim: 0,
        ..Config::default()
    };
    let factor = factor(&graph, config);
    let want: Vec<f64> = (0..graph.n).map(|i| ((7 * i + 3) as f64).sin()).collect();
    let b = graph.apply(&want);
    let iterations = pcg_iterations(
        |x| graph.apply(x),
        |r| factor.solve(r).unwrap(),
        &b,
        1e-8,
        200,
    );
    assert!(
        iterations.is_some(),
        "{route:?} at {copies} copies did not converge"
    );
}

/// A balanced input folds before any route applies, so `Signed` leaves it as `Cover` does.
#[test]
fn a_switched_balanced_input_solves_alike_under_both_routes() {
    let mut graph = frustrated_random(300, 3);
    graph
        .edges
        .iter_mut()
        .for_each(|e| e.positive = (e.a + e.b) % 2 == 1);
    let b: Vec<f64> = (0..graph.n).map(|i| (i % 5) as f64 - 2.0).collect();
    let cover = factor(
        &graph,
        Config {
            exact_max_dim: 0,
            ..Config::default()
        },
    );
    let signed = factor(&graph, signed());
    assert!(!is_signed(&signed), "a balanced block folds");
    assert_eq!(
        bits(&signed.solve(&b).unwrap()),
        bits(&cover.solve(&b).unwrap())
    );
}

/// Three frustrated components factored in parallel solve to the same bits at any thread count.
#[test]
fn a_signed_build_is_thread_invariant() {
    let parts = [
        frustrated_random(400, 1),
        frustrated_random(250, 2),
        frustrated_cycle(90),
    ];
    let n = parts.iter().map(|g| g.n).sum();
    let mut graph = Graph::new(n, Vec::new());
    let mut offset = 0;
    for part in &parts {
        for edge in &part.edges {
            graph.edges.push(Edge {
                a: edge.a + offset,
                b: edge.b + offset,
                ..*edge
            });
        }
        (0..part.n).for_each(|v| graph.ground[offset + v] = part.ground[v]);
        offset += part.n;
    }
    let b: Vec<f64> = (0..n)
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let at = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| bits(&factor(&graph, signed()).solve(&b).unwrap()))
    };
    let serial = at(1);
    for threads in [2, 7] {
        assert!(at(threads) == serial, "{threads} threads differ from 1");
    }
}

#[cfg(feature = "serde")]
mod serde {
    use super::*;

    /// A signed frustrated triangle's payload as JSON.
    fn triangle() -> serde_json::Value {
        let factor = factor(&frustrated_cycle(3), signed());
        assert!(is_signed(&factor));
        serde_json::to_value(&factor).unwrap()
    }

    fn cholesky(payload: &mut serde_json::Value) -> &mut serde_json::Value {
        &mut payload[1][1]["U32"][0]["cholesky"]
    }

    #[test]
    fn a_signed_factor_round_trips() {
        let payload = triangle();
        let back: ApproxInverse = serde_json::from_value(payload.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), payload);
    }

    #[test]
    fn a_signed_factor_past_its_index_width_is_corrupt() {
        let mut payload = triangle();
        cholesky(&mut payload)["n"] = serde_json::json!(u64::from(u32::MAX) + 1);
        assert!(serde_json::from_value::<ApproxInverse>(payload).is_err());
    }

    #[test]
    fn a_signed_column_past_one_is_corrupt() {
        let mut fine = triangle();
        let shares = &mut cholesky(&mut fine)["triangular"]["Signed"]["coefficients"];
        assert!(
            shares
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c.as_f64().unwrap() < 0.0),
            "signed shares decode"
        );
        let mut payload = triangle();
        let shares = &mut cholesky(&mut payload)["triangular"]["Signed"]["coefficients"];
        shares[0] = serde_json::json!(-1.5);
        assert!(serde_json::from_value::<ApproxInverse>(fine).is_ok());
        assert!(serde_json::from_value::<ApproxInverse>(payload).is_err());
    }
}
