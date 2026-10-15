#[path = "common/cases.rs"]
mod cases;
#[path = "common/dense.rs"]
mod dense;
#[path = "common/dimension.rs"]
mod dimension;
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
#[path = "common/sparse.rs"]
mod sparse;
#[path = "common/spectrum.rs"]
mod spectrum;

use std::ops::RangeInclusive;

use approx_chol_next::Config;
use cases::cases;
use dense::dense;
use entry::{Entry, Types};
use graph::{Edge, Graph};
use interleaved::interleaved_components;
use modes::Mode;
use proptest::prelude::*;
use runner::{check, stated_by};
use sparse::Sparse;

/// Rounding alone: what an exact factor leaves of `λ = 1`.
fn exact(types: Types) -> RangeInclusive<f64> {
    let slack = 1024.0 * types.epsilon();
    1.0 - slack..=1.0 + slack
}

/// Sparse holds at edge weights 0–4 (the generator's range); larger ratios measured up to λ 9.2.
fn spectrum_bounds(mode: Mode, types: Types) -> RangeInclusive<f64> {
    match mode {
        Mode::Dense => exact(types),
        Mode::Sparse(_) => 1.0 / 3.0..=6.0,
    }
}

/// The spectrum if it leaves its bounds, as a non-finite solve always does.
fn spectrum_violation(entry: Entry, graph: &Graph, types: Types, mode: Mode) -> Option<String> {
    let reduced = entry.reduce(graph, types);
    let lambda = reduced.factorize(&mode.config()).spectrum(graph);
    let bounds = spectrum_bounds(mode, types);
    let inside = lambda.iter().all(|l| bounds.contains(l));
    (!inside).then(|| format!("{types:?} {mode:?}: λ = {lambda:?}"))
}

/// Up to eight vertices at weights `0..=4`, some bipartite, some signed as a balanced `S A S`.
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

/// Laplacians and SDDMs alike.
fn balanced() -> impl Strategy<Value = Graph> {
    prop_oneof![3 => laplacian(), 1 => sddm()]
}

/// As `graphs`, every edge's sign drawn at random, so blocks frustrate and factor on two sheets.
fn frustrated(graphs: impl Strategy<Value = Graph>) -> impl Strategy<Value = Graph> {
    graphs.prop_flat_map(|graph| {
        prop::collection::vec(any::<bool>(), graph.edges.len()).prop_map(move |signs| {
            let edges = (graph.edges.iter().zip(signs))
                .map(|(&edge, positive)| Edge { positive, ..edge })
                .collect();
            Graph {
                edges,
                ..graph.clone()
            }
        })
    })
}

cases! {
    fn factor_spectrum_on_the_range_is_bounded(
        route in routes(all),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let graphs = stated_by(entry, prop_oneof![balanced(), frustrated(balanced())]);
        check(graphs, |graph| {
            let violation = spectrum_violation(entry, &graph, types, mode);
            prop_assert!(violation.is_none(), "{}", violation.unwrap_or_default());
            Ok(())
        });
    }
}

/// A 3-star sampled as a path: `λ_max` 2.37 at seed 7 under a sampled complement.
fn three_star_tree() -> Graph {
    let edges = [
        (0, 4, 1.0),
        (1, 5, 1.0),
        (2, 3, 3.0),
        (2, 4, 3.0),
        (2, 5, 3.0),
    ];
    Graph::new(6, edges.map(|(a, b, w)| Edge::negative(a, b, w)).to_vec())
}

/// Sides of four each: `λ_max` 2.57 at seed 7 under a sampled complement.
fn tied_eight_vertices() -> Graph {
    let edges = [
        (0, 4, 4.0),
        (0, 5, 1.0),
        (0, 6, 4.0),
        (0, 7, 4.0),
        (1, 5, 1.0),
        (2, 4, 2.0),
        (2, 5, 1.0),
        (2, 6, 3.0),
        (2, 7, 4.0),
        (3, 4, 1.0),
        (3, 5, 2.0),
    ];
    Graph::new(8, edges.map(|(a, b, w)| Edge::negative(a, b, w)).to_vec())
}

cases! {
    #[case::three_star_tree(three_star_tree())]
    #[case::tied_eight_vertices(tied_eight_vertices())]
    fn a_sampled_complement_past_twice_its_clique_keeps_the_spectrum_bounded(
        #[case] graph: Graph,
        route in routes(all),
        types in types(all),
    ) {
        let (entry, mode) = route;
        assert_eq!(spectrum_violation(entry, &graph, types, mode), None);
    }
}

/// A path or a cycle, relabelled, at weights `1..=4`; a signed odd cycle frustrates.
fn path_or_cycle() -> impl Strategy<Value = Graph> {
    (2usize..=8, any::<bool>()).prop_flat_map(|(n, cycle)| {
        let m = if cycle && n >= 3 { n } else { n - 1 };
        let edges = prop::collection::vec((1u8..=4, any::<bool>()), m);
        let order = Just((0..n).collect::<Vec<_>>()).prop_shuffle();
        (edges, order, any::<bool>()).prop_map(move |(edges, order, signed)| {
            let edges = (edges.into_iter().enumerate())
                .map(|(k, (w, flip))| Edge {
                    positive: signed && flip,
                    ..Edge::negative(order[k], order[(k + 1) % n], f64::from(w))
                })
                .collect();
            Graph::new(n, edges)
        })
    })
}

cases! {
    /// Every star has at most two edges, whose clique is one edge: every mode is exact.
    fn a_path_or_cycle_factors_exactly_in_every_mode(route in routes(all), types in types(all)) {
        let (entry, mode) = route;
        let (config, bounds) = (mode.config(), exact(types));
        check(stated_by(entry, path_or_cycle()), |graph| {
            let lambda = entry.reduce(&graph, types).factorize(&config).spectrum(&graph);
            prop_assert!(
                lambda.iter().all(|l| bounds.contains(l)),
                "{types:?} {mode:?}: λ = {lambda:?}"
            );
            Ok(())
        });
    }
}

cases! {
    /// A balanced graph without surplus floats on every entry point, signed or not.
    fn laplacian_factor_has_input_dimension_and_floats(entry in entries(all), types in types(all)) {
        check(stated_by(entry, laplacian()), |graph| {
            let reduced = entry.reduce(&graph, types);
            let floats = reduced.floats();
            prop_assert!(floats.iter().all(|&f| f), "{types:?}: {floats:?}");
            prop_assert_eq!(reduced.factorize(&Config::default()).n(), graph.n);
            Ok(())
        });
    }
}

cases! {
    fn sddm_factor_is_grounded_and_solves_finitely(
        route in routes(grounded),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let config = mode.config();
        check(stated_by(entry, sddm()), |graph| {
            let rhs: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin()).collect();
            let reduced = entry.reduce(&graph, types);
            let floats = reduced.floats();
            prop_assert!(floats.iter().all(|&f| !f), "{types:?}: {floats:?}");
            let factor = reduced.factorize(&config);
            prop_assert_eq!(factor.n(), graph.n);
            let x = factor.solve(&rhs);
            prop_assert!(
                x.iter().all(|x| x.is_finite()),
                "{types:?} {config:?}: {x:?}"
            );
            Ok(())
        });
    }
}

cases! {
    /// Two reductions and factorizations of one input at one seed agree bit for bit.
    fn deterministic_with_fixed_seed(route in routes(all), types in types(all)) {
        let (entry, mode) = route;
        let graphs = stated_by(entry, balanced());
        let config = Config {
            seed: 42,
            ..mode.config()
        };
        check(graphs, |graph| {
            let rhs: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin()).collect();
            let first = entry.reduce(&graph, types);
            let second = entry.reduce(&graph, types);
            let x1 = first.factorize(&config).solve(&rhs);
            let x2 = second.factorize(&config).solve(&rhs);
            let bits = |x: &[f64]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            prop_assert_eq!(bits(&x1), bits(&x2), "{:?} {:?}", types, config);
            Ok(())
        });
    }
}

/// Only class `0` carries surplus; solving its block as floating misses its rows.
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

cases! {
    fn a_grounded_block_beside_floating_ones_solves_its_own_rows(
        entry in entries(grounded),
        types in types(all),
    ) {
        check(stated_by(entry, one_grounded_component()), |graph| {
            let raw: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin() * 3.0).collect();
            let rhs = graph.project_to_range(&raw);
            let reduced = entry.reduce(&graph, types);
            let x = reduced.factorize(&dense()).solve(&rhs);
            let residual = graph.relative_residual(&x, &rhs);
            // Measured at most 3.4ε; a grounded block solved as floating misses by order one.
            prop_assert!(
                residual < 64.0 * types.epsilon(),
                "{types:?}: grounded block solved as floating: residual {residual:e}"
            );
            Ok(())
        });
    }
}
