#[path = "common/dense.rs"]
mod dense;
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
use dense::dense;
use entry::{by_type, Entry, Types};
use graph::{Edge, Graph};
use interleaved::interleaved_components;
use modes::Mode;
use proptest::prelude::*;
use rstest::rstest;
use runner::{check, stated_by};
use sparse::Sparse;

/// Dense is exact to rounding (measured ≤ 47ε); sparse measured `[0.43, 3.94]`; bounding `λ`, not `κ`, catches a dropped scale.
fn spectrum_bounds(mode: Mode, types: Types) -> RangeInclusive<f64> {
    match mode {
        Mode::Dense => {
            let slack = 1024.0 * types.epsilon();
            1.0 - slack..=1.0 + slack
        }
        Mode::Sparse(_) => 1.0 / 3.0..=6.0,
    }
}

/// The spectrum past its bounds, if it is; a non-finite solve gives a non-finite `λ`, which no range contains.
fn spectrum_violation(entry: Entry, graph: &Graph, types: Types, mode: Mode) -> Option<String> {
    let reduced = entry.reduce(graph, types);
    let lambda = reduced.factorize(&mode.config()).spectrum(graph);
    let bounds = spectrum_bounds(mode, types);
    let inside = lambda.iter().all(|l| bounds.contains(l));
    (!inside).then(|| format!("{types:?} {mode:?}: λ = {lambda:?}"))
}

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

fn n(factor: &entry::Factored) -> usize {
    by_type!(&factor.factor, factor => factor.n())
}

#[rstest]
fn factor_spectrum_on_the_range_is_bounded(
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::Laplacian,
        Entry::GroundedLaplacian,
        Entry::SignedGraph
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
    let graphs = stated_by(entry, prop_oneof![balanced(), frustrated(balanced())]);
    check(graphs, |graph| {
        let violation = spectrum_violation(entry, &graph, types, mode);
        prop_assert!(violation.is_none(), "{}", violation.unwrap_or_default());
        Ok(())
    });
}

/// A 3-star sampled as a path: one apply's residual 1.04 at seed 7, `λ_max` 2.37 under a sampled complement.
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

/// Sides of four each: one apply's residual 1.53 at seed 7, `λ_max` 2.57 under a sampled complement.
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

#[rstest]
#[case::three_star_tree(three_star_tree())]
#[case::tied_eight_vertices(tied_eight_vertices())]
fn a_sampled_complement_past_twice_its_clique_keeps_the_spectrum_bounded(
    #[case] graph: Graph,
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::Laplacian,
        Entry::GroundedLaplacian,
        Entry::SignedGraph
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
    assert_eq!(spectrum_violation(entry, &graph, types, mode), None);
}

/// A frustrated `K_5`, heavy around vertex 4: `λ_max` 3.94 under AC at seed 7, so a doubled cover solve leaves the range.
fn frustrated_heavy_k5() -> Graph {
    let edges = [
        (0, 1, 1.0, false),
        (0, 2, 4096.0, false),
        (0, 3, 1.0, true),
        (0, 4, 4096.0, true),
        (1, 2, 4096.0, true),
        (1, 3, 1.0, true),
        (1, 4, 4096.0, false),
        (2, 3, 1.0, true),
        (2, 4, 4096.0, false),
        (3, 4, 4096.0, false),
    ];
    let edges = edges.map(|(a, b, w, positive)| Edge {
        positive,
        ..Edge::negative(a, b, w)
    });
    Graph::new(5, edges.to_vec())
}

#[rstest]
fn a_frustrated_cover_keeps_the_spectrum_bounded(
    #[values(Entry::Csr, Entry::UpperCsr, Entry::Gram, Entry::SignedGraph)] entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
    let violation = spectrum_violation(entry, &frustrated_heavy_k5(), types, mode);
    assert_eq!(violation, None);
}

/// A balanced graph without surplus floats on every entry point, signed or not.
#[rstest]
fn laplacian_factor_has_input_dimension_and_floats(
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::Laplacian,
        Entry::GroundedLaplacian,
        Entry::SignedGraph
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
    check(stated_by(entry, laplacian()), |graph| {
        let reduced = entry.reduce(&graph, types);
        let floats = reduced.floats();
        prop_assert!(floats.iter().all(|&f| f), "{types:?}: {floats:?}");
        prop_assert_eq!(n(&reduced.factorize(&mode.config())), graph.n);
        Ok(())
    });
}

#[rstest]
fn sddm_factor_is_grounded_and_solves_finitely(
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::GroundedLaplacian
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
    let config = mode.config();
    check(stated_by(entry, sddm()), |graph| {
        let rhs: Vec<f64> = (0..graph.n).map(|i| (i as f64 * 1.7).sin()).collect();
        let reduced = entry.reduce(&graph, types);
        let floats = reduced.floats();
        prop_assert!(floats.iter().all(|&f| !f), "{types:?}: {floats:?}");
        let factor = reduced.factorize(&config);
        prop_assert_eq!(n(&factor), graph.n);
        let x = factor.solve(&rhs);
        prop_assert!(
            x.iter().all(|x| x.is_finite()),
            "{types:?} {config:?}: {x:?}"
        );
        Ok(())
    });
}

/// Two reductions and factorizations of one input at one seed agree bit for bit.
#[rstest]
fn deterministic_with_fixed_seed(
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::Laplacian,
        Entry::GroundedLaplacian,
        Entry::SignedGraph
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::AcComplementAc),
        Mode::Sparse(Sparse::AcComplementAc2),
        Mode::Sparse(Sparse::Ac2ExactComplement),
        Mode::Sparse(Sparse::Ac2ComplementAc),
        Mode::Sparse(Sparse::Ac2ComplementAc2)
    )]
    mode: Mode,
) {
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
fn a_grounded_block_beside_floating_ones_solves_its_own_rows(
    #[values(
        Entry::Csr,
        Entry::UpperCsr,
        Entry::Gram,
        Entry::BipartiteGram,
        Entry::GroundedLaplacian
    )]
    entry: Entry,
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
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
