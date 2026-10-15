#[path = "common/entry.rs"]
mod entry;
#[path = "common/floats.rs"]
mod floats;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/modes.rs"]
mod modes;
#[path = "common/residual.rs"]
mod residual;

use approx_chol_next::{
    CliqueRule, Config, Factor, Index, InnerParallelism, Multiplicity, Real, SolveError, SplitCount,
};
use entry::{by_type, cast, Entry, Factored, Types};
use graph::{Edge, Graph};
use modes::modes_for;
use rstest::rstest;

const F64: [Types; 2] = [Types::F64U32, Types::F64U64];
const F32: [Types; 2] = [Types::F32U32, Types::F32U64];

fn assert_close(label: &str, x: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(x.len(), want.len(), "{label}");
    for (i, (&got, &want)) in x.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "{label}: x[{i}] = {got:e}, expected {want:e} ({x:?})"
        );
    }
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

/// Every vertex its own grounded block.
fn diagonal_sddm() -> Graph {
    Graph {
        ground: vec![2.0, 3.0, 5.0],
        ..Graph::new(3, Vec::new())
    }
}

/// `config` with AC2's copy count, wherever it splits, set to `k`.
fn splitting(config: Config, k: u32) -> Config {
    let split = |multiplicity| match multiplicity {
        Multiplicity::Split(_) => Multiplicity::Split(SplitCount::new(k).expect("k ≥ 2")),
        single => single,
    };
    Config {
        approximate: split(config.approximate),
        complement: match config.complement {
            CliqueRule::Sampled(multiplicity) => CliqueRule::Sampled(split(multiplicity)),
            exact => exact,
        },
        ..config
    }
}

/// A grounded singleton is a star at every `k`, so no clique edge is sampled and the solve is `M⁻¹b` to rounding.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
fn sddm_solve_matches_dense_inverse_nonzero_sum_rhs(
    #[case] entry: Entry,
    #[values(2, 3, 7)] k: u32,
) {
    let graph = diagonal_sddm();
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states a diagonal SDDM");
        assert_eq!(reduced.floats(), [false; 3], "{types:?}");
        for config in modes_for(&graph).into_iter().map(|c| splitting(c, k)) {
            let factor = reduced.factorize(&config);
            for b in [[1.0, 2.0, 3.0], [1.0, -2.0, 4.0]] {
                let want: Vec<f64> = b.iter().zip(&graph.ground).map(|(b, d)| b / d).collect();
                let label = format!("{types:?} {config:?}");
                assert_close(&label, &factor.solve(&b), &want, 4.0 * types.epsilon());
            }
        }
    }
}

fn assert_buffers_checked<T: Real, I: Index>(factor: &Factor<T, I>) {
    let n = factor.n();
    for found in [n - 1, n + 1] {
        let error = SolveError::LengthMismatch { expected: n, found };
        assert_eq!(factor.solve(&vec![T::zero(); found]), Err(error));
    }
    let blocks = factor.clone().into_blocks();
    assert!(
        blocks.iter().any(|block| block.scratch_len() > 0),
        "a grounded block solves through scratch"
    );
    for block in blocks {
        let (m, needed) = (block.vertices().len(), block.scratch_len());
        let mut scratch = vec![T::zero(); needed];
        for found in [m - 1, m + 1] {
            let mut x = vec![T::zero(); found];
            let error = SolveError::LengthMismatch { expected: m, found };
            let solved = block.solve_local(&mut x, &mut scratch, InnerParallelism::Forbid);
            assert_eq!(solved, Err(error));
        }
        if needed > 0 {
            let mut x = vec![T::one(); m];
            let short = &mut scratch[..needed - 1];
            let error = SolveError::ScratchTooShort {
                needed,
                found: needed - 1,
            };
            let solved = block.solve_local(&mut x, short, InnerParallelism::Forbid);
            assert_eq!(solved, Err(error));
        }
    }
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
fn every_solve_checks_its_buffer_lengths(#[case] entry: Entry) {
    let graph = diagonal_sddm();
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states a diagonal SDDM");
        for config in modes_for(&graph) {
            by_type!(&reduced.factorize(&config).factor, factor => assert_buffers_checked(factor));
        }
    }
}

/// Each block through `solve_local` on a NaN-filled scratch, scattered back.
fn solve_on_dirty_scratch<T: Real, I: Index>(
    factor: &Factor<T, I>,
    b: &[f64],
    inner: InnerParallelism,
) -> Vec<f64> {
    let mut x = vec![0.0; factor.n()];
    for block in factor.clone().into_blocks() {
        let vertices = block.vertices();
        let mut local: Vec<T> = vertices.iter().map(|&v| cast(b[v.index()])).collect();
        let mut scratch = vec![T::nan(); block.scratch_len() + 1];
        let solved = block.solve_local(&mut local, &mut scratch, inner);
        solved.expect("buffers fit");
        (vertices.iter().zip(&local)).for_each(|(&v, &value)| x[v.index()] = cast(value));
    }
    x
}

fn uses_scratch(factor: &Factored) -> bool {
    by_type!(&factor.factor, factor => factor
        .clone()
        .into_blocks()
        .iter()
        .any(|block| block.scratch_len() > 0))
}

/// NaN reaches the result if any slot, a ground's or a cover's included, is read before it is written.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn dirty_scratch_does_not_change_the_solution(#[case] entry: Entry) {
    let positive = |a, b| Edge {
        positive: true,
        ..Edge::negative(a, b, 1.0)
    };
    let fixtures = [
        (
            "grounded",
            Graph {
                ground: vec![1.0, 1.0],
                ..Graph::new(2, vec![Edge::negative(0, 1, 1.0)])
            },
            true,
        ),
        (
            "frustrated",
            Graph::new(3, vec![positive(0, 1), positive(1, 2), positive(0, 2)]),
            true,
        ),
        // Floating blocks solve in place; this case exercises the gather and scatter.
        (
            "interleaved components",
            Graph::new(
                4,
                vec![Edge::negative(0, 2, 1.0), Edge::negative(1, 3, 1.0)],
            ),
            false,
        ),
    ];
    let mut stated = 0;
    for (label, graph, through_scratch) in fixtures {
        let rhs: Vec<f64> = (0..graph.n).map(|i| i as f64 - 1.5).collect();
        for types in Types::ALL {
            let Some(reduced) = entry.reduce(&graph, types) else {
                continue;
            };
            stated += 1;
            for config in modes_for(&graph) {
                let factor = reduced.factorize(&config);
                if through_scratch {
                    assert!(
                        uses_scratch(&factor),
                        "{label}: the solve goes through scratch"
                    );
                }
                let want = factor.solve(&rhs);
                for inner in [InnerParallelism::Allow, InnerParallelism::Forbid] {
                    let local = factor.labels.to_local(&rhs);
                    let x = by_type!(&factor.factor, f => solve_on_dirty_scratch(f, &local, inner));
                    let x = factor.labels.to_caller(&x);
                    assert_eq!(
                        bits(&x),
                        bits(&want),
                        "{label} {types:?} {config:?} {inner:?}"
                    );
                }
            }
        }
    }
    assert!(stated > 0, "{entry:?} states no fixture");
}

fn path(weight: f64) -> Graph {
    Graph::new(
        4,
        (0..3).map(|i| Edge::negative(i, i + 1, weight)).collect(),
    )
}

/// Scaling a Laplacian by `t` scales its solution by `1/t`, so a factor that drops the scale is wrong by that whole factor.
fn assert_scaled_path_solves(entry: Entry, types: [Types; 2], exponent: i32) {
    let graph = path(10f64.powi(exponent));
    let b = [1.0, 0.0, 0.0, -1.0];
    for types in types {
        let reduced = entry.reduce(&graph, types).expect("states a path");
        for config in modes_for(&graph) {
            let x = reduced.factorize(&config).solve(&b);
            let relative = graph.relative_residual(&x, &b);
            assert!(
                relative < 1e-6,
                "{types:?} {config:?} 1e{exponent}: relative residual {relative:e}"
            );
        }
    }
}

/// The two scalars bottom out at different exponents, so each gets the range its solution is still representable in.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn a_scaled_f64_laplacian_solves_wherever_its_solution_is_representable(
    #[case] entry: Entry,
    #[values(0, -5, -15, -100, -250)] exponent: i32,
) {
    assert_scaled_path_solves(entry, F64, exponent);
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn a_scaled_f32_laplacian_solves_wherever_its_solution_is_representable(
    #[case] entry: Entry,
    #[values(0, -3, -10, -25)] exponent: i32,
) {
    assert_scaled_path_solves(entry, F32, exponent);
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn many_zero_singletons_factor_as_trivial_components(#[case] entry: Entry) {
    let graph = Graph::new(128, Vec::new());
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states a zero matrix");
        assert!(
            reduced.floats().is_empty(),
            "{types:?}: a zero vertex is no block"
        );
        for config in modes_for(&graph) {
            let factor = reduced.factorize(&config);
            assert_eq!(by_type!(&factor.factor, f => f.n()), 128);
            assert_eq!(bits(&factor.solve(&[1.0; 128])), bits(&[0.0; 128]));
        }
    }
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
fn mixed_grounded_and_floating_components_solve_independently(#[case] entry: Entry) {
    let graph = Graph {
        ground: vec![2.0, 0.0, 0.0],
        ..Graph::new(3, vec![Edge::negative(1, 2, 1.0)])
    };
    for types in Types::ALL {
        let reduced = entry
            .reduce(&graph, types)
            .expect("states the mixed system");
        let mut floats = reduced.floats();
        floats.sort();
        assert_eq!(floats, [false, true], "{types:?}");
        for config in modes_for(&graph) {
            let x = reduced.factorize(&config).solve(&[4.0, 1.0, -1.0]);
            let label = format!("{types:?} {config:?}");
            assert_close(&label, &x, &[2.0, 0.5, -0.5], 4.0 * types.epsilon());
        }
    }
}

/// Components `{0, 2}` and `{1, 3}` interleave; an rhs not zero-sum per block has no exact solution, so the second case is the least-squares one.
#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn interleaved_components_solve_in_input_order(#[case] entry: Entry) {
    let edges = vec![Edge::negative(0, 2, 1.0), Edge::negative(1, 3, 1.0)];
    let graph = Graph::new(4, edges);
    let cases = [
        (
            "zero-sum per block",
            [1.0, 2.0, -1.0, -2.0],
            [0.5, 1.0, -0.5, -1.0],
        ),
        (
            "inconsistent",
            [3.0, 5.0, -1.0, -2.0],
            [1.0, 1.75, -1.0, -1.75],
        ),
    ];
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states two edges");
        for config in modes_for(&graph) {
            let factor = reduced.factorize(&config);
            for (label, rhs, want) in cases {
                let label = format!("{label} {types:?} {config:?}");
                assert_close(&label, &factor.solve(&rhs), &want, 8.0 * types.epsilon());
            }
        }
    }
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn singleton_systems_have_defined_solves(#[case] entry: Entry) {
    let fixtures = [
        (
            "zero singleton",
            Graph::new(1, Vec::new()),
            vec![7.0],
            vec![0.0],
        ),
        (
            "positive singleton",
            Graph {
                ground: vec![2.0],
                ..Graph::new(1, Vec::new())
            },
            vec![7.0],
            vec![3.5],
        ),
    ];
    let mut stated = 0;
    for (label, graph, rhs, want) in fixtures {
        for types in Types::ALL {
            let Some(reduced) = entry.reduce(&graph, types) else {
                continue;
            };
            stated += 1;
            for config in modes_for(&graph) {
                let x = reduced.factorize(&config).solve(&rhs);
                let label = format!("{label} {types:?} {config:?}");
                assert_close(&label, &x, &want, types.epsilon());
            }
        }
    }
    assert!(stated > 0, "{entry:?} states no fixture");
}

#[rstest]
#[case::csr(Entry::Csr)]
#[case::upper_csr(Entry::UpperCsr)]
#[case::gram(Entry::Gram)]
#[case::bipartite_gram(Entry::BipartiteGram)]
#[case::laplacian(Entry::Laplacian)]
#[case::grounded_laplacian(Entry::GroundedLaplacian)]
#[case::signed_graph(Entry::SignedGraph)]
fn an_empty_system_solves_to_an_empty_vector(#[case] entry: Entry) {
    let graph = Graph::new(0, Vec::new());
    for types in Types::ALL {
        let reduced = entry.reduce(&graph, types).expect("states an empty matrix");
        for config in modes_for(&graph) {
            assert!(reduced.factorize(&config).solve(&[]).is_empty());
        }
    }
}
