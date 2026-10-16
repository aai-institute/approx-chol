//! Exact mode only: relabeling redraws the sampler, which moved sampled solutions 6-37%.

#[path = "common/cases.rs"]
mod cases;
#[path = "common/dense.rs"]
mod dense;
#[path = "common/entry.rs"]
mod entry;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/interleaved.rs"]
mod interleaved;
#[path = "common/range.rs"]
mod range;
#[path = "common/residual.rs"]
mod residual;
#[path = "common/runner.rs"]
mod runner;
#[path = "common/sparse.rs"]
mod sparse;

use cases::cases;
use dense::dense;
use entry::{Entry, Types};
use graph::{Edge, Graph};
use grid::grid;
use interleaved::interleaved_components;
use proptest::prelude::*;
use runner::{check, stated_by};
use sparse::Sparse;

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

fn interleaved_case(entry: Entry) -> impl Strategy<Value = (Graph, Vec<f64>, Vec<usize>)> {
    let graphs = interleaved_components().prop_map(|(graph, _parts)| graph);
    stated_by(entry, graphs).prop_flat_map(|graph| {
        let n = graph.n;
        (
            Just(graph),
            prop::collection::vec(-10.0f64..10.0, n),
            Just((0..n).collect::<Vec<_>>()).prop_shuffle(),
        )
    })
}

cases! {
    /// `(P A Pᵀ)(P x) = P b`; interleaved components force a non-identity block order.
    fn permuting_interleaved_components_permutes_the_solution(
        entry in entries(all),
        types in types(all),
    ) {
        check(interleaved_case(entry), |(graph, rhs, p)| {
            let mut permuted_rhs = vec![0.0; rhs.len()];
            (rhs.iter().enumerate()).for_each(|(v, &value)| permuted_rhs[p[v]] = value);
            let relabeled = permuted(&graph, &p);
            let base = entry.reduce(&graph, types);
            let got = entry.reduce(&relabeled, types);
            let base = base.factorize(&dense()).solve(&rhs);
            let got = got.factorize(&dense()).solve(&permuted_rhs);
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
            Ok(())
        });
    }
}

cases! {
    /// Equivariance survives an error made alike both times; this checks the solve itself.
    fn interleaved_components_are_solved_not_just_relabelled_consistently(
        entry in entries(all),
        types in types(all),
    ) {
        check(interleaved_case(entry), |(graph, rhs, _p)| {
            let rhs = graph.project_to_range(&rhs);
            prop_assume!(rhs.iter().map(|value| value * value).sum::<f64>().sqrt() > 1e-9);
            let reduced = entry.reduce(&graph, types);
            let x = reduced.factorize(&dense()).solve(&rhs);
            let relative = graph.relative_residual(&x, &rhs);
            // Measured at most 9ε.
            prop_assert!(
                relative < 64.0 * types.epsilon(),
                "{types:?}: components left residual {relative:e}"
            );
            Ok(())
        });
    }
}

/// Dyadic, so `value + shift` is exact below the scalar's spare mantissa bits.
fn assert_a_constant_leaves_the_solution(entry: Entry, types: Types, sparse: Sparse, shift: f64) {
    let graph = grid(100, 100);
    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs: Vec<f64> = (0..graph.n)
        .map(|i| (i * 37 % 2001) as f64 / 1024.0 - 1.0)
        .collect();
    let shifted: Vec<f64> = rhs.iter().map(|value| value + shift).collect();
    let config = sparse.config();
    let factor = entry.reduce(&graph, types).factorize(&config);
    let (base, got) = (factor.solve(&rhs), factor.solve(&shifted));
    let difference: Vec<f64> = got.iter().zip(&base).map(|(g, b)| g - b).collect();
    let moved = norm(&difference) / norm(&base);
    assert!(
        moved < 1e-3,
        "{types:?} {config:?}: the constant moved the solution by {moved:e}"
    );
}

cases! {
    /// A floating block projects a constant out of `b`; only the solve's own sums can lose it.
    fn a_constant_added_to_a_floating_rhs_leaves_the_solution(
        route in sparse_routes(all),
        types in types(f64),
    ) {
        let (entry, sparse) = route;
        assert_a_constant_leaves_the_solution(entry, types, sparse, 2f64.powi(30));
    }
}

cases! {
    fn a_constant_added_to_a_floating_f32_rhs_leaves_the_solution(
        route in sparse_routes(all),
        types in types(f32),
    ) {
        let (entry, sparse) = route;
        assert_a_constant_leaves_the_solution(entry, types, sparse, 2f64.powi(12));
    }
}

/// `D L D` floats with null vector `D⁻¹·1` and needs scaling, so both projections run on scaled factors.
#[test]
fn a_scaled_floating_block_projects_its_null_vector_out() {
    use approx_chol_next::input::Csr;
    use approx_chol_next::{factorize, Config, ScalingPolicy, Sgdd};

    let graph = grid(30, 30);
    let d: Vec<f64> = (0..graph.n)
        .map(|i| 1e-3f64.powf((i * 37 % 101) as f64 / 100.0))
        .collect();
    let (ptrs, columns, values) = graph.csr(false);
    let scaled: Vec<f64> = (0..graph.n)
        .flat_map(|i| (ptrs[i]..ptrs[i + 1]).map(move |k| (i, k)))
        .map(|(i, k)| d[i] * values[k] * d[columns[k]])
        .collect();
    let csr = Csr::new(graph.n, &ptrs, &columns, &scaled).unwrap();
    let sgdd: Sgdd = Sgdd::from_csr(csr, ScalingPolicy::default()).unwrap();
    assert!(sgdd.blocks().iter().all(|block| block.floats()));
    let factor = factorize(&sgdd, &Config::default());

    let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let null: Vec<f64> = d.iter().map(|d| 1.0 / d).collect();
    let rhs: Vec<f64> = (0..graph.n)
        .map(|i| (i * 37 % 2001) as f64 / 1024.0 - 1.0)
        .collect();
    let shifted: Vec<f64> = rhs.iter().zip(&null).map(|(b, z)| b + z).collect();
    let (x, moved_x) = (factor.solve(&rhs).unwrap(), factor.solve(&shifted).unwrap());
    let along = dot(&null, &x).abs() / (norm(&null) * norm(&x));
    let difference: Vec<f64> = moved_x.iter().zip(&x).map(|(a, b)| a - b).collect();
    let moved = norm(&difference) / norm(&x);
    // Measured 5e-16 and 5e-11; with the projections skipped, 0.97 and 1e4.
    assert!(
        along < 1e-12,
        "the solution keeps {along:e} of the null vector"
    );
    assert!(
        moved < 1e-8,
        "the null vector in b moved the solution by {moved:e}"
    );
}
