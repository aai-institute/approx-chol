//! #435: a frustrated block under `Signed` grounds however small its frustration, so it never floats; at `ε = 0` it is balanced and folds. Each fixture is balanced but for `ε`, its near-null vector `1`.

mod sweep;

use approx_chol::{Config, FrustratedRoute};

use crate::common::entry::{by_type, floats, Entry, Factored, Types};
use crate::common::frustrated::is_signed;
use crate::common::graph::{Edge, Graph};
use crate::common::grid::grid;
use crate::common::pcg::pcg;
use crate::common::sparse::clique_tree;

/// A unit-weight cycle, its edge `n−1 — 0` positive at `ε`.
fn cycle(n: usize, epsilon: f64) -> Graph {
    let mut edges: Vec<Edge> = (1..n).map(|v| Edge::negative(v - 1, v, 1.0)).collect();
    edges.push(Edge::positive(n - 1, 0, epsilon));
    Graph::new(n, edges)
}

/// A unit `k × k` grid Laplacian, `flips` of its edges spread over it positive at `ε`.
fn flipped_grid(k: usize, flips: usize, epsilon: f64) -> Graph {
    let mut graph = grid(k, k);
    let m = graph.edges.len();
    for f in 0..flips {
        let edge = &mut graph.edges[(m / (flips + 1)) * (f + 1)];
        (edge.weight, edge.positive) = (epsilon, true);
    }
    graph
}

/// Whether any component floats, and whether any is signed.
fn state(factored: &Factored) -> (bool, bool) {
    by_type!(&factored.factor, factor => {
        (floats(factor).into_iter().any(|f| f), is_signed(factor))
    })
}

/// Every `D⁻¹` the payload stores, approximate steps' and dense diagonals', each as written: one serialized as `null`, a non-finite value, reads as NaN.
#[cfg(feature = "serde")]
fn inverse_pivots(factored: &Factored) -> Vec<f64> {
    use serde_json::Value;
    fn walk(value: &Value, found: &mut Vec<f64>) {
        let number = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
        match value {
            Value::Object(map) => {
                for (key, inner) in map {
                    match (key.as_str(), inner) {
                        ("inverse_degree", d) => found.push(number(d)),
                        ("Dense", Value::Array(parts)) => {
                            let packed = parts[0].as_array().expect("packed rows");
                            let diagonal = (0..).map(|row| row * (row + 1) / 2 + row);
                            let diagonal = diagonal.take_while(|&at| at < packed.len());
                            found.extend(diagonal.map(|at| number(&packed[at])));
                        }
                        _ => walk(inner, found),
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|v| walk(v, found)),
            _ => {}
        }
    }
    let payload = by_type!(&factored.factor, factor => serde_json::to_value(factor).unwrap());
    let mut found = Vec::new();
    walk(&payload, &mut found);
    found
}

/// The smallest pivot, failing closed: some `D⁻¹` is stored, and every one is finite and positive.
#[cfg(feature = "serde")]
fn smallest_pivot(factored: &Factored) -> f64 {
    let inverse = inverse_pivots(factored);
    assert!(!inverse.is_empty(), "the payload stores no pivot");
    let bad = inverse.iter().find(|&&d| !(d.is_finite() && d > 0.0));
    assert!(bad.is_none(), "an inverse pivot {bad:?}");
    1.0 / inverse.iter().fold(0.0f64, |m, &d| m.max(d))
}

/// `‖b‖ = 1`: generic along `1`, or with its mean removed.
fn right_hand_sides(n: usize) -> (Vec<f64>, Vec<f64>) {
    let generic: Vec<f64> = (0..n).map(|i| 1.0 + ((7 * i + 3) as f64).sin()).collect();
    let mean = generic.iter().sum::<f64>() / n as f64;
    let perp: Vec<f64> = generic.iter().map(|g| g - mean).collect();
    let unit = |v: Vec<f64>| {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    };
    (unit(generic), unit(perp))
}

/// `Signed` under AC or AC2, every block approximate.
fn signed(copies: u32, seed: u64) -> Config {
    Config {
        seed,
        sampling: clique_tree(copies),
        frustrated: FrustratedRoute::Signed,
        exact_max_dim: 0,
        ..Config::default()
    }
}

/// A cycle and a grid one edge short of balance, that edge at `ε` from `1e-2` to far below `ε_mach`: the signed factor stays grounded, its smallest pivot positive, and PCG on `b ⟂ 1` reaches 1e-8 within about twice the sweep's worst count over these seeds, 25 iterations in `f64` and 61 at `f32`'s `1e-12`.
#[test]
fn a_nearly_balanced_frustrated_block_stays_grounded_and_preconditions() {
    let points = [
        (Types::F64U32, &[1e-2, 1e-8, 1e-14, 1e-20][..], 50),
        (Types::F32U32, &[1e-2, 1e-6, 1e-12][..], 120),
    ];
    for (types, epsilons, cap) in points {
        for &epsilon in epsilons {
            for (name, graph) in [
                ("cycle", cycle(500, epsilon)),
                ("grid", flipped_grid(32, 1, epsilon)),
            ] {
                let (_, perp) = right_hand_sides(graph.n);
                let reduced = Entry::Csr.reduce(&graph, types);
                for (copies, seed) in [1, 2].into_iter().flat_map(|k| (0..3).map(move |s| (k, s))) {
                    let case = format!(
                        "{name} at ε = {epsilon:e}, {types:?}, {copies} copies, seed {seed}"
                    );
                    let factored = reduced.factorize(&signed(copies, seed));
                    assert_eq!(
                        state(&factored),
                        (false, true),
                        "{case}: grounded and signed"
                    );
                    #[cfg(feature = "serde")]
                    assert!(smallest_pivot(&factored) > 0.0, "{case}: a positive pivot");
                    let run = pcg(|x| graph.apply(x), |r| factored.solve(r), &perp, 1e-8, cap);
                    assert!(
                        run.iterations.is_some(),
                        "{case}: {:e} after {cap} iterations",
                        run.residual
                    );
                }
            }
        }
    }
}

/// At `ε = 0` the flipped edge is gone and the block balanced: it folds before any route applies and floats, `Signed` or not.
#[test]
fn a_balanced_block_folds_and_floats_under_signed() {
    for types in [Types::F64U32, Types::F32U32] {
        for graph in [cycle(500, 0.0), flipped_grid(32, 1, 0.0)] {
            let factored = Entry::Csr.reduce(&graph, types).factorize(&signed(1, 0));
            assert_eq!(state(&factored), (true, false), "floats, folded");
        }
    }
}
