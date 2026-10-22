#[path = "common/dense.rs"]
mod dense;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/residual.rs"]
mod residual;

use approx_chol::input::Laplacian;
use approx_chol::{factorize_with, Backend, Config, Factor, OwnedCsr};
use dense::dense;
use graph::{Edge, Graph};

/// `max_i |Ax − b|_i / (|A||x| + |b|)_i`; NaN when any term is.
fn omega(graph: &Graph, x: &[f64], b: &[f64]) -> f64 {
    let mut scale: Vec<f64> = b.iter().map(|b| b.abs()).collect();
    for ((i, j), a) in graph.entries() {
        scale[i] += (a * x[j]).abs();
    }
    let residual = graph.apply(x);
    (residual.iter().zip(b).zip(&scale))
        .map(|((ax, b), s)| (ax - b).abs() / s)
        .fold(0.0, |worst, w| {
            if w.is_nan() || worst.is_nan() {
                f64::NAN
            } else {
                worst.max(w)
            }
        })
}

/// The sweep's bar: `max(1e3 ε, 10 ω(x*))`.
fn assert_componentwise(graph: &Graph, factor: &Factor, x_true: &[f64]) {
    let b = graph.apply(x_true);
    let x = factor.solve(&b).unwrap();
    assert!(x.iter().all(|v| v.is_finite()), "x = {x:?}");
    let bar = (1e3 * f64::EPSILON).max(10.0 * omega(graph, x_true, &b));
    let got = omega(graph, &x, &b);
    let normwise = graph.relative_residual(&x, &b);
    assert!(
        got <= bar,
        "omega {got:e} past {bar:e} (normwise {normwise:e}), x = {x:?}"
    );
}

fn csr_factor(graph: &Graph, config: Config) -> Factor {
    let (ptrs, cols, vals) = graph.csr(false);
    let csr = OwnedCsr::<f64, u32>::try_from_usize(&ptrs, &cols, &vals, graph.n).unwrap();
    factorize_with(csr.as_csr_ref(), config).unwrap()
}

/// Heavy-row rounding reaches the last-eliminated weak pendant unequally on the cover's two sheets.
#[test]
#[ignore = "#283: componentwise accuracy of the frustrated cover past a weak vertex"]
fn frustrated_cover_solves_a_pendant_behind_a_heavy_triangle() {
    let (h, w) = (1e20, 1.0);
    let positive = |a, b| Edge {
        a,
        b,
        weight: h,
        positive: true,
    };
    let x_true = [1.0, -1.0, 0.5, 2.0];
    let mut graph = Graph::new(
        x_true.len(),
        vec![
            positive(0, 1),
            positive(1, 2),
            positive(0, 2),
            Edge::negative(0, 3, w),
        ],
    );
    // One ulp of the 2e20 row: dominant, by the least it can be.
    graph.ground[0] = 32768.0;
    assert_componentwise(&graph, &csr_factor(&graph, dense()), &x_true);
}

/// The dense solve pins the last vertex, whose only edge is below its neighbour's pivot ulp.
#[test]
#[ignore = "#284: componentwise accuracy of a floating solve pinned at a weakly coupled vertex"]
fn floating_laplacian_solves_with_a_weakly_coupled_last_vertex() {
    let x_true = [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0];
    let graph = Graph::new(
        x_true.len(),
        vec![Edge::negative(0, 1, 0.5), Edge::negative(0, 2, 1e-20)],
    );
    let ends = |end: fn(&Edge) -> u32| graph.edges.iter().map(end).collect::<Vec<u32>>();
    let (u, v) = (ends(|e| e.a as u32), ends(|e| e.b as u32));
    let weights: Vec<f64> = graph.edges.iter().map(|e| e.weight).collect();
    let approximate = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };
    for config in [dense(), approximate] {
        let laplacian = Laplacian::weighted(graph.n, &u, &v, &weights).unwrap();
        let factor = factorize_with(laplacian, config).unwrap();
        assert_componentwise(&graph, &factor, &x_true);
    }
}
