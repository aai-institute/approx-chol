use approx_chol::input::CsrRef;
use approx_chol::{factorize_with, Config, ScalingPolicy, SchurMode};

use crate::common::bipartite::Bipartite;
use crate::common::cycle::Cycle;
use crate::common::dense_residual::apply;
use crate::common::gram::dense_csr;
use crate::common::solve::Solve;

/// `‖Mx − b‖∞` past `64ε (‖M‖∞‖x‖∞ + ‖b‖∞)`, for `b = M·want`.
fn unstable(m: &[Vec<f64>], solve: impl FnOnce(&[f64]) -> Vec<f64>) -> Option<String> {
    let max = |v: &[f64]| v.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let want: Vec<f64> = (0..m.len())
        .map(|i| (7 * i + 3) as f64)
        .map(f64::sin)
        .collect();
    let b = apply(m, &want);
    let x = solve(&b);
    let residual = (apply(m, &x).iter().zip(&b))
        .map(|(mx, b)| (mx - b).abs())
        .fold(0.0, f64::max);
    let norm = (m.iter())
        .map(|row| row.iter().map(|m| m.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let bound = 64.0 * f64::EPSILON * (norm * max(&x) + max(&b));
    (residual > bound).then(|| format!("residual {residual:e} > {bound:e}, x = {x:?}"))
}

#[test]
fn a_frustrated_bipartite_block_with_a_negligible_surplus_solves_backward_stably() {
    let approximate = Config {
        exact_max_dim: 0,
        ..Config::default()
    };
    let exact = Config {
        complement: SchurMode::Exact,
        ..approximate
    };
    let narrow = |cycle, pendants, right_pendants| Cycle {
        cycle,
        pendants,
        wide: 1,
        right_pendants,
    };
    let cycles = [
        ("eliminated", narrow(8, 7, 0)),
        // A two-neighbor pendant's star samples fill between its neighbors.
        (
            "eliminated, two neighbors",
            Cycle {
                wide: 2,
                ..narrow(8, 7, 0)
            },
        ),
        ("kept", narrow(6, 4, 10)),
    ];
    let mut failures = Vec::new();
    for (surplus, cycle) in cycles {
        let g = cycle.bipartite();
        let m = g.dense();
        let (ptrs, columns, values) = dense_csr(&m);
        let gram = g.view();
        let csr = CsrRef::new(m.len(), &ptrs, &columns, &values).unwrap();
        for (complement, config) in [
            ("dense", Config::default()),
            ("exact", exact),
            ("sampled", approximate),
        ] {
            let gram = factorize_with(gram, config, ScalingPolicy::default()).unwrap();
            let csr = factorize_with(csr, config, ScalingPolicy::default()).unwrap();
            for (view, factor) in [("gram", gram), ("csr", csr)] {
                let failure = unstable(&m, |b| factor.solve(b).unwrap());
                let name = format!("{surplus} surplus, {complement} complement, {view}");
                failures.extend(failure.map(|f| format!("{name}: {f}")));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_frustrated_kept_side_as_large_as_exact_max_dim_solves_exactly() {
    // Complete and mixed-signed, so no sampled clique is the whole one.
    let (n_left, n_right) = (14, 10);
    let (mut left, mut right, mut x, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in 0..n_left {
        for r in 0..n_right {
            left.push(l as u32);
            right.push(r as u32);
            x.push(1.0 + ((l * 7 + r * 3) % 5) as f64);
            y.push(if (l + 2 * r) % 3 == 0 { -1.0 } else { 1.0 });
        }
    }
    let g = Bipartite {
        n_left,
        n_right,
        left,
        right,
        values: Some((x, y)),
    };
    let m = g.dense();
    let config = Config {
        exact_max_dim: n_left.min(n_right),
        ..Config::default()
    };
    let factor = factorize_with(g.view(), config, ScalingPolicy::default()).unwrap();
    let failure = unstable(&m, |b| factor.solve(b).unwrap());
    assert!(failure.is_none(), "{failure:?}");
}
