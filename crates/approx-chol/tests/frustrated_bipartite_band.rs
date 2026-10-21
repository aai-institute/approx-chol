#[path = "common/gram.rs"]
mod gram;

use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
use approx_chol::{factorize_with, Backend, Config, SchurMode};
use gram::{dense_csr, matrix};

/// A frustrated unit cycle `L0 R0 L1 R1 …`, `cycle` vertices a side, and `pendants` left vertices each joined by `1e-95` to `wide` cycle vertices and left a `1e-200` surplus: within the band.
struct Cycle {
    cycle: usize,
    pendants: usize,
    wide: usize,
    /// Unit pendants on the right, which make the left side, surplus and all, the kept one.
    right_pendants: usize,
}

impl Cycle {
    fn sides(&self) -> (usize, usize) {
        (self.cycle + self.pendants, self.cycle + self.right_pendants)
    }

    /// Columns `(left, right, x, y)` of `V Wᵀ`.
    fn columns(&self) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
        let m = self.cycle;
        let mut columns = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut push = |l: usize, r: usize, x: f64, y: f64| {
            columns.0.push(l as u32);
            columns.1.push(r as u32);
            columns.2.push(x);
            columns.3.push(y);
        };
        for i in 0..m {
            push(i, i, 1.0, 1.0);
            push(i, (i + 1) % m, 1.0, if i == m - 1 { -1.0 } else { 1.0 });
        }
        for p in 0..self.pendants {
            for k in 0..self.wide {
                let y = if k == 0 { -1e-95 } else { 1e-95 };
                push(m + p, (p + k) % m, 1e-95, y);
            }
            push(m + p, 0, 1e-100, 0.0);
        }
        for p in 0..self.right_pendants {
            push(p % m, m + p, 1.0, 1.0);
        }
        columns
    }
}

/// `‖Mx − b‖∞` past `64ε (‖M‖∞‖x‖∞ + ‖b‖∞)`, for `b = M·want`.
fn unstable(m: &[Vec<f64>], solve: impl FnOnce(&[f64]) -> Vec<f64>) -> Option<String> {
    let max = |v: &[f64]| v.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let apply = |x: &[f64]| -> Vec<f64> {
        (m.iter())
            .map(|row| row.iter().zip(x).map(|(m, x)| m * x).sum())
            .collect()
    };
    let want: Vec<f64> = (0..m.len())
        .map(|i| (7 * i + 3) as f64)
        .map(f64::sin)
        .collect();
    let b = apply(&want);
    let x = solve(&b);
    let residual = (apply(&x).iter().zip(&b))
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
        backend: Backend::Approximate,
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
        // A two-neighbor pendant's star samples its ground share onto one neighbor per sheet, so the copies ground unequally.
        (
            "eliminated, sampled unequally",
            Cycle {
                wide: 2,
                ..narrow(8, 7, 0)
            },
        ),
        ("kept", narrow(6, 4, 10)),
    ];
    let mut failures = Vec::new();
    for (surplus, cycle) in cycles {
        let (left, right, x, y) = cycle.columns();
        let (n_left, n_right) = cycle.sides();
        let n = n_left + n_right;
        let shifted: Vec<u32> = right.iter().map(|&r| r + n_left as u32).collect();
        let m = matrix(n, &left, &shifted, &x, &y);
        let (ptrs, columns, values) = dense_csr(&m);
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(n_left, &left, Some(&x)).unwrap(),
            FactorWidth1::new(n_right, &right, Some(&y)).unwrap(),
        )
        .unwrap();
        let csr = CsrRef::new(&ptrs, &columns, &values, n as u32).unwrap();
        for (complement, config) in [
            ("dense", Config::default()),
            ("exact", exact),
            ("sampled", approximate),
        ] {
            let gram = factorize_with(gram, config).unwrap();
            let csr = factorize_with(csr, config).unwrap();
            for (view, factor) in [("gram", gram), ("csr", csr)] {
                let failure = unstable(&m, |b| factor.solve(b).unwrap());
                let name = format!("{surplus} surplus, {complement} complement, {view}");
                failures.extend(failure.map(|f| format!("{name}: {f}")));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
