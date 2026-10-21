use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
use approx_chol::{factorize_with, Backend, Config, SchurMode};

/// `‖Mx − b‖∞` beside `64ε (‖M‖∞‖x‖∞ + ‖b‖∞)`.
fn residual_and_bound(m: &[Vec<f64>], x: &[f64], b: &[f64]) -> (f64, f64) {
    let max = |v: &[f64]| v.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let rows: Vec<f64> = m
        .iter()
        .map(|row| row.iter().zip(x).map(|(m, x)| m * x).sum())
        .collect();
    let r = (rows.iter().zip(b))
        .map(|(mx, b)| (mx - b).abs())
        .fold(0.0, f64::max);
    let norm = m
        .iter()
        .map(|row| row.iter().map(|m| m.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    (r, 64.0 * f64::EPSILON * (norm * max(x) + max(b)))
}

/// A Gram `V Wᵀ` by its columns `(left, right, x, y)` on `n_left + n_right` vertices.
struct Gram<'a> {
    n_left: usize,
    n_right: usize,
    left: &'a [u32],
    right: &'a [u32],
    x: &'a [f64],
    y: &'a [f64],
}

impl Gram<'_> {
    /// Each view and complement that solves `M` past the bound, named.
    fn failures(&self) -> Vec<String> {
        let (n_left, n) = (self.n_left, self.n_left + self.n_right);
        let mut m = vec![vec![0.0; n]; n];
        for k in 0..self.left.len() {
            let (i, j) = (self.left[k] as usize, n_left + self.right[k] as usize);
            let (x, y) = (self.x[k], self.y[k]);
            m[i][i] += x * x;
            m[j][j] += y * y;
            m[i][j] += x * y;
            m[j][i] += x * y;
        }
        let want = [
            0.1f64.sqrt(),
            -0.7f64.cbrt(),
            2.3f64.ln(),
            0.55f64.exp(),
            -1.3f64.sin(),
            0.9f64.tan(),
            -1.7f64.sqrt(),
        ];
        let rhs: Vec<f64> = (m.iter())
            .map(|row| row.iter().zip(&want[..n]).map(|(m, x)| m * x).sum())
            .collect();
        let values: Vec<f64> = m.iter().flatten().copied().collect();
        let ptrs: Vec<u32> = (0..=n as u32).map(|i| i * n as u32).collect();
        let columns: Vec<u32> = (0..n * n).map(|k| (k % n) as u32).collect();
        let approximate = Config {
            backend: Backend::Approximate,
            ..Config::default()
        };
        let exact = Config {
            complement: SchurMode::Exact,
            ..approximate
        };
        let mut failures = Vec::new();
        for (complement, config) in [
            ("dense", Config::default()),
            ("exact", exact),
            ("sampled", approximate),
        ] {
            let gram = BipartiteFactorWidth2::new(
                FactorWidth1::new(self.n_left, self.left, Some(self.x)).unwrap(),
                FactorWidth1::new(self.n_right, self.right, Some(self.y)).unwrap(),
            )
            .unwrap();
            let csr = CsrRef::new(&ptrs, &columns, &values, n as u32).unwrap();
            let solves = [
                (
                    "gram",
                    factorize_with(gram, config).unwrap().solve(&rhs).unwrap(),
                ),
                (
                    "csr",
                    factorize_with(csr, config).unwrap().solve(&rhs).unwrap(),
                ),
            ];
            for (view, solved) in solves {
                let (residual, bound) = residual_and_bound(&m, &solved, &rhs);
                if residual > bound {
                    failures.push(format!(
                        "{view} {complement}: residual {residual:e} > {bound:e}, x = {solved:?}"
                    ));
                }
            }
        }
        failures
    }
}

#[test]
fn a_frustrated_bipartite_block_with_a_negligible_surplus_solves_backward_stably() {
    // A unit frustrated 4-cycle L0-R0-L1-R1, a pendant L2-R0 of 1e-190, and a one-sided column leaving L2 a 1e-200 surplus: within the band.
    let eliminated = Gram {
        n_left: 3,
        n_right: 2,
        left: &[0, 0, 1, 1, 2, 2],
        right: &[0, 1, 0, 1, 0, 0],
        x: &[1.0, 1.0, 1.0, 1.0, 1e-95, 1e-100],
        y: &[1.0, -1.0, -1.0, -1.0, -1e-95, 0.0],
    };
    // Unit pendants R2-L0 and R3-L1 make the right side the larger, so L2 and its surplus are kept.
    let kept = Gram {
        n_left: 3,
        n_right: 4,
        left: &[0, 0, 1, 1, 2, 2, 0, 1],
        right: &[0, 1, 0, 1, 0, 0, 2, 3],
        x: &[1.0, 1.0, 1.0, 1.0, 1e-95, 1e-100, 1.0, 1.0],
        y: &[1.0, -1.0, -1.0, -1.0, -1e-95, 0.0, 1.0, -1.0],
    };
    let failures: Vec<String> = [("eliminated", eliminated), ("kept", kept)]
        .into_iter()
        .flat_map(|(side, gram)| {
            gram.failures()
                .into_iter()
                .map(move |f| format!("{side} L2: {f}"))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}
