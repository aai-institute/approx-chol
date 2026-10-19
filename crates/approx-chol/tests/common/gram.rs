use approx_chol::Factor;

pub fn solve_residual(factor: &Factor, matrix: &[Vec<f64>]) {
    let exact: Vec<f64> = (0..factor.n()).map(|i| (i % 5) as f64 - 2.0).collect();
    let apply = |x: &[f64]| -> Vec<f64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
            .collect()
    };
    let rhs = apply(&exact);
    let solution = factor.solve(&rhs).unwrap();
    let residual = apply(&solution);
    for (actual, expected) in residual.iter().zip(&rhs) {
        assert!((actual - expected).abs() < 1e-10 * (1.0 + expected.abs()));
    }
}

/// `V Vᵀ` dense, from the arrays `FactorWidth2::new` takes.
pub fn matrix(n: usize, a: &[u32], b: &[u32], x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for k in 0..a.len() {
        let (i, j) = (a[k] as usize, b[k] as usize);
        matrix[i][i] += x[k] * x[k];
        matrix[j][j] += y[k] * y[k];
        matrix[i][j] += x[k] * y[k];
        matrix[j][i] += x[k] * y[k];
    }
    matrix
}
