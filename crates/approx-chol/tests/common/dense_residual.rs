use approx_chol::Factor;

pub fn solve_residual(factor: &Factor, matrix: &[Vec<f64>]) {
    let exact: Vec<f64> = (0..factor.n()).map(|i| (i % 5) as f64 - 2.0).collect();
    let rhs = apply(matrix, &exact);
    let solution = factor.solve(&rhs).unwrap();
    let residual = apply(matrix, &solution);
    for (actual, expected) in residual.iter().zip(&rhs) {
        assert!((actual - expected).abs() < 1e-10 * (1.0 + expected.abs()));
    }
}

pub fn apply(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    (matrix.iter())
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}
