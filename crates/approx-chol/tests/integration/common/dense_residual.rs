use approx_chol::{ApproxInverse, Threads};

pub fn solve_residual(factor: &ApproxInverse, matrix: &[Vec<f64>]) {
    let exact: Vec<f64> = (0..factor.n()).map(|i| (i % 5) as f64 - 2.0).collect();
    let rhs = apply(matrix, &exact);
    let mut solution = rhs.clone();
    factor.apply(&mut solution, Threads::All).unwrap();
    let residual = apply(matrix, &solution);
    for (actual, expected) in residual.iter().zip(&rhs) {
        assert!((actual - expected).abs() < 1e-10 * (1.0 + expected.abs()));
    }
}

/// `max |a_i − b_i|`.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b)).fold(0.0, |m, (a, b)| m.max((a - b).abs()))
}

pub fn apply(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    (matrix.iter())
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}
