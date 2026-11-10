use approx_chol::input::Side;
use approx_chol::Real;

use super::bipartite::Bipartite;

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

/// `matrix`'s nonzeros as whole CSR arrays.
pub fn dense_csr<T: Real>(matrix: &[Vec<T>]) -> (Vec<u32>, Vec<u32>, Vec<T>) {
    let (mut ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for row in matrix {
        for (j, &x) in row.iter().enumerate().filter(|&(_, &x)| x != T::zero()) {
            columns.push(j as u32);
            values.push(x);
        }
        ptrs.push(columns.len() as u32);
    }
    (ptrs, columns, values)
}

impl Bipartite {
    /// `A` dense, the right side numbered after the left; no `x` and `y` reads as an incidence.
    pub fn dense(&self) -> Vec<Vec<f64>> {
        let ones = vec![1.0; self.left.len()];
        let (x, y) = self.values.as_ref().map_or((&ones, &ones), |(x, y)| (x, y));
        let shifted: Vec<u32> = self.right.iter().map(|&r| r + self.n_left as u32).collect();
        matrix(self.n_left + self.n_right, &self.left, &shifted, x, y)
    }

    /// `A`'s Schur complement onto `kept`, dense: the other side's block of a bipartite `A` is diagonal.
    pub fn complement(&self, kept: Side) -> Vec<Vec<f64>> {
        let a = self.dense();
        let (kept, eliminated) = match kept {
            Side::Left => (0..self.n_left, self.n_left..a.len()),
            Side::Right => (self.n_left..a.len(), 0..self.n_left),
        };
        let s_entry = |i: usize, j: usize| {
            let through = (eliminated.clone()).filter(|&e| a[e][e] != 0.0);
            a[i][j] - through.map(|e| a[i][e] * a[e][j] / a[e][e]).sum::<f64>()
        };
        (kept.clone())
            .map(|i| kept.clone().map(|j| s_entry(i, j)).collect())
            .collect()
    }
}
