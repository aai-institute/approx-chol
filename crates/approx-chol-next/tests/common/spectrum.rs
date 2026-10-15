use faer::{Mat, Side};

use crate::entry::Factored;
use crate::graph::Graph;

impl Factored {
    /// Eigenvalues of `M⁺A` on `range(A)`, ascending: those of `W M⁺ Wᵀ` for `A = Wᵀ W`, `W = S^{1/2} Uᵀ` over `A`'s nonzero eigenpairs.
    pub fn spectrum(&self, graph: &Graph) -> Vec<f64> {
        let n = graph.n;
        let entries = graph.entries();
        let a = Mat::from_fn(n, n, |i, j| entries.get(&(i, j)).copied().unwrap_or(0.0));
        let eigen = a.self_adjoint_eigen(Side::Lower).expect("A is symmetric");
        let (u, s) = (eigen.U(), eigen.S().column_vector());
        let top = (0..n).map(|k| s[k]).fold(0.0, f64::max);
        let range: Vec<usize> = (0..n).filter(|&k| s[k] > 1e-10 * top).collect();
        let w = Mat::from_fn(range.len(), n, |k, i| s[range[k]].sqrt() * u[(i, range[k])]);
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|j| self.solve(&(0..n).map(|i| f64::from(i == j)).collect::<Vec<_>>()))
            .collect();
        let pinv = Mat::from_fn(n, n, |i, j| columns[j][i]);
        let sandwich = &w * pinv * w.transpose();
        // The eigensolver reads one triangle, so an asymmetric `M⁺` would pass unseen.
        let m = range.len();
        let pairs = (0..m).flat_map(|i| (0..m).map(move |j| (i, j)));
        let (skew, top) = pairs.fold((0.0f64, 0.0f64), |(skew, top), (i, j)| {
            let g = sandwich[(i, j)];
            (skew.max((g - sandwich[(j, i)]).abs()), top.max(g.abs()))
        });
        assert!(
            skew <= 1e-3 * top,
            "M⁺ is asymmetric: {skew:e} against {top:e}"
        );
        sandwich
            .self_adjoint_eigenvalues(Side::Lower)
            .expect("M⁺ is symmetric")
    }
}
