use crate::graph::Graph;

impl Graph {
    /// `A x` from the edges, independent of every entry point under test.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y: Vec<f64> = (self.ground.iter().zip(x)).map(|(g, x)| g * x).collect();
        for edge in &self.edges {
            let sign = if edge.positive { 1.0 } else { -1.0 };
            let t = edge.weight * (x[edge.a] + sign * x[edge.b]);
            y[edge.a] += t;
            y[edge.b] += sign * t;
        }
        y
    }

    /// `‖A x − b‖ / ‖b‖`.
    pub fn relative_residual(&self, x: &[f64], b: &[f64]) -> f64 {
        let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|v| v * v).sum::<f64>().sqrt();
        let ax = self.apply(x);
        norm(&mut ax.iter().zip(b).map(|(ax, b)| ax - b)) / norm(&mut b.iter().copied())
    }
}
