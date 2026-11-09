use std::collections::BTreeMap;

use super::pcg::relative_residual;

/// `w (e_a − e_b)(e_a − e_b)ᵀ`, or `w (e_a + e_b)(e_a + e_b)ᵀ` when `positive`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub weight: f64,
    pub positive: bool,
}

impl Edge {
    pub fn negative(a: usize, b: usize, weight: f64) -> Self {
        Self {
            a,
            b,
            weight,
            positive: false,
        }
    }

    /// The sign of the edge's off-diagonal entries.
    pub fn sign(&self) -> f64 {
        if self.positive {
            1.0
        } else {
            -1.0
        }
    }

    /// `w (e_a + e_b)(e_a + e_b)ᵀ`: a positive off-diagonal.
    pub fn positive(a: usize, b: usize, weight: f64) -> Self {
        Self {
            positive: true,
            ..Self::negative(a, b, weight)
        }
    }
}

/// `A = Σ_k A_k + diag(ground)` over its edges: the one description every entry point reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<Edge>,
    pub ground: Vec<f64>,
}

impl Graph {
    pub fn new(n: usize, edges: Vec<Edge>) -> Self {
        Self {
            n,
            edges,
            ground: vec![0.0; n],
        }
    }

    /// `A`'s nonzeros, both triangles.
    pub fn entries(&self) -> BTreeMap<(usize, usize), f64> {
        let mut entries = BTreeMap::new();
        for (i, &g) in self.ground.iter().enumerate() {
            *entries.entry((i, i)).or_insert(0.0) += g;
        }
        for edge in &self.edges {
            let off = edge.sign() * edge.weight;
            for (i, j, value) in [
                (edge.a, edge.a, edge.weight),
                (edge.b, edge.b, edge.weight),
                (edge.a, edge.b, off),
                (edge.b, edge.a, off),
            ] {
                *entries.entry((i, j)).or_insert(0.0) += value;
            }
        }
        entries.retain(|_, value| *value != 0.0);
        entries
    }

    /// `A` as CSR row pointers, columns and values; the upper triangle alone when `upper`.
    pub fn csr(&self, upper: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (mut ptrs, mut columns, mut values) = (vec![0], vec![], vec![]);
        let mut entries = self.entries().into_iter().peekable();
        for row in 0..self.n {
            while let Some(((_, j), value)) = entries.next_if(|((i, _), _)| *i == row) {
                if !upper || j >= row {
                    columns.push(j);
                    values.push(value);
                }
            }
            ptrs.push(columns.len());
        }
        (ptrs, columns, values)
    }

    /// `A x` from the edges, independent of every entry point under test.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y: Vec<f64> = (self.ground.iter().zip(x)).map(|(g, x)| g * x).collect();
        for edge in &self.edges {
            let sign = edge.sign();
            let t = edge.weight * (x[edge.a] + sign * x[edge.b]);
            y[edge.a] += t;
            y[edge.b] += sign * t;
        }
        y
    }

    /// `‖A x − b‖ / ‖b‖`.
    pub fn relative_residual(&self, x: &[f64], b: &[f64]) -> f64 {
        relative_residual(&self.apply(x), b)
    }
}
