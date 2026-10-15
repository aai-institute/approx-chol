use std::collections::BTreeMap;

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

    pub fn is_grounded(&self) -> bool {
        self.ground.iter().any(|&g| g != 0.0)
    }

    /// `A`'s nonzeros, both triangles.
    pub fn entries(&self) -> BTreeMap<(usize, usize), f64> {
        let mut entries = BTreeMap::new();
        for (i, &g) in self.ground.iter().enumerate() {
            *entries.entry((i, i)).or_insert(0.0) += g;
        }
        for edge in &self.edges {
            let off = if edge.positive {
                edge.weight
            } else {
                -edge.weight
            };
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
}
