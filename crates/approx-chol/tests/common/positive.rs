use crate::graph::Edge;

impl Edge {
    /// `w (e_a + e_b)(e_a + e_b)ᵀ`: a positive off-diagonal.
    pub fn positive(a: usize, b: usize, weight: f64) -> Self {
        Self {
            positive: true,
            ..Self::negative(a, b, weight)
        }
    }
}
