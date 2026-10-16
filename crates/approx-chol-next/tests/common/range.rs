use crate::graph::Graph;

impl Graph {
    /// `b` with each floating component's null vector projected out, so `A x = b` is consistent.
    pub fn project_to_range(&self, b: &[f64]) -> Vec<f64> {
        let mut adjacency = vec![Vec::new(); self.n];
        for edge in self.edges.iter().filter(|edge| edge.weight != 0.0) {
            adjacency[edge.a].push((edge.b, -edge.sign()));
            adjacency[edge.b].push((edge.a, -edge.sign()));
        }
        let mut projected = b.to_vec();
        let mut null = vec![0.0f64; self.n];
        for root in 0..self.n {
            if null[root] != 0.0 {
                continue;
            }
            // The null vector `s` has `s_b = −σ s_a` across an edge of sign `σ`.
            null[root] = 1.0;
            let (mut component, mut stack, mut balanced) = (vec![], vec![root], true);
            while let Some(v) = stack.pop() {
                component.push(v);
                for &(u, sign) in &adjacency[v] {
                    match null[u] {
                        0.0 => {
                            null[u] = sign * null[v];
                            stack.push(u);
                        }
                        s => balanced &= s == sign * null[v],
                    }
                }
            }
            let floats = balanced && component.iter().all(|&v| self.ground[v] == 0.0);
            if floats {
                // Twice: on a nearly null `b`, one pass leaves rounding at `ε‖b‖`, far above `ε‖projected‖`.
                for _ in 0..2 {
                    let along = component
                        .iter()
                        .map(|&v| null[v] * projected[v])
                        .sum::<f64>();
                    let along = along / component.len() as f64;
                    component
                        .iter()
                        .for_each(|&v| projected[v] -= along * null[v]);
                }
            }
        }
        projected
    }
}
