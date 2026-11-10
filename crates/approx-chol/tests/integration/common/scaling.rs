use approx_chol::{Real, ScalingPolicy};

use super::graph::Graph;
use super::lcg::Lcg;

/// No scaling step and no tolerance: every block short of dominance is clamped.
pub fn exhausted() -> ScalingPolicy {
    ScalingPolicy::new(0.0, 0).expect("a valid policy")
}

/// `graph`'s `A` as `D A D` in CSR, both triangles, `D` drawn from `lcg` over `decades`.
pub fn dld<T: Real>(
    graph: &Graph,
    lcg: &mut Lcg,
    decades: f64,
) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    let d: Vec<f64> = (0..graph.n)
        .map(|_| 10f64.powf(decades * lcg.unit()))
        .collect();
    let (ptrs, columns, values) = graph.csr(false);
    let scaled = (0..graph.n)
        .flat_map(|i| (ptrs[i]..ptrs[i + 1]).map(move |k| (i, k)))
        .map(|(i, k)| T::from(d[i] * values[k] * d[columns[k]]).expect("in range"))
        .collect();
    (ptrs, columns, scaled)
}
