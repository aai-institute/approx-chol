use approx_chol::input::Laplacian;
use approx_chol::{factorize_with, ApproxInverse, Config, ScalingPolicy};

use crate::common::solve::Solve;
use crate::common::sparse::clique_tree;

fn approximate(copies: u32) -> Config {
    Config {
        sampling: clique_tree(copies),
        exact_max_dim: 0,
        ..Config::default()
    }
}

#[test]
fn a_single_edge_queues_and_solves_at_the_most_copies() {
    // One edge samples nothing: this exercises only the degree queue and its capped keys.
    let edge = Laplacian::new(2, &[0u32], &[1]).unwrap();
    let factor: ApproxInverse =
        factorize_with(edge, approximate(u32::MAX), ScalingPolicy::default()).unwrap();
    let x = factor.solve(&[1.0, -1.0]).unwrap();
    assert!(
        (x[0] - 0.5).abs() < 1e-12 && (x[1] + 0.5).abs() < 1e-12,
        "{x:?}"
    );
}

#[test]
fn a_triangle_samples_every_copy_at_many_copies() {
    // The first pivot draws 2^20 copies, all onto the one later neighbor, so the factor is exact.
    let triangle = Laplacian::new(3, &[0u32, 0, 1], &[1, 2, 2]).unwrap();
    let factor: ApproxInverse =
        factorize_with(triangle, approximate(1 << 20), ScalingPolicy::default()).unwrap();
    let x = factor.solve(&[1.0, -1.0, 0.0]).unwrap();
    let expected = [1.0 / 3.0, -1.0 / 3.0, 0.0];
    assert!(
        x.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12),
        "{x:?}"
    );
}
