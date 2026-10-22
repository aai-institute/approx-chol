use approx_chol::input::Laplacian;
use approx_chol::{factorize_with, Backend, Config, Factor, ScalingPolicy};

fn approximate(split_merge: u32) -> Config {
    Config {
        split_merge: Some(split_merge),
        backend: Backend::Approximate,
        ..Config::default()
    }
}

#[test]
fn a_single_edge_queues_and_solves_at_the_largest_split_merge() {
    // One edge samples nothing: this exercises only the degree queue and its capped keys.
    let edge = Laplacian::new(2, &[0u32], &[1]).unwrap();
    let factor: Factor =
        factorize_with(edge, approximate(u32::MAX), ScalingPolicy::default()).unwrap();
    let x = factor.solve(&[1.0, -1.0]).unwrap();
    assert!(
        (x[0] - 0.5).abs() < 1e-12 && (x[1] + 0.5).abs() < 1e-12,
        "{x:?}"
    );
}

#[test]
fn a_triangle_samples_every_copy_at_a_large_split_merge() {
    // The first pivot draws 2^20 copies, all onto the one later neighbor, so the factor is exact.
    let triangle = Laplacian::new(3, &[0u32, 0, 1], &[1, 2, 2]).unwrap();
    let factor: Factor =
        factorize_with(triangle, approximate(1 << 20), ScalingPolicy::default()).unwrap();
    let x = factor.solve(&[1.0, -1.0, 0.0]).unwrap();
    let expected = [1.0 / 3.0, -1.0 / 3.0, 0.0];
    assert!(
        x.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12),
        "{x:?}"
    );
}
