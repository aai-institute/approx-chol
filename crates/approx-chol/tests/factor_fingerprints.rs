#[path = "common/grid.rs"]
mod grid;
use grid::grid_laplacian;

use approx_chol::{factorize_with, Backend, Config};
use rstest::rstest;

/// Recorded on main; a refactor that should leave the factor alone may move it by ulps only.
const RELATIVE_TOLERANCE: f64 = 1e-12;

fn fingerprint(grounded: bool, split_merge: Option<u32>) -> [f64; 2] {
    let mut lap = grid_laplacian(20, 20);
    if grounded {
        // Row 0's diagonal is its first stored entry.
        lap.values[0] += 1.0;
    }
    let config = Config {
        backend: Backend::Approximate,
        seed: 7,
        split_merge,
    };
    let factor = factorize_with(lap.as_csr().expect("valid grid"), config).expect("factors");
    let n = lap.n as usize;
    let b: Vec<f64> = (0..n).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
    let x = factor.solve(&b).expect("solves");
    let probe = x
        .iter()
        .enumerate()
        .map(|(i, &value)| value * ((i * 104_729) % 17) as f64)
        .sum();
    let norm = x.iter().map(|&value| value * value).sum();
    [probe, norm]
}

#[rstest]
#[case::ac_floating(false, None, [343.4512488486909, 14155.969601523408])]
#[case::ac_grounded(true, None, [-2772.066630121868, 8932.801889440036])]
#[case::ac2_floating(false, Some(2), [91.45211538676477, 12869.429864654201])]
#[case::ac2_grounded(true, Some(2), [-6000.061115101542, 9806.65335123535])]
fn factor_matches_recorded_fingerprint(
    #[case] grounded: bool,
    #[case] split_merge: Option<u32>,
    #[case] recorded: [f64; 2],
) {
    let got = fingerprint(grounded, split_merge);
    for (got, recorded) in got.into_iter().zip(recorded) {
        assert!(
            (got - recorded).abs() <= RELATIVE_TOLERANCE * recorded.abs(),
            "{got:e} vs recorded {recorded:e}"
        );
    }
}
