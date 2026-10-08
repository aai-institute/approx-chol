#[path = "common/grid.rs"]
mod grid;
use grid::grid_laplacian;

use approx_chol::{factorize_with, Config, Sddm};
use rstest::rstest;

/// The scale at which bucket layout and fill-in bookkeeping carry load the property
/// suite's eight-vertex graphs never reach.
#[rstest]
#[case::ac(Config::default())]
#[case::ac2(Config { seed: 42, split_merge: Some(2), ..Config::default() })]
fn smoke_medium_grid(#[case] config: Config) {
    let lap = grid_laplacian(100, 100);
    let factor = factorize_with(
        Sddm::try_from(lap.as_csr().expect("grid_laplacian must build valid CSR"))
            .expect("an SDDM"),
        config,
    )
    .expect("factorization should succeed");

    let n = factor.n();
    assert_eq!(
        factor.n_steps(),
        n - 1,
        "a connected Laplacian pins one vertex"
    );
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let work = factor.solve(&rhs).expect("solve should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
    assert!(work.iter().any(|x| x.abs() > 1e-12));
}
