#[path = "common/grid.rs"]
mod grid;
use grid::grid_laplacian;

use approx_chol::low_level::Builder;
use approx_chol::Config;

fn run_smoke_case(rows: usize, cols: usize, config: Config) {
    let lap = grid_laplacian(rows, cols);
    let builder = Builder::new(config);
    let factor = builder
        .build(lap.as_csr().expect("grid_laplacian must build valid CSR"))
        .expect("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut work = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut work)
        .expect("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
    assert!(work.iter().any(|x| x.abs() > 1e-12));
}

/// The scale at which bucket layout and fill-in bookkeeping carry load the property
/// suite's eight-vertex graphs never reach.
#[test]
fn smoke_medium_grid_ac() {
    run_smoke_case(100, 100, Config::default());
}

#[test]
fn smoke_medium_grid_ac2() {
    run_smoke_case(
        100,
        100,
        Config {
            seed: 42,
            split_merge: Some(2),
            ..Config::default()
        },
    );
}
