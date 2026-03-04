#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use grid::grid_laplacian;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::Config;

fn run_smoke_case(rows: usize, cols: usize, config: Config) {
    let lap = grid_laplacian(rows, cols);
    let builder = Builder::new(config);
    let factor = builder
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut work = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut work)
        .or_panic("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
    assert!(work.iter().any(|x| x.abs() > 1e-12));
}

#[test]
fn smoke_medium_grid_ac() {
    for size in [70, 100] {
        run_smoke_case(size, size, Config::default());
    }
}

#[test]
fn smoke_medium_grid_ac2() {
    let config = Config {
        seed: 42,
        split_merge: Some(2),
    };
    for size in [70, 100] {
        run_smoke_case(size, size, config);
    }
}
