#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/residual.rs"]
mod residual;
use grid::grid_laplacian;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;
use residual::relative_residual_over;

use approx_chol::{
    factorize_with, Backend, Config, DenseFailure, Error, ExactFailure, Fallback, UnusablePivot,
};
use grid::GridLaplacian;

fn side_by_side(left: &GridLaplacian, right: &GridLaplacian) -> GridLaplacian {
    let offset = left.n;
    let mut row_ptrs = left.row_ptrs.clone();
    let nnz = *row_ptrs.last().expect("row pointers are never empty");
    row_ptrs.extend(right.row_ptrs[1..].iter().map(|&end| end + nnz));
    let mut col_indices = left.col_indices.clone();
    col_indices.extend(right.col_indices.iter().map(|&column| column + offset));
    let mut values = left.values.clone();
    values.extend_from_slice(&right.values);
    GridLaplacian {
        row_ptrs,
        col_indices,
        values,
        n: offset + right.n,
    }
}

#[test]
fn a_claimed_block_solves_exactly_where_elimination_does_not() {
    let lap = grid_laplacian(4, 4);
    let csr = lap.as_csr().or_panic("valid CSR");
    let mut b = vec![0.0; 16];
    b[0] = 1.0;
    b[15] = -1.0;

    let exact = factorize_with(csr, Config::default()).or_panic("exact factorization");
    let approximate = factorize_with(
        csr,
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    )
    .or_panic("approximate factorization");

    assert!(
        exact.fallbacks().is_empty(),
        "a block that factors reports nothing: {:?}",
        exact.fallbacks()
    );

    let rows = 0..b.len();
    let exact = relative_residual_over(csr, &exact.solve(&b).or_panic("solve"), &b, rows.clone());
    let approximate =
        relative_residual_over(csr, &approximate.solve(&b).or_panic("solve"), &b, rows);

    assert!(
        exact < 1e-12,
        "claimed block did not solve exactly: {exact:e}"
    );
    assert!(
        approximate > 1e-3,
        "fixture does not separate the backends: {approximate:e}"
    );
}

/// The fixture above separates the backends, so a block claimed after all changes
/// the solve rather than passing unnoticed.
#[test]
fn a_bound_of_zero_claims_no_block() {
    let lap = grid_laplacian(4, 4);
    let csr = lap.as_csr().or_panic("valid CSR");
    let mut b = vec![0.0; 16];
    b[0] = 1.0;
    b[15] = -1.0;

    let solve = |backend| {
        factorize_with(
            csr,
            Config {
                backend,
                ..Config::default()
            },
        )
        .or_panic("factorization")
        .solve(&b)
        .or_panic("solve")
    };

    assert_eq!(
        solve(Backend::ExactBelow {
            max_dim: 0,
            on_failure: ExactFailure::FallBackToApproximate,
        }),
        solve(Backend::Approximate),
        "a zero bound claimed a block"
    );
}

#[test]
fn routing_is_decided_per_block() {
    let small = grid_laplacian(3, 3);
    let large = grid_laplacian(7, 7);
    let (small_n, large_n) = (small.n as usize, large.n as usize);
    let lap = side_by_side(&small, &large);
    let csr = lap.as_csr().or_panic("valid CSR");

    let mut b = vec![0.0; small_n + large_n];
    b[0] = 1.0;
    b[small_n - 1] = -1.0;
    b[small_n] = 1.0;
    b[small_n + large_n - 1] = -1.0;

    let factor = factorize_with(csr, Config::default()).or_panic("factorization");
    let x = factor.solve(&b).or_panic("solve");

    let claimed = relative_residual_over(csr, &x, &b, 0..small_n);
    let approximated = relative_residual_over(csr, &x, &b, small_n..small_n + large_n);
    assert!(
        claimed < 1e-12,
        "small component was not claimed: {claimed:e}"
    );
    assert!(
        approximated > 1e-3,
        "large component was claimed too: {approximated:e}"
    );
}

fn cancelling_path() -> GridLaplacian {
    let (big, small) = (f64::powi(2.0, 500), f64::powi(2.0, -500));
    GridLaplacian {
        row_ptrs: vec![0, 2, 5, 7],
        col_indices: vec![0, 1, 0, 1, 2, 1, 2],
        values: vec![big, -big, -big, big + small, -small, -small, small],
        n: 3,
    }
}

#[test]
fn a_pivot_lost_to_cancellation_is_reported() {
    let lap = cancelling_path();
    let factor = factorize_with(lap.as_csr().or_panic("valid CSR"), Config::default())
        .or_panic("factorization");

    assert_eq!(
        factor.fallbacks(),
        [Fallback::InvalidPivot(UnusablePivot {
            vertex: 1,
            failure: DenseFailure::NonPositivePivot,
        })],
        "the failing pivot is vertex 1"
    );
    let x = factor.solve(&[1.0, 0.0, -1.0]).or_panic("solve");
    assert!(x.iter().all(|value| value.is_finite()));
}

#[test]
fn a_reported_pivot_is_translated_out_of_block_local_numbering() {
    let grid = grid_laplacian(3, 3);
    let lap = side_by_side(&grid, &cancelling_path());
    let factor = factorize_with(lap.as_csr().or_panic("valid CSR"), Config::default())
        .or_panic("factorization");

    assert_eq!(
        factor.fallbacks(),
        [Fallback::InvalidPivot(UnusablePivot {
            vertex: 10,
            failure: DenseFailure::NonPositivePivot,
        })]
    );
}

#[test]
fn a_failed_pivot_can_be_asked_to_fail_the_factorization() {
    let lap = cancelling_path();
    let error = factorize_with(
        lap.as_csr().or_panic("valid CSR"),
        Config {
            backend: Backend::ExactBelow {
                max_dim: 24,
                on_failure: ExactFailure::Error,
            },
            ..Config::default()
        },
    )
    .err_or_panic("ExactFailure::Error must fail the factorization");

    assert_eq!(
        error,
        Error::DenseFactorizationFailed(UnusablePivot {
            vertex: 1,
            failure: DenseFailure::NonPositivePivot,
        })
    );
}

/// Squaring anything stored beside the factor overflows at these weights and rejects
/// a factor that is fine.
#[test]
fn a_block_whose_weights_square_to_infinity_still_factors() {
    let big = 1e200;
    let lap = GridLaplacian {
        row_ptrs: vec![0, 2, 5, 7],
        col_indices: vec![0, 1, 0, 1, 2, 1, 2],
        values: vec![big, -big, -big, 2.0 * big, -big, -big, big],
        n: 3,
    };
    let factor = factorize_with(lap.as_csr().or_panic("valid CSR"), Config::default())
        .or_panic("factorization");

    assert!(factor.fallbacks().is_empty());
    let x = factor.solve(&[1.0, 0.0, -1.0]).or_panic("solve");
    assert!(x.iter().all(|value| value.is_finite()), "{x:?}");
}
