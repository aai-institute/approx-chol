#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use grid::grid_laplacian;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::{Builder, Config, CsrRef, SolveError};

// ---------------------------------------------------------------------------
// Gremban augmentation: SDDM vs pure Laplacian
// ---------------------------------------------------------------------------

/// Build a 4x4 SDDM matrix (positive diagonal row sums — strictly diagonally dominant).
///
/// We use the path Laplacian (0-1-2-3) and add 1.0 to each diagonal entry,
/// so row sums are positive (1 for interior, 2 for boundary).
fn sddm_4() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    let row_ptrs = vec![0u32, 2, 5, 8, 10];
    let col_indices = vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    //                     diag  off  off  diag off  off  diag off  off  diag
    let values = vec![2.0f64, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
    let n = 4u32;
    (row_ptrs, col_indices, values, n)
}

#[test]
fn gremban_augmented_for_sddm() {
    let (rp, ci, vals, n) = sddm_4();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    // Gremban augmentation adds one extra vertex for SDDM matrices
    assert!(
        factor.n() > n as usize,
        "expected factor.n() > {n}, got {}",
        factor.n()
    );
}

#[test]
fn no_augmentation_for_pure_laplacian() {
    let lap = grid_laplacian(3, 3); // pure Laplacian: zero row sums
    let original_n = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");
    assert_eq!(
        factor.n(),
        original_n,
        "pure Laplacian should not be augmented"
    );
}

#[test]
fn near_zero_surplus_f32_does_not_augment() {
    let eps = 5e-7_f32;
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [1.0_f32 + eps, -1.0_f32, -1.0_f32, 1.0_f32 + eps];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).or_panic("valid csr");
    let factor = Builder::<f32>::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(
        factor.n(),
        2,
        "roundoff-scale row-sum drift should not trigger augmentation for f32"
    );
}

#[test]
fn near_zero_surplus_f64_does_not_augment() {
    let eps = 5e-11_f64;
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [1.0_f64 + eps, -1.0_f64, -1.0_f64, 1.0_f64 + eps];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).or_panic("valid csr");
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(
        factor.n(),
        2,
        "roundoff-scale row-sum drift should not trigger augmentation for f64"
    );
}

// ---------------------------------------------------------------------------
// Solve quality: grid Laplacian
// ---------------------------------------------------------------------------

#[test]
fn solve_into_gives_finite_nontrivial_solution() {
    let lap = grid_laplacian(8, 8);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    let mut work = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut work)
        .or_panic("solve_into should succeed");

    assert!(
        work.iter().all(|x| x.is_finite()),
        "solution has non-finite values"
    );
    assert!(
        work.iter().any(|x| x.abs() > 1e-12),
        "solution is trivially zero"
    );
}

// ---------------------------------------------------------------------------
// solve_into_with_projection(false) gives different result than with projection
// ---------------------------------------------------------------------------

#[test]
fn no_projection_differs_from_projection() {
    let lap = grid_laplacian(5, 5);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    let mut with_proj = vec![0.0; n];
    factor
        .solve_into_with_projection(&rhs, &mut with_proj, true)
        .or_panic("solve_into_with_projection should succeed");

    let mut no_proj = vec![0.0; n];
    factor
        .solve_into_with_projection(&rhs, &mut no_proj, false)
        .or_panic("solve_into_with_projection should succeed");

    // The zero-mean projection should shift the solution; results must differ
    let any_different = with_proj
        .iter()
        .zip(no_proj.iter())
        .any(|(a, b)| (a - b).abs() > 1e-14);
    assert!(any_different, "expected projection to change the solution");
}

// ---------------------------------------------------------------------------
// Allocating solve() gives same result as solve_into()
// ---------------------------------------------------------------------------

#[test]
fn allocating_solve_matches_solve_into() {
    let lap = grid_laplacian(6, 6);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    // solve_into reference
    let mut work = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut work)
        .or_panic("solve_into should succeed");

    // allocating solve()
    let result = factor.solve(&rhs).or_panic("solve should succeed");

    assert_eq!(result.len(), n);
    for (a, b) in result.iter().zip(work.iter()) {
        assert_eq!(*a, *b, "allocating solve() must match solve_into()");
    }
}

#[test]
fn solve_in_place_matches_no_projection() {
    let lap = grid_laplacian(5, 5);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    // solve_into_with_projection(false) — copies rhs, then forward+backward
    let mut reference = vec![0.0; n];
    factor
        .solve_into_with_projection(&rhs, &mut reference, false)
        .or_panic("solve_into_with_projection should succeed");

    // solve_in_place — caller does the copy, then forward+backward
    let mut in_place = vec![0.0; n];
    in_place[..rhs.len()].copy_from_slice(&rhs);
    factor
        .solve_in_place(&mut in_place)
        .or_panic("solve_in_place should succeed");

    for (a, b) in reference.iter().zip(in_place.iter()) {
        assert!(
            (a - b).abs() < 1e-14,
            "solve_in_place must match solve_into_with_projection(false): {a} vs {b}"
        );
    }
}

#[test]
fn try_solve_matches_solve() {
    let lap = grid_laplacian(6, 6);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    let x = factor.solve(&rhs).or_panic("solve should succeed");
    let x_try = factor.try_solve(&rhs).or_panic("try_solve should succeed");
    assert_eq!(x, x_try, "try_solve must match solve");
}

#[test]
fn try_solve_into_reports_rhs_too_long() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let rhs = vec![0.0; factor.n() + 1];
    let mut work = vec![0.0; factor.n()];
    let err = factor
        .try_solve_into(&rhs, &mut work)
        .err_or_panic("rhs longer than factor dimension must fail");
    assert!(matches!(
        err,
        SolveError::RhsLengthExceedsFactor {
            rhs_len: _,
            factor_dim: _
        }
    ));
}

#[test]
fn try_solve_into_reports_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;
    let mut work = vec![0.0; factor.n().saturating_sub(1)];

    let err = factor
        .try_solve_into(&rhs, &mut work)
        .err_or_panic("short work buffer must fail");
    assert!(matches!(
        err,
        SolveError::WorkBufferTooSmall {
            work_len: _,
            factor_dim: _
        }
    ));
}

#[test]
fn try_solve_in_place_reports_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut y = vec![0.0; factor.n().saturating_sub(1)];
    let err = factor
        .try_solve_in_place(&mut y)
        .err_or_panic("short in-place work buffer must fail");
    assert!(matches!(
        err,
        SolveError::WorkBufferTooSmall {
            work_len: _,
            factor_dim: _
        }
    ));
}

#[test]
fn solve_into_reports_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;
    let mut work = vec![0.0; factor.n().saturating_sub(1)];
    let err = factor
        .solve_into(&rhs, &mut work)
        .err_or_panic("short work buffer must fail");
    assert!(matches!(
        err,
        SolveError::WorkBufferTooSmall {
            work_len: _,
            factor_dim: _
        }
    ));
}

#[test]
fn solve_in_place_reports_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut y = vec![0.0; factor.n().saturating_sub(1)];
    let err = factor
        .solve_in_place(&mut y)
        .err_or_panic("short in-place work buffer must fail");
    assert!(matches!(
        err,
        SolveError::WorkBufferTooSmall {
            work_len: _,
            factor_dim: _
        }
    ));
}
