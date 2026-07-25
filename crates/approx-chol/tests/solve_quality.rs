#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use grid::grid_laplacian;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Backend, Config, CsrRef, ExactFailure, SolveError};

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
// solve_in_place skips projection (differs from solve_into)
// ---------------------------------------------------------------------------

#[test]
fn solve_in_place_skips_projection() {
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
        .solve_into(&rhs, &mut with_proj)
        .or_panic("solve_into should succeed");

    let mut no_proj = vec![0.0; n];
    no_proj[..rhs.len()].copy_from_slice(&rhs);
    factor
        .solve_in_place(&mut no_proj)
        .or_panic("solve_in_place should succeed");

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

    // solve_into reference (work buffer is full augmented dimension)
    let mut work = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut work)
        .or_panic("solve_into should succeed");

    // allocating solve() returns original_n elements
    let result = factor.solve(&rhs).or_panic("solve should succeed");

    assert_eq!(result.len(), factor.original_n());
    for (a, b) in result.iter().zip(work[..factor.original_n()].iter()) {
        assert_eq!(*a, *b, "allocating solve() must match solve_into()");
    }
}

#[test]
fn solve_returns_original_n_for_sddm() {
    let (rp, ci, vals, n) = sddm_4();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");

    assert!(
        factor.n() > n as usize,
        "SDDM should trigger Gremban augmentation"
    );
    assert_eq!(factor.original_n(), n as usize);

    let mut rhs = vec![0.0; n as usize];
    rhs[0] = 1.0;
    rhs[(n as usize) - 1] = -1.0;

    let result = factor.solve(&rhs).or_panic("solve should succeed");
    assert_eq!(
        result.len(),
        n as usize,
        "solve() must return original_n elements, not augmented"
    );
    assert!(
        result.iter().all(|x| x.is_finite()),
        "solution has non-finite values"
    );
}

// ---------------------------------------------------------------------------
// SDDM solve accuracy (issue #35): solve(b) must equal M^-1 b
//
// A *diagonal* SDDM matrix augments to a star graph (aux vertex + leaves),
// which is a tree, so approximate Cholesky is EXACT here — no sampling error.
// That isolates the Gremban *recovery* (grounding) from AC approximation, so we
// can assert against the closed-form inverse x_i = b_i / d_i with tight tol.
// The RHS deliberately has a non-zero sum, which is exactly the case the old
// global zero-mean projection got wrong.
// ---------------------------------------------------------------------------

/// Diagonal SDDM matrix `diag(2, 3, 5)` as CSR (row sums 2, 3, 5 -> augmented).
fn diagonal_sddm() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    (vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 5.0], 3)
}

#[test]
fn sddm_solve_matches_dense_inverse_nonzero_sum_rhs() {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");

    assert!(factor.n() > n as usize, "diagonal SDDM should be augmented");

    // Both a strictly-positive and a mixed-sign RHS, each with a non-zero sum —
    // the case the old global zero-mean recovery corrupted. For diagonal M the
    // closed-form solution is x_i = b_i / d_i.
    for b in [[1.0_f64, 2.0, 3.0], [1.0, -2.0, 4.0]] {
        let x = factor.solve(&b).or_panic("solve should succeed");
        assert_eq!(x.len(), n as usize);
        for i in 0..n as usize {
            let want = b[i] / vals[i];
            assert!(
                (x[i] - want).abs() < 1e-9,
                "b={b:?}: x[{i}] = {:.6}, expected {want:.6} (M^-1 b); \
                 solve returned a non-solution of the SDDM system",
                x[i]
            );
        }
    }
}

#[test]
fn solve_into_reports_rhs_too_long() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let rhs = vec![0.0; factor.n() + 1];
    let mut work = vec![0.0; factor.n()];
    let err = factor
        .solve_into(&rhs, &mut work)
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
fn solve_into_rejects_rhs_longer_than_original_for_augmented_factor() {
    // For an augmented SDDM factor n() == original_n() + 1, and the aux slot is
    // internal scratch. A RHS of length original_n + 1 must be rejected, not
    // silently accepted with its last entry overwritten by the grounding setup.
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(factor.n(), factor.original_n() + 1, "SDDM augments by one");

    let rhs = vec![0.0; factor.original_n() + 1]; // == factor.n(): the aux slot
    let mut work = vec![0.0; factor.n()];
    let err = factor
        .solve_into(&rhs, &mut work)
        .err_or_panic("rhs longer than original dimension must fail");
    assert!(matches!(err, SolveError::RhsLengthExceedsFactor { .. }));
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

// A grounded block's anchored solve *is* the SDDM solution, so solve_in_place and
// solve_into must agree — and must do so for either backend. Before the anchored
// invariant this held only for the exact backend.
#[test]
fn grounded_raw_solve_matches_recovered_solve_on_both_backends() {
    let row_ptrs = [0u32, 2, 4];
    let columns = [0u32, 1, 0, 1];
    let values = [2.0, -1.0, -1.0, 2.0];
    for backend in [
        Backend::Approximate,
        Backend::ExactBelow {
            max_dim: 24,
            on_failure: ExactFailure::FallBackToApproximate,
        },
    ] {
        let factor = Builder::<f64>::new(Config {
            backend,
            ..Config::default()
        })
        .build(CsrRef::new(&row_ptrs, &columns, &values, 2).or_panic("valid CSR"))
        .or_panic("factorization should succeed");

        let n = factor.n();
        assert_eq!(n, 3, "strictly dominant input must gain a ground vertex");

        let rhs = [1.0, -2.0];
        let mut recovered = vec![0.0; n];
        factor
            .solve_into(&rhs, &mut recovered)
            .or_panic("solve_into should succeed");

        let mut raw = vec![0.0; n];
        raw[..rhs.len()].copy_from_slice(&rhs);
        factor
            .solve_in_place(&mut raw)
            .or_panic("solve_in_place should succeed");

        assert_eq!(raw[..2], recovered[..2], "backend {backend:?}");
        assert_eq!(
            raw[n - 1],
            0.0,
            "backend {backend:?}: ground must be pinned"
        );
    }
}
