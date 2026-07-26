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
use approx_chol::{Config, CsrRef, SolveError};
use num_traits::Float;

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

/// Row-sum drift at roundoff scale must not read as diagonal dominance. The
/// floor is precision-dependent, so `eps` comes from the caller.
fn assert_no_augmentation_at_surplus<T: Float + Send + Sync + 'static>(eps: T) {
    let one = T::one();
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [one + eps, -one, -one, one + eps];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).or_panic("valid csr");
    let factor = Builder::<T>::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(
        factor.n(),
        2,
        "roundoff drift must not trigger augmentation"
    );
}

#[test]
fn near_zero_surplus_does_not_augment() {
    assert_no_augmentation_at_surplus(5e-7_f32);
    assert_no_augmentation_at_surplus(5e-11_f64);
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
    factor.solve_in_place(&mut no_proj);

    // The zero-mean projection should shift the solution; results must differ
    let any_different = with_proj
        .iter()
        .zip(no_proj.iter())
        .any(|(a, b)| (a - b).abs() > 1e-14);
    assert!(any_different, "expected projection to change the solution");
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
#[should_panic(expected = "work buffer too small")]
fn short_work_buffer_panics() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    factor.solve_in_place(&mut vec![0.0; factor.n() - 1]);
}
