mod common;
use common::*;

use approx_chol::{Config, Builder, CsrRef};

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
    let csr = CsrRef::new_unchecked(&rp, &ci, &vals, n);
    let factor = Builder::new(Config::default())
        .build(csr)
        .unwrap();
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
        .build(lap.as_csr())
        .unwrap();
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
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).expect("valid csr");
    let factor = Builder::<f32>::new(Config::default())
        .build(csr)
        .expect("factorization should succeed");
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
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).expect("valid csr");
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .expect("factorization should succeed");
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
        .build(lap.as_csr())
        .unwrap();

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    let mut work = vec![0.0; n];
    factor.solve_into(&rhs, &mut work);

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
        .build(lap.as_csr())
        .unwrap();

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    let mut with_proj = vec![0.0; n];
    factor.solve_into_with_projection(&rhs, &mut with_proj, true);

    let mut no_proj = vec![0.0; n];
    factor.solve_into_with_projection(&rhs, &mut no_proj, false);

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
        .build(lap.as_csr())
        .unwrap();

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    // solve_into reference
    let mut work = vec![0.0; n];
    factor.solve_into(&rhs, &mut work);

    // allocating solve()
    let result = factor.solve(&rhs);

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
        .build(lap.as_csr())
        .unwrap();

    let n = factor.n();
    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;

    // solve_into_with_projection(false) — copies rhs, then forward+backward
    let mut reference = vec![0.0; n];
    factor.solve_into_with_projection(&rhs, &mut reference, false);

    // solve_in_place — caller does the copy, then forward+backward
    let mut in_place = vec![0.0; n];
    in_place[..rhs.len()].copy_from_slice(&rhs);
    factor.solve_in_place(&mut in_place);

    for (a, b) in reference.iter().zip(in_place.iter()) {
        assert!(
            (a - b).abs() < 1e-14,
            "solve_in_place must match solve_into_with_projection(false): {a} vs {b}"
        );
    }
}

#[test]
#[should_panic(expected = "work buffer too small")]
fn solve_into_panics_on_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr())
        .unwrap();

    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;
    let mut work = vec![0.0; factor.n().saturating_sub(1)];
    factor.solve_into(&rhs, &mut work);
}

#[test]
#[should_panic(expected = "work buffer too small")]
fn solve_in_place_panics_on_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr())
        .unwrap();

    let mut y = vec![0.0; factor.n().saturating_sub(1)];
    factor.solve_in_place(&mut y);
}
