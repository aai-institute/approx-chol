#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{factorize, factorize_with, Config, CsrRef};
use laplacian_prop::{
    csr_matvec, is_connected, laplacian_csr_strategy, laplacian_with_rhs_strategy, norm2,
    rhs_for_dimension, sddm_csr_strategy,
};
use proptest::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

proptest! {
    // -----------------------------------------------------------------------
    // Existing: panic-freedom and finiteness
    // -----------------------------------------------------------------------

    #[test]
    fn default_factorization_solve_is_panic_free_and_finite(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let run = catch_unwind(AssertUnwindSafe(|| {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .or_panic("generated CSR must be valid");
            let factor = factorize(csr).or_panic("factorization should succeed");
            let rhs = rhs_for_dimension(n as usize);
            let mut work = vec![0.0_f64; factor.n()];
            factor
                .solve_into(&rhs, &mut work)
                .or_panic("solve_into should succeed");
            work
        }));

        prop_assert!(run.is_ok(), "default factorization or solve panicked");
        let work = run.or_panic("checked above");
        prop_assert!(work.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn default_solve_matches_solve_into(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("generated CSR must be valid");
        let factor = factorize(csr).or_panic("factorization should succeed");
        let rhs = rhs_for_dimension(n as usize);

        let from_alloc = factor.solve(&rhs).or_panic("solve should succeed");
        let mut from_into = vec![0.0_f64; factor.n()];
        factor
            .solve_into(&rhs, &mut from_into)
            .or_panic("solve_into should succeed");

        prop_assert_eq!(from_alloc.len(), from_into.len());
        for (a, b) in from_alloc.iter().zip(from_into.iter()) {
            prop_assert!((*a - *b).abs() <= 1e-12);
        }
    }

    #[test]
    fn ac2_factorization_solve_is_panic_free_and_finite(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let run = catch_unwind(AssertUnwindSafe(|| {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .or_panic("generated CSR must be valid");
            let factor = factorize_with(
                csr,
                Config {
                    seed: 7,
                    split_merge: Some(2),
                },
            )
            .or_panic("AC2 factorization should succeed");
            let rhs = rhs_for_dimension(n as usize);
            let mut work = vec![0.0_f64; factor.n()];
            factor
                .solve_into(&rhs, &mut work)
                .or_panic("solve_into should succeed");
            work
        }));

        prop_assert!(run.is_ok(), "AC2 factorization or solve panicked");
        let work = run.or_panic("checked above");
        prop_assert!(work.iter().all(|x| x.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Solution quality: residual ||Ax - b|| / ||b|| is bounded
    // -----------------------------------------------------------------------

    #[test]
    fn ac_residual_is_bounded(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));

        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let factor = factorize(csr).or_panic("factorization");
        let rhs = rhs_for_dimension(n as usize);
        let x = factor.solve(&rhs).or_panic("solve");

        let ax = csr_matvec(&row_ptrs, &col_indices, &values, &x);
        let residual: Vec<f64> = ax.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
        let r_norm = norm2(&residual);
        let b_norm = norm2(&rhs);

        prop_assert!(
            b_norm < 1e-15 || r_norm / b_norm < 100.0,
            "AC relative residual too large: {:.4e} (r_norm={:.4e}, b_norm={:.4e})",
            r_norm / b_norm, r_norm, b_norm
        );
    }

    #[test]
    fn ac2_residual_is_bounded(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));

        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let factor = factorize_with(
            csr,
            Config {
                seed: 7,
                split_merge: Some(2),
            },
        )
        .or_panic("AC2 factorization");
        let rhs = rhs_for_dimension(n as usize);
        let x = factor.solve(&rhs).or_panic("solve");

        let ax = csr_matvec(&row_ptrs, &col_indices, &values, &x);
        let residual: Vec<f64> = ax.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
        let r_norm = norm2(&residual);
        let b_norm = norm2(&rhs);

        prop_assert!(
            b_norm < 1e-15 || r_norm / b_norm < 100.0,
            "AC2 relative residual too large: {:.4e} (r_norm={:.4e}, b_norm={:.4e})",
            r_norm / b_norm, r_norm, b_norm
        );
    }

    #[test]
    fn random_rhs_residual_is_bounded(
        ((row_ptrs, col_indices, values, n), rhs) in laplacian_with_rhs_strategy()
    ) {
        prop_assume!(n >= 2);
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        let b_norm = norm2(&rhs);
        prop_assume!(b_norm > 1e-10);

        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let factor = factorize(csr).or_panic("factorization");
        let x = factor.solve(&rhs).or_panic("solve");

        let ax = csr_matvec(&row_ptrs, &col_indices, &values, &x);
        let residual: Vec<f64> = ax.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
        let r_norm = norm2(&residual);

        prop_assert!(
            r_norm / b_norm < 100.0,
            "random-RHS relative residual too large: {:.4e}", r_norm / b_norm
        );
    }

    // -----------------------------------------------------------------------
    // SDDM matrices (Gremban augmentation path)
    // -----------------------------------------------------------------------

    #[test]
    fn sddm_factorization_is_panic_free_and_finite(
        (row_ptrs, col_indices, values, n) in sddm_csr_strategy()
    ) {
        let run = catch_unwind(AssertUnwindSafe(|| {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .or_panic("valid SDDM CSR");
            let factor = factorize(csr).or_panic("factorization");
            let mut rhs = vec![0.0_f64; n as usize];
            if n >= 2 {
                rhs[0] = 1.0;
                rhs[(n as usize) - 1] = -1.0;
            }
            factor.solve(&rhs).or_panic("solve")
        }));

        prop_assert!(run.is_ok(), "SDDM factorization or solve panicked");
        let x = run.or_panic("checked above");
        prop_assert!(x.iter().all(|v| v.is_finite()), "SDDM solution has non-finite values");
    }

    // -----------------------------------------------------------------------
    // Determinism: same seed + same input → identical output
    // -----------------------------------------------------------------------

    #[test]
    fn deterministic_with_fixed_seed(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let config = Config { seed: 42, ..Default::default() };
        let rhs = rhs_for_dimension(n as usize);

        let csr1 = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let x1 = factorize_with(csr1, config).or_panic("factorize 1")
            .solve(&rhs).or_panic("solve 1");

        let csr2 = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let x2 = factorize_with(csr2, config).or_panic("factorize 2")
            .solve(&rhs).or_panic("solve 2");

        prop_assert_eq!(x1.len(), x2.len());
        for (a, b) in x1.iter().zip(x2.iter()) {
            prop_assert!(
                a.to_bits() == b.to_bits(),
                "non-deterministic: {} vs {}", a, b
            );
        }
    }

    // -----------------------------------------------------------------------
    // Factor dimensions are consistent
    // -----------------------------------------------------------------------

    #[test]
    fn factor_dimensions_are_consistent(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid CSR");
        let factor = factorize(csr).or_panic("factorization");

        prop_assert_eq!(
            factor.original_n(), n as usize,
            "original_n must match input dimension"
        );
        prop_assert!(
            factor.n() >= n as usize,
            "factor.n() must be >= input dimension"
        );
        // Pure Laplacians should not be augmented
        prop_assert_eq!(
            factor.n(), n as usize,
            "pure Laplacian should not trigger Gremban augmentation"
        );
    }

    #[test]
    fn sddm_factor_dimensions_are_consistent(
        (row_ptrs, col_indices, values, n) in sddm_csr_strategy()
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("valid SDDM CSR");
        let factor = factorize(csr).or_panic("factorization");

        prop_assert_eq!(
            factor.original_n(), n as usize,
            "original_n must match input dimension"
        );
        prop_assert!(
            factor.n() > n as usize,
            "SDDM should trigger Gremban augmentation (factor.n() must be > n)"
        );
    }

    // -----------------------------------------------------------------------
    // f32 support
    // -----------------------------------------------------------------------

    #[test]
    fn f32_factorization_is_finite(
        (row_ptrs, col_indices, values_f64, n) in laplacian_csr_strategy()
    ) {
        let values_f32: Vec<f32> = values_f64.iter().map(|&v| v as f32).collect();
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values_f32, n)
            .or_panic("valid f32 CSR");
        let factor = factorize(csr).or_panic("f32 factorization");
        let mut rhs = vec![0.0_f32; n as usize];
        if n >= 2 {
            rhs[0] = 1.0;
            rhs[(n as usize) - 1] = -1.0;
        }
        let x = factor.solve(&rhs).or_panic("f32 solve");
        prop_assert!(x.iter().all(|v| v.is_finite()), "f32 solution has non-finite values");
    }
}
