#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{factorize, factorize_with, Config, CsrRef};
use laplacian_prop::{laplacian_csr_strategy, rhs_for_dimension};
use proptest::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

proptest! {
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
}
