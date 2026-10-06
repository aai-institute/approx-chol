#[path = "common/backends.rs"]
mod backends;
#[path = "common/factor.rs"]
mod factor;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/residual.rs"]
mod residual;

use approx_chol::{Config, CsrRef, Sddm};
use backends::backends;
use factor::factor;
use laplacian_prop::{
    is_connected, laplacian_csr_strategy, laplacian_with_rhs_strategy, rhs_for_dimension,
    sddm_csr_strategy, LaplacianCsr,
};
use proptest::prelude::*;
use residual::relative_residual_over;

/// `x = 0` scores exactly `1`, so this is the weakest bound that still demands a
/// factor beat answering nothing; measured max is `0.74` over seeds `0..96`.
const RESIDUAL_LIMIT: f64 = 1.0;

/// `None` when `b` is too small for the ratio to carry information. A non-finite
/// solve shows up as a non-finite ratio, so this subsumes a separate check.
fn relative_residual(csr: &LaplacianCsr, config: Config, rhs: &[f64]) -> Option<f64> {
    let (row_ptrs, col_indices, values, n) = csr;
    let view = CsrRef::new(row_ptrs, col_indices, values, *n).expect("valid CSR");
    let x = factor(config, view)
        .expect("factorization")
        .solve(rhs)
        .expect("solve");

    // `relative_residual_over` divides by the row range's own norm, so the guard
    // stays here: a `b` too small to divide by would come back NaN, not `None`.
    let b_norm = rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
    (b_norm > 1e-15).then(|| relative_residual_over(view, &x, rhs, 0..rhs.len()))
}

proptest! {
    #[test]
    fn residual_is_bounded(
        ((row_ptrs, col_indices, values, n), rhs) in laplacian_with_rhs_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        let csr = (row_ptrs, col_indices, values, n);

        for backend in backends() {
        for config in [
            Config { backend, ..Config::default() },
            Config { seed: 7, split_merge: Some(2), backend },
        ] {
            if let Some(relative) = relative_residual(&csr, config, &rhs) {
                prop_assert!(
                    relative < RESIDUAL_LIMIT,
                    "{config:?}: relative residual too large: {relative:.4e}"
                );
            }
        }
        }
    }

    #[test]
    fn f32_solve_is_finite(
        (row_ptrs, col_indices, values_f64, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        let values_f32: Vec<f32> = values_f64.iter().map(|&v| v as f32).collect();
        let rhs: Vec<f32> = rhs_for_dimension(n as usize).iter().map(|&v| v as f32).collect();
        for backend in backends() {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values_f32, n)
                .expect("valid f32 CSR");
            let config = Config { backend, ..Config::default() };
            let factor = factor(config, csr).expect("f32 factorization");

            let x = factor.solve(&rhs).expect("f32 solve");
            prop_assert!(
                x.iter().all(|v| v.is_finite()),
                "{backend:?}: f32 solution has non-finite values"
            );
        }
    }

    #[test]
    fn solve_matches_solve_in_place(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        let rhs = rhs_for_dimension(n as usize);
        for backend in backends() {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("generated CSR must be valid");
            let config = Config { backend, ..Config::default() };
            let factor = factor(config, csr).expect("factorization should succeed");

            prop_assert_eq!(factor.n(), n as usize);
            // Connected and floating, so no scratch: `&mut []` below relies on it.
            prop_assert_eq!(factor.scratch_len(), 0);

            let from_alloc = factor.solve(&rhs).expect("solve should succeed");
            let mut from_into = rhs.clone();
            factor
                .solve_in_place(&mut from_into, &mut [])
                .expect("solve_in_place should succeed");

            // `solve` is `solve_in_place` on a copy, so nothing may differ.
            prop_assert_eq!(from_alloc.len(), from_into.len());
            for (a, b) in from_alloc.iter().zip(from_into.iter()) {
                prop_assert!(a.to_bits() == b.to_bits(), "{backend:?}: {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn grounded_input_solves_finitely(
        (row_ptrs, col_indices, values, n) in sddm_csr_strategy()
    ) {
        for backend in backends() {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("valid SDDM CSR");
            let config = Config { backend, ..Config::default() };
            let factor = factor(config, csr).expect("factorization");

            prop_assert_eq!(factor.n(), n as usize, "n must match input dimension");
            prop_assert!(
                matches!(Sddm::try_from(csr), Ok(Sddm::Grounded(_))),
                "SDDM should be grounded"
            );

            let x = factor.solve(&rhs_for_dimension(n as usize)).expect("solve");
            prop_assert!(
                x.iter().all(|v| v.is_finite()),
                "{backend:?}: SDDM solution has non-finite values"
            );
        }
    }

    #[test]
    fn deterministic_with_fixed_seed(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        let rhs = rhs_for_dimension(n as usize);
        for backend in backends() {
            let config = Config { seed: 42, backend, ..Default::default() };

            let csr1 = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("valid CSR");
            let x1 = factor(config, csr1).expect("factorize 1")
                .solve(&rhs).expect("solve 1");

            let csr2 = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("valid CSR");
            let x2 = factor(config, csr2).expect("factorize 2")
                .solve(&rhs).expect("solve 2");

            prop_assert_eq!(x1.len(), x2.len());
            for (a, b) in x1.iter().zip(x2.iter()) {
                prop_assert!(
                    a.to_bits() == b.to_bits(),
                    "{backend:?}: non-deterministic: {} vs {}", a, b
                );
            }
        }
    }
}
