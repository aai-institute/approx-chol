#[path = "common/backends.rs"]
mod backends;
#[path = "common/grounded.rs"]
mod grounded;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/residual.rs"]
mod residual;

use approx_chol::{factorize_with, Backend, Config, CsrRef, Sddm};
use backends::backends;
use grounded::is_grounded;
use laplacian_prop::{
    is_connected, laplacian_csr_strategy, laplacian_with_rhs_strategy,
    one_grounded_component_strategy, per_component_consistent_rhs, rhs_for_dimension,
    sddm_csr_strategy,
};
use proptest::prelude::*;
use residual::relative_residual_over;

/// `x = 0` scores exactly `1`, so this is the weakest bound that still demands a
/// factor beat answering nothing; measured max is `0.74` over seeds `0..96`.
const RESIDUAL_LIMIT: f64 = 1.0;

proptest! {
    // -----------------------------------------------------------------------
    // Solution quality: residual ||Ax - b|| / ||b|| is bounded
    // -----------------------------------------------------------------------

    #[test]
    fn residual_is_bounded(
        ((row_ptrs, col_indices, values, n), rhs) in laplacian_with_rhs_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        // `relative_residual_over` would return NaN, not skip, for a `b` too small to divide by.
        prop_assume!(rhs.iter().map(|b| b * b).sum::<f64>().sqrt() > 1e-15);
        let view = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect("valid CSR");

        for backend in backends() {
        for config in [
            Config { backend, ..Config::default() },
            Config { seed: 7, split_merge: Some(2), backend },
        ] {
            let x = factorize_with(Sddm::try_from(view).expect("an SDDM"), config)
                .expect("factorization")
                .solve(&rhs)
                .expect("solve");
            // A non-finite solve shows up as a non-finite ratio, so this subsumes a finiteness check.
            let relative = relative_residual_over(view, &x, &rhs, 0..rhs.len());
            prop_assert!(
                relative < RESIDUAL_LIMIT,
                "{config:?}: relative residual too large: {relative:.4e}"
            );
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
            let factor = factorize_with(Sddm::try_from(csr).expect("an SDDM"), config).expect("f32 factorization");

            let x = factor.solve(&rhs).expect("f32 solve");
            prop_assert!(
                x.iter().all(|v| v.is_finite()),
                "{backend:?}: f32 solution has non-finite values"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Factor dimensions of the connected-Laplacian strategy.
    // -----------------------------------------------------------------------

    #[test]
    fn laplacian_factor_has_input_dimension_and_no_ground(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(is_connected(&row_ptrs, &col_indices, n));
        for backend in backends() {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("generated CSR must be valid");
            let config = Config { backend, ..Config::default() };
            let factor = factorize_with(Sddm::try_from(csr).expect("an SDDM"), config).expect("factorization should succeed");

            prop_assert_eq!(factor.n(), n as usize);
            // A pure Laplacian has no surplus, so it is not augmented.
            prop_assert!(
                !is_grounded(&factor),
                "pure Laplacian should not trigger Gremban augmentation"
            );
        }
    }

    // -----------------------------------------------------------------------
    // SDDM matrices (Gremban augmentation path)
    // -----------------------------------------------------------------------

    #[test]
    fn sddm_factor_is_augmented_and_solves_finitely(
        (row_ptrs, col_indices, values, n) in sddm_csr_strategy()
    ) {
        for backend in backends() {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("valid SDDM CSR");
            let config = Config { backend, ..Config::default() };
            let factor = factorize_with(Sddm::try_from(csr).expect("an SDDM"), config).expect("factorization");

            prop_assert_eq!(factor.n(), n as usize, "n must match input dimension");
            // Every row carries surplus, so the ground joins all of them into one block.
            prop_assert!(is_grounded(&factor), "SDDM should trigger Gremban augmentation");

            let x = factor.solve(&rhs_for_dimension(n as usize)).expect("solve");
            prop_assert!(
                x.iter().all(|v| v.is_finite()),
                "{backend:?}: SDDM solution has non-finite values"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Determinism: same seed + same input → identical output
    // -----------------------------------------------------------------------

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
            let x1 = factorize_with(Sddm::try_from(csr1).expect("an SDDM"), config).expect("factorize 1")
                .solve(&rhs).expect("solve 1");

            let csr2 = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("valid CSR");
            let x2 = factorize_with(Sddm::try_from(csr2).expect("an SDDM"), config).expect("factorize 2")
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

    // -----------------------------------------------------------------------
    // A grounded block among floating ones is still recognized as grounded
    // -----------------------------------------------------------------------

    /// The ground vertex is appended above every real vertex, so it is a block's last
    /// only once the block is sorted — the DFS reaches it early, since ingestion adds
    /// its edge after the component's own. Leave the block unsorted and
    /// the builder reads the grounded block as floating, which solves the wrong
    /// system on those rows. Only reachable when the ground vertex does not bridge the
    /// components, i.e. when exactly one of them carries diagonal surplus.
    #[test]
    fn a_grounded_block_beside_floating_ones_solves_its_own_rows(
        ((row_ptrs, col_indices, values, n), parts) in one_grounded_component_strategy()
    ) {
        let rhs = per_component_consistent_rhs(n as usize, parts);
        let view = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect("valid CSR");
        let config = Config { seed: 11, backend: Backend::default(), ..Default::default() };
        let x = factorize_with(Sddm::try_from(view).expect("an SDDM"), config).expect("factorize").solve(&rhs).expect("solve");

        let residual = relative_residual_over(view, &x, &rhs, 0..rhs.len());
        prop_assert!(
            residual < 1e-9,
            "grounded block solved as floating: residual {residual:e}"
        );
    }
}
