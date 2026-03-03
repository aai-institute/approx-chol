use approx_chol::{factorize, factorize_with, Config, CsrRef};
use proptest::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn build_laplacian_csr(n: usize, edge_weights: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    let mut dense = vec![0.0_f64; n * n];
    let mut edge_pos = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let w = edge_weights[edge_pos] as f64;
            edge_pos += 1;
            if w <= 0.0 {
                continue;
            }
            dense[i * n + j] -= w;
            dense[j * n + i] -= w;
            dense[i * n + i] += w;
            dense[j * n + j] += w;
        }
    }

    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0u32);
    for i in 0..n {
        for j in 0..n {
            let value = dense[i * n + j];
            if i == j || value != 0.0 {
                col_indices.push(j as u32);
                values.push(value);
            }
        }
        row_ptrs.push(col_indices.len() as u32);
    }

    (row_ptrs, col_indices, values, n as u32)
}

fn laplacian_csr_strategy() -> impl Strategy<Value = (Vec<u32>, Vec<u32>, Vec<f64>, u32)> {
    (1usize..=8).prop_flat_map(|n| {
        let pair_count = n * (n - 1) / 2;
        prop::collection::vec(0u8..=4, pair_count)
            .prop_map(move |edge_weights| build_laplacian_csr(n, &edge_weights))
    })
}

fn rhs_for_dimension(n: usize) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n];
    if n >= 2 {
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
    }
    rhs
}

proptest! {
    #[test]
    fn default_factorization_solve_is_panic_free_and_finite(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let run = catch_unwind(AssertUnwindSafe(|| {
            let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
                .expect("generated CSR must be valid");
            let factor = factorize(csr).expect("factorization should succeed");
            let rhs = rhs_for_dimension(n as usize);
            let mut work = vec![0.0_f64; factor.n()];
            factor
                .solve_into(&rhs, &mut work)
                .expect("solve_into should succeed");
            work
        }));

        prop_assert!(run.is_ok(), "default factorization or solve panicked");
        let work = run.expect("checked above");
        prop_assert!(work.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn default_solve_matches_solve_into(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .expect("generated CSR must be valid");
        let factor = factorize(csr).expect("factorization should succeed");
        let rhs = rhs_for_dimension(n as usize);

        let from_alloc = factor.solve(&rhs).expect("solve should succeed");
        let mut from_into = vec![0.0_f64; factor.n()];
        factor
            .solve_into(&rhs, &mut from_into)
            .expect("solve_into should succeed");

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
                .expect("generated CSR must be valid");
            let factor = factorize_with(
                csr,
                Config {
                    seed: 7,
                    split_merge: Some(2),
                },
            )
            .expect("AC2 factorization should succeed");
            let rhs = rhs_for_dimension(n as usize);
            let mut work = vec![0.0_f64; factor.n()];
            factor
                .solve_into(&rhs, &mut work)
                .expect("solve_into should succeed");
            work
        }));

        prop_assert!(run.is_ok(), "AC2 factorization or solve panicked");
        let work = run.expect("checked above");
        prop_assert!(work.iter().all(|x| x.is_finite()));
    }
}
