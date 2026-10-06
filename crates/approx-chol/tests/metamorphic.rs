//! Relations between *related* inputs, which is what pins arithmetic that
//! `property_factorization.rs` would accept as consistently wrong in the same way.
//!
//! Equivariance is asserted on the exact arm alone, because `BlockFactorizer::factor`
//! restarts the sampler at `component.first_vertex()` — a global vertex label — so
//! relabeling redraws every clique edge: the approximate arm's solution moves 8.6-19% and
//! its residual up to 5.1x across 12 seeds. No tolerance both admits that and rejects a
//! broken permutation.
//!
//! Scaling equivariance lives in `scale_invariance.rs`, over 20 exponents spanning the
//! augmentation floor.

#[path = "common/grid.rs"]
mod grid;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/residual.rs"]
mod residual;

use approx_chol::{factorize, factorize_with, Config, CsrRef, Factor};
use grid::grid_laplacian;
use laplacian_prop::{
    interleaved_components_strategy, permutation_strategy, permute_csr, LaplacianCsr,
};
use proptest::prelude::*;
use residual::relative_residual_over;

/// The exact arm, and a check that it really was exact: a block reaching an unusable pivot
/// falls back to the sampler by default, which would quietly make this the approximate arm.
fn solve_exactly(csr: CsrRef<'_>, rhs: &[f64]) -> Vec<f64> {
    let factor: Factor<f64> = factorize_with(csr, Config::default()).expect("factorization");
    assert!(
        factor.fallbacks().is_empty(),
        "block fell back to the sampler: {:?}",
        factor.fallbacks()
    );
    factor.solve(rhs).expect("solve")
}

fn solve_generated(csr: &LaplacianCsr, rhs: &[f64]) -> Vec<f64> {
    let (row_ptrs, col_indices, values, n) = csr;
    let view = CsrRef::new(row_ptrs, col_indices, values, *n).expect("generated CSR is valid");
    solve_exactly(view, rhs)
}

fn interleaved_case() -> impl Strategy<Value = (LaplacianCsr, usize, Vec<f64>, Vec<usize>)> {
    interleaved_components_strategy().prop_flat_map(|(csr, parts)| {
        let n = csr.3 as usize;
        (
            Just(csr),
            Just(parts),
            prop::collection::vec(-10.0f64..10.0, n),
            permutation_strategy(n),
        )
    })
}

proptest! {
    /// `(P A Pᵀ)(P x) = P b`. Components interleaved by construction are what drive a
    /// non-identity `Permutation` through the solve at all: an unconstrained Laplacian
    /// strategy reaches one in about 3 cases of 512, and then almost always as its own
    /// inverse.
    #[test]
    fn permuting_interleaved_components_permutes_the_solution(
        (csr, _parts, rhs, p) in interleaved_case()
    ) {
        let base = solve_generated(&csr, &rhs);
        let mut permuted_rhs = vec![0.0; rhs.len()];
        for (vertex, &value) in rhs.iter().enumerate() {
            permuted_rhs[p[vertex]] = value;
        }
        let got = solve_generated(&permute_csr(&csr, &p), &permuted_rhs);

        // Exact-arm roundoff (worst 5.6e-16) vs a gauge's ~0.1; abs+rel for near-zero entries.
        for (vertex, &want) in base.iter().enumerate() {
            let got = got[p[vertex]];
            prop_assert!(
                (got - want).abs() <= 1e-10 + 1e-8 * want.abs(),
                "x[{vertex}] -> x'[{}]: {got:e} vs {want:e} (p={p:?})",
                p[vertex]
            );
        }
    }

    /// Equivariance above is self-consistency, so it survives anything wrong in the same way
    /// both times — scaling every recovered solution by two passes it. This is the direct
    /// claim on that path: the components are *solved*, not merely relabelled alike.
    #[test]
    fn interleaved_components_are_solved_not_just_relabelled_consistently(
        (csr, parts, mut rhs, _p) in interleaved_case()
    ) {
        // Only a zero-sum rhs per floating component is solved exactly; a stride of `parts` is one component.
        for part in 0..parts {
            let mean = rhs[part..].iter().step_by(parts).sum::<f64>()
                / rhs[part..].iter().step_by(parts).count() as f64;
            for value in rhs[part..].iter_mut().step_by(parts) {
                *value -= mean;
            }
        }
        prop_assume!(rhs.iter().map(|value| value * value).sum::<f64>().sqrt() > 1e-9);

        let (row_ptrs, col_indices, values, n) = &csr;
        let view = CsrRef::new(row_ptrs, col_indices, values, *n).expect("generated CSR");
        let x = solve_exactly(view, &rhs);
        let relative = relative_residual_over(view, &x, &rhs, 0..rhs.len());
        prop_assert!(relative < 1e-9, "components left residual {relative:e}");
    }
}

/// A floating block solves `b - mean(b)`, so adding a constant to `b` must leave `x`.
#[test]
fn a_constant_added_to_a_floating_rhs_leaves_the_solution() {
    let grid = grid_laplacian(100, 100);
    let factor: Factor<f64> = factorize(grid.as_csr().expect("grid CSR")).expect("factorization");
    // Dyadic, so `value + 2^30` is exact and only the solve's own sums can lose the constant.
    let rhs: Vec<f64> = (0..grid.n as usize)
        .map(|i| (i * 37 % 2001) as f64 / 1024.0 - 1.0)
        .collect();
    let shifted: Vec<f64> = rhs.iter().map(|value| value + 2f64.powi(30)).collect();
    let base = factor.solve(&rhs).expect("solve");
    let got = factor.solve(&shifted).expect("solve");

    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let difference: Vec<f64> = got.iter().zip(&base).map(|(g, b)| g - b).collect();
    let moved = norm(&difference) / norm(&base);
    // Compensated sums leave 1.9e-5, a plain fold 3.1e-2.
    assert!(moved < 1e-3, "the constant moved the solution by {moved:e}");
}
