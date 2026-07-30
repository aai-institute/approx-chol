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

#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/residual.rs"]
mod residual;

use approx_chol::{factorize_with, Config, CsrRef, Factor};
use laplacian_prop::{
    interleaved_components_strategy, permutation_strategy, permute_csr, LaplacianCsr,
};
use panic_ok::OrPanic;
use proptest::prelude::*;
use residual::relative_residual_over;

/// The exact arm's error here is roundoff — measured worst 5.6e-16 relative — while a gauge
/// read off a vertex moves the solution by order 0.1. Mixed abs+rel because a relative-only
/// bound is unstable on the near-zero entries a zero-mean solution always has.
fn agrees(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-10 + 1e-8 * want.abs()
}

/// The exact arm, and a check that it really was exact: a block reaching an unusable pivot
/// falls back to the sampler by default, which would quietly make this the approximate arm.
fn solve_exactly(csr: CsrRef<'_>, rhs: &[f64]) -> Vec<f64> {
    let factor: Factor<f64> = factorize_with(csr, Config::default()).or_panic("factorization");
    assert!(
        factor.fallbacks().is_empty(),
        "block fell back to the sampler: {:?}",
        factor.fallbacks()
    );
    factor.solve(rhs).or_panic("solve")
}

fn solve_generated(csr: &LaplacianCsr, rhs: &[f64]) -> Vec<f64> {
    let (row_ptrs, col_indices, values, n) = csr;
    let view = CsrRef::new(row_ptrs, col_indices, values, *n).or_panic("generated CSR is valid");
    solve_exactly(view, rhs)
}

/// Component `part` holds the vertices congruent to it, so a stride reaches exactly one.
fn per_component_zero_mean(rhs: &mut [f64], parts: usize) {
    for part in 0..parts {
        let mean = rhs[part..].iter().step_by(parts).sum::<f64>()
            / rhs[part..].iter().step_by(parts).count() as f64;
        for value in rhs[part..].iter_mut().step_by(parts) {
            *value -= mean;
        }
    }
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

        for (vertex, &want) in base.iter().enumerate() {
            prop_assert!(
                agrees(got[p[vertex]], want),
                "x[{vertex}] -> x'[{}]: {:e} vs {want:e} (p={p:?})",
                p[vertex],
                got[p[vertex]]
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
        // A floating component answers only a zero-sum right-hand side exactly; anything
        // else leaves its mean in the residual and would mask a real error.
        per_component_zero_mean(&mut rhs, parts);
        prop_assume!(rhs.iter().map(|value| value * value).sum::<f64>().sqrt() > 1e-9);

        let (row_ptrs, col_indices, values, n) = &csr;
        let view = CsrRef::new(row_ptrs, col_indices, values, *n).or_panic("generated CSR");
        let x = solve_exactly(view, &rhs);
        let relative = relative_residual_over(view, &x, &rhs, 0..rhs.len());
        prop_assert!(relative < 1e-9, "components left residual {relative:e}");
    }
}
