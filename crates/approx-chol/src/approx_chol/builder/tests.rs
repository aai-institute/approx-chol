use super::*;
use crate::approx_chol::factorization::exact::NotFactorable;
use crate::test_utils::OrPanic;
use crate::{DenseFailure, ExactFailure, UnusablePivot};

/// Naming the pivot and applying the policy are separate steps, so the cases sweep
/// both: the same failure under either policy, and either failure under `Error`.
#[test]
fn only_an_unusable_pivot_answers_to_the_failure_policy() {
    let pivot = NotFactorable::InvalidPivot {
        pivot: 2,
        failure: DenseFailure::NonPositivePivot,
    };
    let too_large = NotFactorable::WillNotFit { dim: 9 };
    let component = Some(&[0u32, 15, 30][..]);
    let named = |vertex| UnusablePivot {
        vertex,
        failure: DenseFailure::NonPositivePivot,
    };

    let cases = [
        (
            "pivot, falling back",
            pivot,
            ExactFailure::FallBackToApproximate,
            component,
            Ok(Fallback::InvalidPivot(named(30))),
        ),
        (
            "pivot, erroring",
            pivot,
            ExactFailure::Error,
            component,
            Err(Error::DenseFactorizationFailed(named(30))),
        ),
        (
            "pivot of a whole-graph block is already global",
            pivot,
            ExactFailure::FallBackToApproximate,
            None,
            Ok(Fallback::InvalidPivot(named(2))),
        ),
        (
            "will not fit, falling back",
            too_large,
            ExactFailure::FallBackToApproximate,
            component,
            Ok(Fallback::WillNotFit { dim: 9 }),
        ),
        (
            "will not fit, erroring",
            too_large,
            ExactFailure::Error,
            component,
            Ok(Fallback::WillNotFit { dim: 9 }),
        ),
    ];
    for (label, reason, on_failure, vertices, expected) in cases {
        assert_eq!(on_failure.accept(reason.at(vertices)), expected, "{label}");
    }
}

fn make_csr<'a>(indptr: &'a [u32], indices: &'a [u32], data: &'a [f64]) -> CsrRef<'a, f64, u32> {
    CsrRef::new(indptr, indices, data, (indptr.len() - 1) as u32).or_panic("valid CSR test fixture")
}

fn assert_ac2_augmented_solve_is_finite(indptr: &[u32], indices: &[u32], data: &[f64], b: &[f64]) {
    let csr = make_csr(indptr, indices, data);
    for seed in 0..8u64 {
        let factor = Builder::<f64>::new(Config {
            split_merge: Some(2),
            seed,
            ..Config::default()
        })
        .build(csr)
        .unwrap_or_else(|e| panic!("AC2 factorization failed (seed={seed}): {e}"));
        assert!(
            factor.n() > b.len(),
            "fixture must reach the Gremban-augmented path"
        );

        let mut work = vec![0.0f64; factor.n()];
        work[..b.len()].copy_from_slice(b);
        factor
            .solve_in_place(&mut work)
            .or_panic("solve_in_place should succeed");
        assert!(
            work.iter().all(|x| x.is_finite()),
            "seed={seed}: non-finite solve output: {work:?}"
        );
        assert!(
            work.iter().any(|x| x.abs() > 1e-10),
            "seed={seed}: trivially zero solve output: {work:?}"
        );
    }
}

#[test]
fn ac2_one_neighbor_star_keeps_augmentation_mass() {
    assert_ac2_augmented_solve_is_finite(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[5.0, -1.0, -1.0, 6.0, -1.0, -1.0, 5.0],
        &[4.0, 4.0, 4.0],
    );
    assert_ac2_augmented_solve_is_finite(
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[10.0, -1.0, -1.0, 10.0],
        &[9.0, -9.0],
    );
}

#[test]
fn ac2_near_zero_weight_star_skips_fill_sampling() {
    let eps = 1e-300;
    assert_ac2_augmented_solve_is_finite(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[2.0, -eps, -eps, 2.0, -eps, -eps, 2.0],
        &[1.0, -1.0, 1.0],
    );
}

#[test]
fn test_ac_marginally_sdd_laplacian_no_capacity_drift() {
    let indptr: Vec<u32> = vec![0, 4, 8, 13, 15, 19, 25, 29, 34];
    let indices: Vec<u32> = vec![
        0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 3, 7, 2, 3, 4, 5, 6, 7, 0, 1, 4, 5, 6, 7, 4, 5, 6, 7, 2,
        4, 5, 6, 7,
    ];
    let data_f32: Vec<f32> = vec![
        171.20395, -7.728917, -67.65843, -95.81661, -7.728917, 118.25102, -88.94253, -21.579578,
        -67.65843, -88.94253, 266.40173, -34.345234, -75.45554, -34.345234, 34.345234, 102.9335,
        -25.572166, -8.642439, -68.71889, -95.81661, -21.579578, -25.572166, 178.5495, -17.176064,
        -18.405073, -8.642439, -17.176064, 29.55138, -3.732876, -75.45554, -68.71889, -18.405073,
        -3.732876, 166.31238,
    ];

    let csr = CsrRef::new(&indptr, &indices, &data_f32, 8).or_panic("valid marginal-SDD CSR");
    let b: [f32; 8] = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

    for seed in 0..16u64 {
        let config = Config {
            seed,
            ..Default::default()
        };
        let factor = Builder::<f32>::new(config)
            .build(csr)
            .unwrap_or_else(|e| panic!("seed={seed}: AC factorization failed: {e}"));

        let mut work = vec![0.0f32; factor.n()];
        factor
            .solve_into(&b, &mut work)
            .unwrap_or_else(|e| panic!("seed={seed}: solve_into failed: {e}"));

        assert!(
            work.iter().all(|x| x.is_finite()),
            "seed={seed}: non-finite solve output: {work:?}"
        );
    }
}
