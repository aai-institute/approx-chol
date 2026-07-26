use super::*;
use crate::test_utils::{path_laplacian_4, OrPanic};

fn make_csr<'a>(indptr: &'a [u32], indices: &'a [u32], data: &'a [f64]) -> CsrRef<'a, f64, u32> {
    CsrRef::new(indptr, indices, data, (indptr.len() - 1) as u32).or_panic("valid CSR test fixture")
}

/// Factorize an SDDM fixture with AC2 across seeds and assert the solve stays
/// finite and non-trivial. Asserts the fixture is actually augmented, since
/// every caller is a regression against losing the augmentation mass.
fn assert_ac2_augmented_solve_is_finite(indptr: &[u32], indices: &[u32], data: &[f64], b: &[f64]) {
    let csr = make_csr(indptr, indices, data);
    for seed in 0..8u64 {
        let factor = Builder::<f64>::new(Config {
            split_merge: 2,
            seed,
        })
        .build(csr)
        .unwrap_or_else(|e| panic!("AC2 factorization failed (seed={seed}): {e}"));
        assert!(
            factor.n() > b.len(),
            "fixture must reach the Gremban-augmented path"
        );

        let mut work = vec![0.0f64; factor.n()];
        work[..b.len()].copy_from_slice(b);
        factor.solve_in_place(&mut work);
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
fn test_ac_default_solve_roundtrip() {
    let (indptr, indices, data) = path_laplacian_4();
    let csr = make_csr(&indptr, &indices, &data);

    let builder = Builder::<f64>::new(Config::default());
    let factor = builder.build(csr).or_panic("factorization should succeed");
    assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));

    let b = [1.0, -1.0, 1.0, -1.0];
    let mut work = vec![0.0; factor.n()];
    factor
        .solve_into(&b, &mut work)
        .or_panic("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
    assert!(work.iter().any(|x| x.abs() > 1e-10));
    let mean = work.iter().sum::<f64>() / work.len() as f64;
    assert!(mean.abs() < 1e-10);
}

/// Regression: a one-neighbor star must take its capacity from the pivot
/// diagonal, not from the single neighbor's weight. Taking the weight dropped
/// the augmentation mass and degenerated the Schur update to NaN/Inf.
#[test]
fn ac2_one_neighbor_star_keeps_augmentation_mass() {
    // 3-node path 0-1-2, diagonal far above the edge weights.
    assert_ac2_augmented_solve_is_finite(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[5.0, -1.0, -1.0, 6.0, -1.0, -1.0, 5.0],
        &[4.0, 4.0, 4.0],
    );
    // 2 nodes, so every vertex's star has exactly one neighbor.
    assert_ac2_augmented_solve_is_finite(
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[10.0, -1.0, -1.0, 10.0],
        &[9.0, -9.0],
    );
}

/// Regression: a star whose total weight underflows to near zero must skip fill
/// sampling rather than divide by it.
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

/// Regression: the AC single-sample path must not drift `diag[v]` below
/// `T::epsilon()` on marginally-SDD Laplacians, where `diag[v] = Σ |off-diag(v)|`
/// exactly. Accumulated fill error can push the maintained `diag[v]` below the
/// live off-diagonal sum, tripping `StarElimination::fraction`'s capacity
/// assertion. This 8-vertex f32 fixture reproduces the panic on every seed;
/// the same path runs for f64.
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
    // Balanced RHS (must sum to zero for a pure Laplacian solve).
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
