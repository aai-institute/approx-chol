use super::*;
use crate::test_utils::OrPanic;

/// Build a 4-node path graph Laplacian as raw CSR arrays.
fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let indptr = vec![0u32, 2, 5, 8, 10];
    let indices = vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let data = vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    (indptr, indices, data)
}

fn make_csr<'a>(indptr: &'a [u32], indices: &'a [u32], data: &'a [f64]) -> CsrRef<'a, f64, u32> {
    CsrRef::new(indptr, indices, data, (indptr.len() - 1) as u32).or_panic("valid CSR test fixture")
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

/// Regression test for the AC2 n==1 diagonal bug.
///
/// Constructs a 3-node augmented SDDM matrix where every vertex has
/// exactly one off-diagonal neighbor but the diagonal is larger than the
/// neighbor weight (due to augmentation). The old code incorrectly set
/// `column.diagonal = entries[0].1` (the neighbor weight) instead of
/// `pivot_diag` (the actual matrix diagonal), losing the augmentation
/// mass and producing an incorrect factorization.
///
/// Matrix (3 nodes, path graph 0–1–2, augmented diagonal):
///   A = [ 5.0  -1.0   0.0 ]
///       [-1.0   6.0  -1.0 ]
///       [ 0.0  -1.0   5.0 ]
///
/// Every node has at most 2 neighbors, but nodes 0 and 2 have only 1
/// neighbor each, and their diagonal (5.0) is much larger than their
/// edge weight (1.0), making this a clear augmented case.
#[test]
fn test_ac2_n_eq_1_augmented_diagonal_regression() {
    // 3-node path graph 0-1-2 with diagonal augmentation.
    // Node 0: diag=5.0, edge to 1 with weight 1.0
    // Node 1: diag=6.0, edges to 0 and 2 with weight 1.0 each
    // Node 2: diag=5.0, edge to 1 with weight 1.0
    let indptr = vec![0u32, 2, 5, 7];
    let indices = vec![0u32, 1, 0, 1, 2, 1, 2];
    let data = vec![5.0f64, -1.0, -1.0, 6.0, -1.0, -1.0, 5.0];

    let csr = CsrRef::new(&indptr, &indices, &data, 3).or_panic("valid SDDM matrix");

    let config = Config {
        split_merge: Some(2),
        ..Default::default()
    };
    let builder = Builder::<f64>::new(config);

    // Should complete without panic (old code would produce NaN/Inf for
    // vertices with n==1 and pivot_diag > neighbor_weight).
    let factor = builder
        .build(csr)
        .or_panic("AC2 factorization must succeed");
    // The factor may be larger than 3 due to Gremban augmentation (the
    // matrix is SDDM but not Laplacian, so an auxiliary vertex is added).
    assert!(
        factor.n() >= 3,
        "factor dimension should be at least the original matrix size"
    );

    // Solve using the factorization as a preconditioner application.
    // The RHS covers the original 3 nodes; the factorization may operate
    // on an augmented system (4 nodes due to Gremban's reduction).
    let b = [4.0f64, 4.0, 4.0];
    let mut work = vec![0.0f64; factor.n()];
    factor
        .solve_into_with_projection(&b, &mut work, false)
        .or_panic("solve_into_with_projection should succeed");

    // All entries must be finite — the old bug set column.diagonal to the
    // small edge weight (1.0) instead of pivot_diag (5.0), making the
    // Schur complement update degenerate and producing NaN/Inf.
    assert!(
        work.iter().all(|x| x.is_finite()),
        "AC2 solve produced non-finite output with n==1 augmented diagonal: {:?}",
        work
    );

    // The output must be non-trivially non-zero (the system has a unique
    // solution since A is strictly diagonally dominant).
    assert!(
        work.iter().any(|x| x.abs() > 1e-10),
        "AC2 solve produced trivially zero output: {:?}",
        work
    );
}

/// Additional regression: verify AC2 handles n==1 with split=2 edge
/// replication, ensuring the augmentation mass is never lost across
/// multiple seeds.
#[test]
fn test_ac2_n_eq_1_solve_produces_finite_for_multiple_seeds() {
    // Minimal 2-node SDDM: each node has exactly 1 neighbor,
    // diagonal (10.0) >> edge weight (1.0) — strong augmentation.
    //   A = [10.0  -1.0]
    //       [-1.0  10.0]
    let indptr = vec![0u32, 2, 4];
    let indices = vec![0u32, 1, 0, 1];
    let data = vec![10.0f64, -1.0, -1.0, 10.0];
    let b = [9.0f64, -9.0];

    for seed in 0..8u64 {
        let csr = CsrRef::new(&indptr, &indices, &data, 2).or_panic("valid SDDM");
        let config = Config {
            split_merge: Some(2),
            seed,
        };
        let factor = Builder::<f64>::new(config)
            .build(csr)
            .unwrap_or_else(|e| panic!("AC2 factorization failed (seed={seed}): {e}"));

        let mut work = vec![0.0f64; factor.n()];
        factor
            .solve_into_with_projection(&b, &mut work, false)
            .or_panic("solve_into_with_projection should succeed");

        assert!(
            work.iter().all(|x| x.is_finite()),
            "AC2 produced non-finite output for seed={seed}: {:?}",
            work
        );
    }
}

/// Regression test: AC2 handles near-zero total weight without division-by-zero.
///
/// Constructs an SDDM matrix with extremely small edge weights (1e-300) but
/// normal diagonal. The AC2 path encounters near-zero `total_weight` in the
/// star neighborhood and must skip fill sampling gracefully.
#[test]
fn test_ac2_near_zero_weight_star() {
    // 3-node path graph with tiny edge weights and normal diagonal.
    //   A = [ 2.0    -1e-300   0.0     ]
    //       [-1e-300  2.0     -1e-300   ]
    //       [ 0.0    -1e-300   2.0      ]
    let eps = 1e-300_f64;
    let indptr = vec![0u32, 2, 5, 7];
    let indices = vec![0u32, 1, 0, 1, 2, 1, 2];
    let data = vec![2.0, -eps, -eps, 2.0, -eps, -eps, 2.0];

    let csr = CsrRef::new(&indptr, &indices, &data, 3).or_panic("valid SDDM matrix");

    let config = Config {
        split_merge: Some(2),
        ..Default::default()
    };
    let factor = Builder::<f64>::new(config)
        .build(csr)
        .or_panic("AC2 factorization must succeed with near-zero weights");

    assert!(factor.n() >= 3);

    let b = [1.0f64, -1.0, 1.0];
    let mut work = vec![0.0f64; factor.n()];
    factor
        .solve_into_with_projection(&b, &mut work, false)
        .or_panic("solve_into_with_projection should succeed");

    assert!(
        work.iter().all(|x| x.is_finite()),
        "AC2 solve produced non-finite output with near-zero edge weights: {:?}",
        work
    );
}

/// Regression: the AC single-sample path must not drift `diag[v]` below
/// `T::epsilon()` on marginally-SDD Laplacian inputs.
///
/// Before the fix, `clique_tree_sample_column` initialized
/// `StarElimination::capacity` from the externally-maintained `diag[v]`.
/// For pure Laplacians, `diag[v] = Σ |off-diag(v)|` exactly (zero SDD
/// slack). Floating-point error from accumulated fill additions can then
/// push the maintained `diag[v]` below the live off-diagonal sum, and a
/// later pivot pop trips `StarElimination::fraction`'s
/// `debug_assert!(capacity > T::epsilon())`.
///
/// This 8-vertex dense Laplacian (with the AC default config) was found
/// by random search to reproduce the panic deterministically in f32 across
/// every solve seed. f32 was the easiest tier to surface the drift in a
/// small test fixture; the same code path runs for f64 and is also fixed.
#[test]
fn test_ac_marginally_sdd_laplacian_no_capacity_drift() {
    // 8-vertex Laplacian (zero row sums, marginally SDD by construction).
    // Layout: row-sorted CSR with off-diagonal magnitudes spread across
    // ~10x weight range — enough relative-error variation to push the
    // maintained `diag[v]` below the live entry sum in f32.
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
            "seed={seed}: non-finite solve output: {:?}",
            work
        );
    }
}
