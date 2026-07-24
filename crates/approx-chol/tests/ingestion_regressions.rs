#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error};

fn solve_with_default_ac(
    row_ptrs: &[u32],
    col_indices: &[u32],
    values: &[f64],
    n: u32,
    rhs: &[f64],
) -> Vec<f64> {
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).or_panic("valid CSR");
    let factor = Builder::<f64>::new(Config {
        seed: 7,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization");
    let mut work = vec![0.0; factor.n()];
    work[..rhs.len()].copy_from_slice(rhs);
    factor
        .solve_in_place(&mut work)
        .or_panic("solve_in_place should succeed");
    work
}

#[test]
fn duplicate_diagonal_entries_do_not_change_solve_behavior() {
    // Two mathematically equivalent 2x2 SDDM matrices:
    // - `dup_*`: diagonal split into duplicate entries in each row
    // - `coal_*`: diagonal already coalesced
    //
    // A = [ 5  -1 ]
    //     [ -1  4 ]
    let dup_rp = vec![0u32, 3, 6];
    let dup_ci = vec![0u32, 0, 1, 0, 1, 1];
    let dup_vals = vec![2.0f64, 3.0, -1.0, -1.0, 1.5, 2.5];

    let coal_rp = vec![0u32, 2, 4];
    let coal_ci = vec![0u32, 1, 0, 1];
    let coal_vals = vec![5.0f64, -1.0, -1.0, 4.0];

    let rhs = vec![1.0f64, -1.0];

    let x_dup = solve_with_default_ac(&dup_rp, &dup_ci, &dup_vals, 2, &rhs);
    let x_coal = solve_with_default_ac(&coal_rp, &coal_ci, &coal_vals, 2, &rhs);

    assert_eq!(
        x_dup.len(),
        x_coal.len(),
        "equivalent inputs must produce factors with equal dimension"
    );
    for (i, (&a, &b)) in x_dup.iter().zip(x_coal.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "equivalent matrices should solve identically; mismatch at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn positive_off_diagonal_is_rejected_not_silently_dropped() {
    // A = [ 5  1 ]   the +1 off-diagonals are outside the SDDM/Laplacian class.
    //     [ 1  4 ]   Ingestion used to fall through both the diagonal and the
    // `val < 0` edge branch, silently dropping them and factorizing diag(5, 4)
    // — a confidently wrong factor. It now rejects the input instead.
    let rp = [0u32, 2, 4];
    let ci = [0u32, 1, 0, 1];
    let vals = [5.0f64, 1.0, 1.0, 4.0];

    // The AC (None) and AC2 (Some) paths share this ingestion; both must reject.
    for split_merge in [None, Some(2)] {
        let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
        let err = Builder::<f64>::new(Config {
            split_merge,
            ..Config::default()
        })
        .build(csr)
        .expect_err("positive off-diagonal must be rejected");
        assert!(
            matches!(err, Error::PositiveOffDiagonal { edge } if edge == (0, 1)),
            "expected PositiveOffDiagonal at (0, 1), got {err:?}"
        );
    }
}

/// CSR for `k` disjoint 2-node path Laplacians, stacked block-diagonally — a
/// graph with `k` connected components. For `k = 2`:
///   [ 1 -1  .  . ]
///   [-1  1  .  . ]
///   [ .  .  1 -1 ]
///   [ .  . -1  1 ]
fn block_diagonal_paths(k: u32) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut vals) = (vec![0u32], Vec::new(), Vec::new());
    for b in 0..k {
        let (a, z) = (2 * b, 2 * b + 1);
        // both rows span columns [a, z]: +1 on the diagonal, -1 off it
        for row_vals in [[1.0, -1.0], [-1.0, 1.0]] {
            ci.extend([a, z]);
            vals.extend(row_vals);
            rp.push(ci.len() as u32);
        }
    }
    (rp, ci, vals)
}

#[test]
fn disconnected_laplacian_is_rejected() {
    // Two components => a null space bigger than a connected Laplacian's single
    // constant, so it must be rejected, not mis-solved. AC and AC2 both.
    let (rp, ci, vals) = block_diagonal_paths(2);

    for split_merge in [None, Some(2)] {
        let csr = CsrRef::new(&rp, &ci, &vals, 4).or_panic("valid CSR");
        let err = Builder::<f64>::new(Config {
            split_merge,
            ..Config::default()
        })
        .build(csr)
        .expect_err("disconnected input must be rejected");
        assert!(
            matches!(err, Error::Disconnected { components: 2 }),
            "expected Disconnected {{ components: 2 }}, got {err:?}"
        );
    }
}

#[test]
fn disconnected_component_count_is_reported() {
    // Three components: the reported count must track k (guards against a
    // hardcoded 2 or an off-by-one), not merely "> 1".
    let (rp, ci, vals) = block_diagonal_paths(3);

    let csr = CsrRef::new(&rp, &ci, &vals, 6).or_panic("valid CSR");
    let err = Builder::<f64>::new(Config::default())
        .build(csr)
        .expect_err("disconnected input must be rejected");
    assert!(
        matches!(err, Error::Disconnected { components: 3 }),
        "expected Disconnected {{ components: 3 }}, got {err:?}"
    );
}

#[test]
fn connected_single_component_input_is_not_falsely_rejected() {
    // The guard must fire only on genuine disconnection: a connected pure
    // Laplacian and a connected SDDM are single-component and must factorize.
    let lap = ([0u32, 2, 4], [0u32, 1, 0, 1], [1.0f64, -1.0, -1.0, 1.0]);
    let sddm = ([0u32, 2, 4], [0u32, 1, 0, 1], [5.0f64, -1.0, -1.0, 4.0]);

    for (rp, ci, vals) in [lap, sddm] {
        for split_merge in [None, Some(2)] {
            let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
            Builder::<f64>::new(Config {
                split_merge,
                ..Config::default()
            })
            .build(csr)
            .expect("connected single-component input must factorize");
        }
    }
}

#[test]
fn connected_non_dominant_input_is_not_reported_disconnected() {
    // Non-dominant input (negative row sums) leaves the ground vertex isolated;
    // that artifact must not be miscounted as a component. The graph is connected,
    // so it must not be reported as Disconnected (even though it's out of class).
    let rp = [0u32, 2, 4];
    let ci = [0u32, 1, 0, 1];
    let vals = [1.0f64, -3.0, -3.0, 1.0];
    let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
    let result = Builder::<f64>::new(Config::default()).build(csr);
    assert!(
        !matches!(result, Err(Error::Disconnected { .. })),
        "connected input must not be reported as Disconnected, got {result:?}"
    );
}

#[test]
fn block_diagonal_sddm_is_accepted_via_shared_ground() {
    // Two PD SDDM blocks, block-diagonal: the off-diagonal graph has two
    // components, but every row has surplus so the ground vertex links them into
    // one. Valid SDDM, must be accepted (a pre-augmentation count would reject it).
    let rp = [0u32, 2, 4, 6, 8];
    let ci = [0u32, 1, 0, 1, 2, 3, 2, 3];
    let vals = [5.0f64, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0];

    for split_merge in [None, Some(2)] {
        let csr = CsrRef::new(&rp, &ci, &vals, 4).or_panic("valid CSR");
        Builder::<f64>::new(Config {
            split_merge,
            ..Config::default()
        })
        .build(csr)
        .expect("block-diagonal SDDM must be accepted via the shared ground vertex");
    }
}
