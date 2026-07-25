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
fn disconnected_laplacian_solves_per_component() {
    let (rp, ci, vals) = block_diagonal_paths(2);

    for split_merge in [None, Some(2)] {
        let csr = CsrRef::new(&rp, &ci, &vals, 4).or_panic("valid CSR");
        let factor = Builder::<f64>::new(Config {
            split_merge,
            ..Config::default()
        })
        .build(csr)
        .expect("disconnected Laplacian must factor block-diagonally");
        let solution = factor.solve(&[1.0, -1.0, 2.0, -2.0]).unwrap();
        assert_eq!(solution, vec![1.0, 0.0, 2.0, 0.0]);
    }
}

#[test]
fn disconnected_sparse_path_projects_each_component() {
    let (rp, ci, vals) = block_diagonal_paths(2);
    for split_merge in [None, Some(2)] {
        let factor = Builder::<f64>::new(Config {
            split_merge,
            dense_threshold: 0,
            ..Config::default()
        })
        .build(CsrRef::new(&rp, &ci, &vals, 4).or_panic("valid CSR"))
        .or_panic("disconnected sparse factor");
        let solution = factor.solve(&[1.0, -1.0, 2.0, -2.0]).unwrap();
        assert_eq!(solution, vec![0.5, -0.5, 1.0, -1.0]);
    }
}

#[test]
fn disconnected_laplacian_handles_three_components() {
    let (rp, ci, vals) = block_diagonal_paths(3);

    let csr = CsrRef::new(&rp, &ci, &vals, 6).or_panic("valid CSR");
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .expect("three components must factor");
    assert_eq!(factor.n_steps(), 3);
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
fn tiny_scale_sddm_is_augmented_and_solves() {
    // Surplus below the old absolute floor (1e-10) but above near_zero (1e-14) is
    // genuine dominance: augment and solve, don't misreport a disconnected Laplacian.
    let rp = [0u32, 1, 2];
    let ci = [0u32, 1];
    let vals = [5e-11_f64, 5e-11];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR"))
        .expect("tiny-scale PD SDDM must be augmented and accepted");
    // diag(5e-11) x = b  =>  x = b / 5e-11 (exact to rounding)
    let x = factor.solve(&[1.0, 2.0]).or_panic("solve");
    assert!((x[0] - 1.0 / 5e-11).abs() <= 1e-6 / 5e-11, "x[0]={}", x[0]);
    assert!((x[1] - 2.0 / 5e-11).abs() <= 1e-6 / 5e-11, "x[1]={}", x[1]);
}

#[test]
fn dense_path_solves_sub_sparse_threshold_scale() {
    let rp = [0u32, 1, 2];
    let ci = [0u32, 1];
    let vals = [1e-15_f64, 1e-15];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR"))
        .expect("exact dense path does not clamp small positive pivots");
    let solution = factor.solve(&[1.0, 2.0]).or_panic("solve");
    assert!((solution[0] - 1e15).abs() / 1e15 < 1e-14);
    assert!((solution[1] - 2e15).abs() / 2e15 < 1e-14);
}

#[test]
fn connected_non_dominant_input_is_rejected_as_non_sddm() {
    let rp = [0u32, 2, 4];
    let ci = [0u32, 1, 0, 1];
    let vals = [1.0f64, -3.0, -3.0, 1.0];
    let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
    let error = Builder::<f64>::new(Config::default())
        .build(csr)
        .expect_err("non-SDDM input must be rejected");
    assert!(matches!(error, Error::NotDiagonallyDominant { .. }));
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

#[test]
fn empty_and_singleton_systems_have_defined_solves() {
    let empty = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&[0u32], &[], &[], 0).or_panic("valid empty CSR"))
        .or_panic("empty factor");
    assert_eq!(empty.solve(&[]).or_panic("empty solve"), Vec::<f64>::new());
    assert_eq!(empty.n_steps(), 0);

    let zero = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&[0u32, 1], &[0], &[0.0], 1).or_panic("valid singleton"))
        .or_panic("zero singleton factor");
    assert_eq!(
        zero.solve(&[7.0]).or_panic("zero singleton solve"),
        vec![0.0]
    );

    let positive = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&[0u32, 1], &[0], &[2.0], 1).or_panic("valid singleton"))
        .or_panic("positive singleton factor");
    let solution = positive.solve(&[7.0]).or_panic("positive singleton solve");
    assert!((solution[0] - 3.5).abs() < 1e-14);
}

#[test]
fn many_zero_singletons_factor_as_trivial_components() {
    let n = 128u32;
    let row_ptrs = vec![0u32; n as usize + 1];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &[], &[], n).or_panic("valid zero CSR"))
        .or_panic("zero components");
    assert_eq!(factor.n_steps(), 0);
    assert_eq!(
        factor.solve(&vec![1.0; n as usize]).unwrap(),
        vec![0.0; n as usize]
    );
}

#[test]
fn asymmetric_input_is_rejected_after_duplicate_coalescing() {
    let missing = CsrRef::new(&[0u32, 2, 3], &[0, 1, 1], &[1.0, -1.0, 1.0], 2)
        .or_panic("structurally valid CSR");
    assert!(matches!(
        Builder::<f64>::new(Config::default()).build(missing),
        Err(Error::Asymmetric { edge: (0, 1) })
    ));

    let unequal = CsrRef::new(&[0u32, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -2.0, 2.0], 2)
        .or_panic("structurally valid CSR");
    assert!(matches!(
        Builder::<f64>::new(Config::default()).build(unequal),
        Err(Error::Asymmetric { edge: (0, 1) })
    ));

    let coalesced = CsrRef::new(
        &[0u32, 3, 6],
        &[0, 1, 1, 0, 0, 1],
        &[2.0, -0.25, -0.75, -0.5, -0.5, 2.0],
        2,
    )
    .or_panic("structurally valid CSR");
    Builder::<f64>::new(Config::default())
        .build(coalesced)
        .or_panic("equal coalesced transpose entries");
}

#[test]
fn mixed_grounded_and_floating_components_solve_independently() {
    let row_ptrs = [0u32, 1, 3, 5];
    let columns = [0u32, 1, 2, 1, 2];
    let values = [2.0, 1.0, -1.0, -1.0, 1.0];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &columns, &values, 3).or_panic("valid mixed CSR"))
        .or_panic("mixed factor");
    let solution = factor.solve(&[4.0, 1.0, -1.0]).unwrap();
    assert!((solution[0] - 2.0).abs() < 1e-14);
    assert_eq!(&solution[1..], &[1.0, 0.0]);
}

#[test]
fn dense_threshold_is_applied_per_component() {
    let row_ptrs = [0u32, 2, 4, 6, 9, 11];
    let columns = [0u32, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4];
    let values = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let factor = Builder::<f64>::new(Config {
        dense_threshold: 2,
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 5).or_panic("valid CSR"))
    .or_panic("mixed backend factor");
    let solution = factor.solve(&[1.0, -1.0, 1.0, 0.0, -1.0]).unwrap();
    assert_eq!(solution, vec![1.0, 0.0, 1.0, 0.0, -1.0]);
}

#[test]
fn non_finite_input_is_reported() {
    let non_finite =
        CsrRef::new(&[0u32, 1], &[0], &[f64::NAN], 1).or_panic("structurally valid CSR");
    assert_eq!(
        Builder::<f64>::new(Config::default())
            .build(non_finite)
            .unwrap_err(),
        Error::NonFiniteValue { position: 0 }
    );
}
