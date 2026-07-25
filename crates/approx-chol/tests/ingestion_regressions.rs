#[path = "common/backends.rs"]
mod backends;
#[path = "common/grid.rs"]
mod grid;
use backends::backends;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Backend, Config, CsrRef, DenseFailure, Error, ExactFailure};

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

#[test]
fn duplicate_off_diagonals_are_classified_after_coalescing() {
    let row_ptrs = [0u32, 3, 6];
    let columns = [0u32, 1, 1, 0, 0, 1];
    let values = [1.0, 1.0, -2.0, 1.0, -2.0, 1.0];
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 2).or_panic("valid CSR");

    Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("coalesced off-diagonal is negative");
}

#[test]
fn transpose_entries_within_roundoff_are_symmetric() {
    let next_after_one = f64::from_bits(1.0f64.to_bits() + 1);
    let row_ptrs = [0u32, 2, 4];
    let columns = [0u32, 1, 0, 1];
    let values = [2.0, -1.0, -next_after_one, 2.0];
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 2).or_panic("valid CSR");

    Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("one-ulp transpose difference");
}

#[test]
fn high_scale_positive_surplus_is_preserved() {
    let row_ptrs = [0u32, 2, 4];
    let columns = [0u32, 1, 0, 1];
    let values = [1e12 + 100.0, -1e12, -1e12, 1e12 + 100.0];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &columns, &values, 2).or_panic("valid CSR"))
        .or_panic("high-scale SDDM");
    let solution = factor.solve(&[1.0, 1.0]).or_panic("solve");

    assert!(
        (solution[0] - 0.01).abs() < 1e-8,
        "unexpected solution: {solution:?}"
    );
    assert!(
        (solution[1] - 0.01).abs() < 1e-8,
        "unexpected solution: {solution:?}"
    );
}

#[test]
fn dense_ac2_uses_total_virtual_edge_weight() {
    let row_ptrs = [0u32, 3, 6, 9];
    let columns = [0u32, 1, 2, 0, 1, 2, 0, 1, 2];
    let values = [3.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, 3.0];
    let factor = Builder::<f64>::new(Config {
        split_merge: Some(2),
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 3).or_panic("valid CSR"))
    .or_panic("dense AC2 factor");
    let solution = factor.solve(&[1.0, 2.0, 3.0]).or_panic("solve");

    for (actual, expected) in solution.iter().zip([1.75, 2.0, 2.25]) {
        assert!((actual - expected).abs() < 1e-12);
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
        assert_eq!(solution, vec![0.5, -0.5, 1.0, -1.0]);
    }
}

#[test]
fn disconnected_sparse_path_projects_each_component() {
    let (rp, ci, vals) = block_diagonal_paths(2);
    for split_merge in [None, Some(2)] {
        let factor = Builder::<f64>::new(Config {
            split_merge,
            backend: Backend::Approximate,
            ..Config::default()
        })
        .build(CsrRef::new(&rp, &ci, &vals, 4).or_panic("valid CSR"))
        .or_panic("disconnected sparse factor");
        let solution = factor.solve(&[1.0, -1.0, 2.0, -2.0]).unwrap();
        assert_eq!(solution, vec![0.5, -0.5, 1.0, -1.0]);
    }
}

#[test]
fn disconnected_sparse_ac2_preserves_virtual_edge_multiplicity() {
    let row_ptrs = [0u32, 2, 5, 7, 9, 12, 14];
    let columns = [0u32, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 4, 5];
    let values = [
        1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0,
    ];
    let factor = Builder::<f64>::new(Config {
        seed: 7,
        split_merge: Some(3),
        backend: Backend::Approximate,
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 6).or_panic("valid CSR"))
    .or_panic("disconnected AC2 factor");

    let solution = factor
        .solve(&[1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
        .or_panic("solve");
    assert_eq!(solution, vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
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
    assert_eq!(&solution[1..], &[0.5, -0.5]);
}

#[test]
fn exact_backend_is_applied_per_component() {
    let row_ptrs = [0u32, 2, 4, 6, 9, 11];
    let columns = [0u32, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4];
    let values = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let factor = Builder::<f64>::new(Config {
        backend: Backend::ExactBelow {
            max_dim: 2,
            on_failure: ExactFailure::FallBackToApproximate,
        },
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 5).or_panic("valid CSR"))
    .or_panic("mixed backend factor");
    let solution = factor.solve(&[1.0, -1.0, 1.0, 0.0, -1.0]).unwrap();
    assert_eq!(solution, vec![0.5, -0.5, 1.0, 0.0, -1.0]);
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

// Components {0,2} and {1,3} concatenate to [0,2,1,3]: a non-identity block
// permutation, unlike the block-diagonal fixtures above.
#[test]
fn interleaved_components_permute_and_restore_input_order() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let columns = [0u32, 2, 1, 3, 0, 2, 1, 3];
    let values = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    for backend in backends() {
        let factor = Builder::<f64>::new(Config {
            backend,
            ..Config::default()
        })
        .build(CsrRef::new(&row_ptrs, &columns, &values, 4).or_panic("valid CSR"))
        .or_panic("interleaved factor");
        let solution = factor.solve(&[1.0, 2.0, -1.0, -2.0]).or_panic("solve");
        for (got, want) in solution.iter().zip([0.5, 1.0, -0.5, -1.0]) {
            assert!(
                (got - want).abs() < 1e-12,
                "backend {backend:?}: got {solution:?}"
            );
        }
    }
}

// A floating block whose right-hand side is not zero-sum has no exact solution;
// projecting onto the range makes both backends return the same least-squares one.
#[test]
fn inconsistent_rhs_is_projected_onto_the_range() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let columns = [0u32, 2, 1, 3, 0, 2, 1, 3];
    let values = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    for backend in backends() {
        let factor = Builder::<f64>::new(Config {
            backend,
            ..Config::default()
        })
        .build(CsrRef::new(&row_ptrs, &columns, &values, 4).or_panic("valid CSR"))
        .or_panic("interleaved factor");
        // Block {0,2} gets [3, -1] (sum 2), block {1,3} gets [5, -2] (sum 3).
        let solution = factor.solve(&[3.0, 5.0, -1.0, -2.0]).or_panic("solve");
        for (got, want) in solution.iter().zip([1.0, 1.75, -1.0, -1.75]) {
            assert!(
                (got - want).abs() < 1e-12,
                "backend {backend:?}: got {solution:?}"
            );
        }
    }
}

// Ingestion clamps row 1's deficit of -1 (tiny beside a row scale of ~2e16) and so
// accepts a matrix that is exactly singular. Exact Cholesky rightly refuses it;
// the default backend must fall back rather than reject the input outright.
#[test]
fn exact_pivot_failure_falls_back_and_is_recorded() {
    let row_ptrs = [0u32, 2, 5, 7];
    let columns = [0u32, 1, 0, 1, 2, 1, 2];
    let values = [1e16, -1e16, -1e16, 1e16 + 1.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 3).or_panic("valid CSR");

    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("clamped input must factor, not error");
    let fallbacks = factor.exact_fallbacks();
    assert_eq!(fallbacks.len(), 1, "expected one recorded fallback");
    assert_eq!(fallbacks[0].vertex, 1);
    assert_eq!(fallbacks[0].failure, DenseFailure::NonPositivePivot);
    assert!(factor.solve(&[1.0, 0.0, -1.0]).is_ok());

    // Nothing to fall back from when no block was selected for exact Cholesky.
    let approximate = Builder::<f64>::new(Config {
        backend: Backend::Approximate,
        ..Config::default()
    })
    .build(csr)
    .or_panic("approximate factor");
    assert!(approximate.exact_fallbacks().is_empty());
}

#[test]
fn exact_failure_error_policy_propagates_instead_of_falling_back() {
    let row_ptrs = [0u32, 2, 5, 7];
    let columns = [0u32, 1, 0, 1, 2, 1, 2];
    let values = [1e16, -1e16, -1e16, 1e16 + 1.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 3).or_panic("valid CSR");

    let error = Builder::<f64>::new(Config {
        backend: Backend::ExactBelow {
            max_dim: 24,
            on_failure: ExactFailure::Error,
        },
        ..Config::default()
    })
    .build(csr)
    .expect_err("ExactFailure::Error must not fall back");
    assert_eq!(
        error,
        Error::DenseFactorizationFailed {
            vertex: 1,
            failure: DenseFailure::NonPositivePivot,
        }
    );
}

// CSR column indices are not required to be sorted, and duplicates coalesce, so
// bucketed ingestion must not depend on the order entries arrive in.
#[test]
fn unsorted_and_split_entries_match_the_canonical_form() {
    let canonical = {
        let row_ptrs = [0u32, 2, 5, 7];
        let columns = [0u32, 1, 0, 1, 2, 1, 2];
        let values = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
        solve_path(&row_ptrs, &columns, &values)
    };
    let shuffled = {
        // Row 0's edge is split in two, row 1's entries are out of order.
        let row_ptrs = [0u32, 3, 6, 8];
        let columns = [1u32, 0, 1, 2, 0, 1, 2, 1];
        let values = [-0.5, 1.0, -0.5, -1.0, -1.0, 2.0, 1.0, -1.0];
        solve_path(&row_ptrs, &columns, &values)
    };
    for (a, b) in canonical.iter().zip(&shuffled) {
        assert!(
            (a - b).abs() < 1e-12,
            "canonical {canonical:?} vs shuffled {shuffled:?}"
        );
    }
}

// Descending columns per row denote the same matrix but force the bucketed
// ingestion path, so this pins the canonical fast path as bit-identical rather
// than merely close.
#[test]
fn canonical_and_reordered_ingestion_agree_bit_for_bit() {
    let grid = grid::grid_laplacian(6, 7);
    let n = grid.n as usize;

    let mut reversed_columns = Vec::with_capacity(grid.col_indices.len());
    let mut reversed_values = Vec::with_capacity(grid.values.len());
    for row in 0..n {
        let start = grid.row_ptrs[row] as usize;
        let end = grid.row_ptrs[row + 1] as usize;
        reversed_columns.extend(grid.col_indices[start..end].iter().rev());
        reversed_values.extend(grid.values[start..end].iter().rev());
    }

    let rhs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let canonical = solve_grid(grid.as_csr().or_panic("valid CSR"), &rhs);
    let reordered = solve_grid(
        CsrRef::new(&grid.row_ptrs, &reversed_columns, &reversed_values, grid.n)
            .or_panic("valid CSR"),
        &rhs,
    );

    assert_eq!(canonical, reordered);
}

fn solve_grid(csr: CsrRef<'_>, rhs: &[f64]) -> Vec<f64> {
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("grid factor");
    factor.solve(rhs).or_panic("solve")
}

fn solve_path(row_ptrs: &[u32], columns: &[u32], values: &[f64]) -> Vec<f64> {
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(row_ptrs, columns, values, 3).or_panic("valid CSR"))
        .or_panic("path factor");
    factor.solve(&[1.0, 0.0, -1.0]).or_panic("solve")
}

#[test]
fn rows_summing_to_non_finite_are_rejected_on_both_paths() {
    // Stored values are finite, symmetric and non-positive; the row sums are not.
    let max = f64::MAX;
    let expect_rejected =
        |label: &str, row_ptrs: &[u32], col_indices: &[u32], values: &[f64], n| {
            let csr = CsrRef::new(row_ptrs, col_indices, values, n).or_panic("valid CSR");
            for backend in [
                Backend::Approximate,
                Backend::ExactBelow {
                    max_dim: 24,
                    on_failure: ExactFailure::Error,
                },
            ] {
                let result = Builder::<f64>::new(Config {
                    backend,
                    ..Config::default()
                })
                .build(csr);
                assert!(
                    matches!(result, Err(Error::NonFiniteRow { .. })),
                    "{label} under {backend:?} must be rejected, got {result:?}"
                );
            }
        };

    // Bucketed path.
    expect_rejected(
        "coalesced duplicates",
        &[0, 3, 6],
        &[0, 1, 1, 0, 0, 1],
        &[max, -max, -max, -max, -max, max],
        2,
    );
    // Canonical path: no duplicates, two near-MAX off-diagonals in one row.
    expect_rejected(
        "canonical row overflow",
        &[0, 3, 5, 7],
        &[0, 1, 2, 0, 1, 0, 2],
        &[0.0, -max, -max, -max, max, -max, max],
        3,
    );
    expect_rejected("duplicate diagonal", &[0, 2], &[0, 0], &[max, max], 1);
}

#[test]
fn edge_splitting_does_not_reach_the_exact_backend() {
    // AC2's `weight / k` then `weight * count` round trip underflows a subnormal
    // to zero, which fed the exact assembly a diagonal matrix instead of a path.
    let row_ptrs = [0u32, 2, 5, 7];
    let col_indices = [0u32, 1, 0, 1, 2, 1, 2];
    let exact = Backend::ExactBelow {
        max_dim: 24,
        on_failure: ExactFailure::Error,
    };

    for weight in [1.0f64, f64::from_bits(1)] {
        let values = [
            weight,
            -weight,
            -weight,
            2.0 * weight,
            -weight,
            -weight,
            weight,
        ];
        let rhs = [weight, 0.0, -weight];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 3).or_panic("valid CSR");

        let solve = |split_merge| {
            Builder::<f64>::new(Config {
                backend: exact,
                split_merge,
                ..Config::default()
            })
            .build(csr)
            .or_panic("exact factorization")
            .solve(&rhs)
            .or_panic("solve")
        };

        assert_eq!(
            solve(None),
            solve(Some(2)),
            "exact factor must not depend on the AC2 knob (weight={weight:e})"
        );
    }
}
