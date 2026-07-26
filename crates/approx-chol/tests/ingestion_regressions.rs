#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error};

#[test]
fn positive_off_diagonal_is_rejected_not_silently_dropped() {
    // A = [ 5  1 ]   the +1 off-diagonals are outside the SDDM/Laplacian class.
    //     [ 1  4 ]   Ingestion used to fall through both the diagonal and the
    // `val < 0` edge branch, silently dropping them and factorizing diag(5, 4)
    // — a confidently wrong factor. It now rejects the input instead.
    let rp = [0u32, 2, 4];
    let ci = [0u32, 1, 0, 1];
    let vals = [5.0f64, 1.0, 1.0, 4.0];

    let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
    let err = Builder::<f64>::new(Config::default())
        .build(csr)
        .expect_err("positive off-diagonal must be rejected");
    assert!(
        matches!(err, Error::PositiveOffDiagonal { edge } if edge == (0, 1)),
        "expected PositiveOffDiagonal at (0, 1), got {err:?}"
    );
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
fn disconnected_component_count_is_reported() {
    // A null space bigger than a connected Laplacian's single constant must be
    // rejected, not mis-solved, and the reported count must track k (guards
    // against a hardcoded 2 or an off-by-one) rather than merely "> 1".
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

    for split_merge in [0, 2] {
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
        // Row 0 splits both its diagonal and its edge in two, and row 1's entries
        // are out of order. Splitting the diagonal matters because a coalescing
        // bug there is invisible to the off-diagonal sign and symmetry checks.
        let row_ptrs = [0u32, 4, 7, 9];
        let columns = [1u32, 0, 1, 0, 2, 0, 1, 2, 1];
        let values = [-0.5, 0.25, -0.5, 0.75, -1.0, -1.0, 2.0, 1.0, -1.0];
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
            let result = Builder::<f64>::new(Config::default()).build(csr);
            assert!(
                matches!(result, Err(Error::NonFiniteRow { .. })),
                "{label} must be rejected, got {result:?}"
            );
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
