#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error, Factor};

/// `(label, row_ptrs, col_indices, values)`, and the error the shape must be
/// rejected with.
type Rejected<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], Error);
/// A shape that must be accepted.
type Accepted<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64]);
/// A shape with a right-hand side and the solution it must produce.
type Solved<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], [f64; 2], [f64; 2]);

/// `n` follows from `rp`, so no case can disagree with its own row count.
fn build(config: Config, rp: &[u32], ci: &[u32], vals: &[f64]) -> Result<Factor<f64>, Error> {
    let n = (rp.len() - 1) as u32;
    let csr = CsrRef::new(rp, ci, vals, n).or_panic("structurally valid CSR");
    Builder::<f64>::new(config).build(csr)
}

/// Each shape is the smallest matrix reaching one rejection, and the expected
/// error is compared whole: a shape cannot pass by being rejected elsewhere, and
/// the reported coordinate or row cannot drift.
#[test]
fn out_of_class_input_is_rejected_at_its_reported_position() {
    let max = f64::MAX;
    let cases: [Rejected<'_>; 9] = [
        // Used to fall through both the diagonal and the `val < 0` edge branch,
        // silently factorizing diag(5, 4) — a confidently wrong factor.
        (
            "positive off-diagonal",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[5.0, 1.0, 1.0, 4.0],
            Error::PositiveOffDiagonal { edge: (0, 1) },
        ),
        (
            "missing transpose",
            &[0, 2, 3],
            &[0, 1, 1],
            &[1.0, -1.0, 1.0],
            Error::Asymmetric { edge: (0, 1) },
        ),
        (
            "unequal transpose",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -1.0, -2.0, 2.0],
            Error::Asymmetric { edge: (0, 1) },
        ),
        (
            "connected but not dominant",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -3.0, -3.0, 1.0],
            Error::NotDiagonallyDominant { row: 0 },
        ),
        (
            "stored NaN",
            &[0, 1],
            &[0],
            &[f64::NAN],
            Error::NonFiniteValue { position: 0 },
        ),
        // A null space bigger than a connected Laplacian's single constant. Three
        // disjoint 2-node paths, so the count guards against a hardcoded 2.
        (
            "three components",
            &[0, 2, 4, 6, 8, 10, 12],
            &[0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5],
            &[
                1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
            ],
            Error::Disconnected { components: 3 },
        ),
        // The three row-sum overflows below store only finite, symmetric,
        // non-positive values; one per ingestion path.
        (
            "overflow via coalesced duplicates",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[max, -max, -max, -max, -max, max],
            Error::NonFiniteRow { row: 0 },
        ),
        (
            "overflow on the canonical path",
            &[0, 3, 5, 7],
            &[0, 1, 2, 0, 1, 0, 2],
            &[0.0, -max, -max, -max, max, -max, max],
            Error::NonFiniteRow { row: 0 },
        ),
        (
            "overflow via duplicate diagonal",
            &[0, 2],
            &[0, 0],
            &[max, max],
            Error::NonFiniteRow { row: 0 },
        ),
    ];

    for (label, rp, ci, vals, expected) in cases {
        let observed = build(Config::default(), rp, ci, vals).expect_err(label);
        assert_eq!(observed, expected, "{label}");
    }
}

/// Shapes that look out of class to a stricter check but are valid SDDM, on both
/// the AC and AC2 paths.
#[test]
fn in_class_input_is_accepted_on_both_paths() {
    let next_after_one = f64::from_bits(1.0f64.to_bits() + 1);
    let cases: [Accepted<'_>; 3] = [
        (
            "one-ulp transpose difference",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0, -1.0, -next_after_one, 2.0],
        ),
        (
            "transposes equal only after coalescing",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[2.0, -0.25, -0.75, -0.5, -0.5, 2.0],
        ),
        // Two PD SDDM blocks: the off-diagonal graph has two components, but every
        // row has surplus, so the ground vertex links them into one. A
        // pre-augmentation component count would reject it.
        (
            "block-diagonal SDDM sharing a ground vertex",
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
        ),
    ];

    for (label, rp, ci, vals) in cases {
        for split_merge in [0, 2] {
            let config = Config {
                split_merge,
                ..Config::default()
            };
            build(config, rp, ci, vals)
                .unwrap_or_else(|err| panic!("{label} at split_merge {split_merge}: {err}"));
        }
    }
}

/// Both ends of the per-row surplus floor: 1e12-scale dominance must survive the
/// relative tolerance, and a 5e-11 surplus (below the old absolute 1e-10 floor,
/// above `near_zero`) must still count as dominance rather than a disconnected
/// Laplacian. The accuracy bound is relative for the same reason the floor is.
#[test]
fn genuine_surplus_at_either_scale_is_augmented_and_solves() {
    let cases: [Solved<'_>; 2] = [
        (
            "1e12 scale",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1e12 + 100.0, -1e12, -1e12, 1e12 + 100.0],
            [1.0, 1.0],
            [0.01, 0.01],
        ),
        // diag(5e-11) x = b  =>  x = b / 5e-11 (exact to rounding)
        (
            "5e-11 scale",
            &[0, 1, 2],
            &[0, 1],
            &[5e-11, 5e-11],
            [1.0, 2.0],
            [1.0 / 5e-11, 2.0 / 5e-11],
        ),
    ];

    for (label, rp, ci, vals, rhs, expected) in cases {
        let solution = build(Config::default(), rp, ci, vals)
            .or_panic(label)
            .solve(&rhs)
            .or_panic("solve");
        for (got, want) in solution.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "{label}: {solution:?} vs {expected:?}"
            );
        }
    }
}

#[test]
fn empty_and_singleton_systems_have_defined_solves() {
    let empty = build(Config::default(), &[0], &[], &[]).or_panic("empty factor");
    assert_eq!(empty.solve(&[]).or_panic("empty solve"), Vec::<f64>::new());
    assert_eq!(empty.n_steps(), 0);

    let zero = build(Config::default(), &[0, 1], &[0], &[0.0]).or_panic("zero singleton");
    assert_eq!(
        zero.solve(&[7.0]).or_panic("zero singleton solve"),
        vec![0.0]
    );

    let positive = build(Config::default(), &[0, 1], &[0], &[2.0]).or_panic("positive singleton");
    let solution = positive.solve(&[7.0]).or_panic("positive singleton solve");
    assert!((solution[0] - 3.5).abs() < 1e-14);
}

// CSR column indices are not required to be sorted, and duplicates coalesce, so
// bucketed ingestion must not depend on the order entries arrive in.
#[test]
fn unsorted_and_split_entries_match_the_canonical_form() {
    let canonical = solve_path(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    );
    // Row 0 splits both its diagonal and its edge in two, and row 1's entries are
    // out of order. Splitting the diagonal matters because a coalescing bug there
    // is invisible to the off-diagonal sign and symmetry checks.
    let shuffled = solve_path(
        &[0, 4, 7, 9],
        &[1, 0, 1, 0, 2, 0, 1, 2, 1],
        &[-0.5, 0.25, -0.5, 0.75, -1.0, -1.0, 2.0, 1.0, -1.0],
    );
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
        let span = grid.row_ptrs[row] as usize..grid.row_ptrs[row + 1] as usize;
        reversed_columns.extend(grid.col_indices[span.clone()].iter().rev());
        reversed_values.extend(grid.values[span].iter().rev());
    }

    let rhs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let solve = |csr| {
        Builder::<f64>::new(Config::default())
            .build(csr)
            .or_panic("grid factor")
            .solve(&rhs)
            .or_panic("solve")
    };

    let reordered = CsrRef::new(&grid.row_ptrs, &reversed_columns, &reversed_values, grid.n)
        .or_panic("valid CSR");
    assert_eq!(solve(grid.as_csr().or_panic("valid CSR")), solve(reordered));
}

fn solve_path(rp: &[u32], ci: &[u32], vals: &[f64]) -> Vec<f64> {
    build(Config::default(), rp, ci, vals)
        .or_panic("path factor")
        .solve(&[1.0, 0.0, -1.0])
        .or_panic("solve")
}
