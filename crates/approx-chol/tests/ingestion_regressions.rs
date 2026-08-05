#[path = "common/grid.rs"]
mod grid;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error, Factor};

/// The error the shape must be rejected with.
type Rejected<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], Error);
type Accepted<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64]);
type Solved<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], [f64; 2], [f64; 2]);

/// `n` follows from `rp`, so no case can disagree with its own row count.
fn build(config: Config, rp: &[u32], ci: &[u32], vals: &[f64]) -> Result<Factor<f64>, Error> {
    let n = (rp.len() - 1) as u32;
    let csr = CsrRef::new(rp, ci, vals, n).expect("structurally valid CSR");
    Builder::<f64>::new(config).build(csr)
}

/// The expected error is compared whole, so a shape cannot pass by being rejected
/// elsewhere and the reported coordinate cannot drift.
#[test]
fn out_of_class_input_is_rejected_at_its_reported_position() {
    let max = f64::MAX;
    let cases: [Rejected<'_>; 10] = [
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
        // Reaches the asymmetry the mirror cursor skips past, not the one the
        // comparison rejects: the lower entry is stored and its upper is absent.
        (
            "missing upper mirror",
            &[0, 1, 3],
            &[0, 0, 1],
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
        // Descending columns in row 0, so the position must be the caller's flat
        // index and must be reported in preference to the non-canonical shape.
        (
            "stored NaN in a later row of non-canonical input",
            &[0, 2, 5, 7],
            &[1, 0, 2, 1, 0, 2, 1],
            &[-1.0, 1.0, -1.0, 2.0, f64::NAN, 1.0, -1.0],
            Error::NonFiniteValue { position: 4 },
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

/// Shapes a stricter check would reject, but which are valid SDDM.
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
        for split_merge in [None, Some(2)] {
            let config = Config {
                split_merge,
                ..Config::default()
            };
            build(config, rp, ci, vals)
                .unwrap_or_else(|err| panic!("{label} at split_merge {split_merge:?}: {err}"));
        }
    }
}

/// Both ends of the row-scale range, where a floor set in absolute terms fails at one
/// end or the other: 1e12-scale dominance, a 5e-11-scale system, and a surplus real
/// against its own row scale but far below any absolute floor.
#[test]
fn genuine_surplus_at_either_scale_is_augmented_and_solves() {
    let cases: [Solved<'_>; 3] = [
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
        // Surplus 6e-15 on a 2e-6-scale row: 1e9 times the error the row's own additions
        // could carry, so it is dominance rather than noise. Eigenvalue 6e-15 on [1, 1].
        (
            "surplus far below any absolute floor",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1e-6 + 6e-15, -1e-6, -1e-6, 1e-6 + 6e-15],
            [1.0, 1.0],
            [1.0 / 6e-15, 1.0 / 6e-15],
        ),
    ];

    for (label, rp, ci, vals, rhs, expected) in cases {
        let solution = build(Config::default(), rp, ci, vals)
            .expect(label)
            .solve(&rhs)
            .expect("solve");
        for (got, want) in solution.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "{label}: {solution:?} vs {expected:?}"
            );
        }
    }
}

/// A two-row system `[[a+s, -a], [-a, a+s]]` — dominant by `s` on both rows, so a
/// grounded factor carries one extra vertex and a floating one does not.
fn surplus_pair<T>(a: T, s: T) -> ([u32; 3], [u32; 4], [T; 4])
where
    T: Copy + core::ops::Add<Output = T> + core::ops::Neg<Output = T>,
{
    ([0, 2, 4], [0, 1, 0, 1], [a + s, -a, -a, a + s])
}

/// The routing, not the solution: an ill-conditioned pair's solve carries too much
/// round-off to pin, while `n() > original_n()` says which branch was taken. The floor
/// here is `epsilon * scale * (degree + 1)` = `2.2e-16 * 2e-6 * 2` = `8.9e-22`.
#[test]
fn surplus_is_judged_against_summation_error_alone() {
    // (label, surplus, grounded)
    let cases: [(&str, f64, bool); 5] = [
        ("far above the floor", 6e-12, true),
        ("above the floor", 6e-15, true),
        // Was discarded by a floor 1e6 coarser than rounding, and answered as a
        // singular Laplacian: relative error 1.0 on a system with an exact solution.
        ("just above the floor", 6e-18, true),
        // Genuinely within the row's own summation error, so indistinguishable from
        // a floating Laplacian and treated as one.
        ("below the floor", 6e-22, false),
        ("not representable at this scale", 1e-30, false),
    ];

    for (label, surplus, grounded) in cases {
        let (rp, ci, vals) = surplus_pair(1e-6, surplus);
        let factor = build(Config::default(), &rp, &ci, &vals).expect(label);
        assert_eq!(
            factor.n() > factor.original_n(),
            grounded,
            "{label}: surplus {surplus:e} routed to n={} original_n={}",
            factor.n(),
            factor.original_n()
        );
        if grounded {
            let solution = factor.solve(&[1.0, 1.0]).expect("solve");
            let want = 1.0 / surplus;
            assert!(
                (solution[0] - want).abs() <= 1e-3 * want,
                "{label}: {solution:?} vs {want:e}"
            );
        }
    }
}

/// The same window in `f32`, where #85 measured it at a 2e-3 row scale: the old floor
/// was `1e-6 * 2e-3 = 2e-9`, summation error is `1.19e-7 * 2e-3 * 2 = 4.8e-10`.
#[test]
fn f32_surplus_is_judged_against_summation_error_alone() {
    for (label, surplus, grounded) in [
        ("above the floor", 1.2e-9f32, true),
        ("below", 2e-10, false),
    ] {
        let (rp, ci, vals) = surplus_pair(1e-3f32, surplus);
        let n = (rp.len() - 1) as u32;
        let csr = CsrRef::new(&rp, &ci, &vals, n).expect("structurally valid CSR");
        let factor = Builder::<f32>::new(Config::default())
            .build(csr)
            .expect(label);
        assert_eq!(
            factor.n() > factor.original_n(),
            grounded,
            "{label}: surplus {surplus:e}"
        );
    }
}

/// The graph symmetrizes an accepted mirror pair to one value, but classification must
/// not: charging the upper value to both rows made the tolerated difference look like the
/// lower row's own surplus, so the same matrix routed differently depending on which
/// triangle held the larger magnitude.
#[test]
fn tolerated_mirror_difference_is_not_one_row_s_surplus() {
    let off = 1.0 + 5.0 * f64::EPSILON;
    let cases = [
        ("upper holds the smaller", [1.0, -1.0, -off, off]),
        ("lower holds the smaller", [off, -off, -1.0, 1.0]),
    ];
    for (label, vals) in cases {
        let factor = build(Config::default(), &[0, 2, 4], &[0, 1, 0, 1], &vals).expect(label);
        assert_eq!(
            factor.n(),
            factor.original_n(),
            "{label}: every stored row sums to zero, so neither may be grounded"
        );
    }
}

/// `rewrite` folds each duplicate group with its own additions, so the error allowance
/// counts stored entries rather than coalesced neighbours. Ten sub-ULP duplicates are
/// absorbed one way and accumulate the other, which invented a surplus on a row that
/// sums exactly to zero.
#[test]
fn coalescing_additions_are_inside_the_error_allowance() {
    let half = f64::EPSILON / 2.0;
    let (mut rp, mut ci, mut vals) = (vec![0u32], Vec::new(), Vec::new());
    for row in 0..2u32 {
        ci.extend(core::iter::repeat_n(row, 10));
        vals.extend(core::iter::repeat_n(half, 10));
        ci.extend([row, 1 - row]);
        vals.extend([1.0, -1.0]);
        ci.extend(core::iter::repeat_n(1 - row, 10));
        vals.extend(core::iter::repeat_n(-half, 10));
        rp.push(ci.len() as u32);
    }
    let factor = build(Config::default(), &rp, &ci, &vals).expect("coalesced duplicates");
    assert_eq!(
        factor.n(),
        factor.original_n(),
        "duplicates coalescing to a balanced Laplacian must not be grounded"
    );
}

/// `|d| + d` overflowed before the excess was subtracted, rejecting a solvable row for
/// being near the top of the range rather than for anything about its balance.
#[test]
fn a_diagonal_near_the_type_maximum_still_solves() {
    let factor = build(Config::default(), &[0, 1], &[0], &[f64::MAX]).expect("max diagonal");
    let solution = factor.solve(&[f64::MAX]).expect("solve");
    assert!((solution[0] - 1.0).abs() < 1e-12, "{solution:?}");
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

/// Each 2-node block solves independently, contributing one elimination step.
#[test]
fn disconnected_laplacian_solves_per_component() {
    for k in [2u32, 3] {
        let (rp, ci, vals) = block_diagonal_paths(k);
        let n = 2 * k;
        let rhs: Vec<f64> = (1..=k).flat_map(|b| [b as f64, -(b as f64)]).collect();
        let expected: Vec<f64> = rhs.iter().map(|value| value / 2.0).collect();

        for split_merge in [None, Some(2)] {
            let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid CSR");
            let factor = Builder::<f64>::new(Config {
                split_merge,
                ..Config::default()
            })
            .build(csr)
            .expect("disconnected Laplacian must factor block-diagonally");
            assert_eq!(factor.n_steps(), k as usize, "one step per 2-node block");
            assert_eq!(factor.solve(&rhs).expect("solve"), expected);
        }
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
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 6).expect("valid CSR"))
    .expect("disconnected AC2 factor");

    let solution = factor
        .solve(&[1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
        .expect("solve");
    assert_eq!(solution, vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
}

#[test]
fn many_zero_singletons_factor_as_trivial_components() {
    let n = 128u32;
    let row_ptrs = vec![0u32; n as usize + 1];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &[], &[], n).expect("valid zero CSR"))
        .expect("zero components");
    assert_eq!(factor.n_steps(), 0);
    assert_eq!(
        factor.solve(&vec![1.0; n as usize]).expect("solve"),
        vec![0.0; n as usize]
    );
}

#[test]
fn mixed_grounded_and_floating_components_solve_independently() {
    let row_ptrs = [0u32, 1, 3, 5];
    let columns = [0u32, 1, 2, 1, 2];
    let values = [2.0, 1.0, -1.0, -1.0, 1.0];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &columns, &values, 3).expect("valid mixed CSR"))
        .expect("mixed factor");
    let solution = factor.solve(&[4.0, 1.0, -1.0]).expect("solve");
    assert!((solution[0] - 2.0).abs() < 1e-14);
    assert_eq!(&solution[1..], &[0.5, -0.5]);
}

// Components {0,2} and {1,3} concatenate to [0,2,1,3]: a non-identity block
// permutation, unlike the block-diagonal fixtures above. A right-hand side that
// is not zero-sum per block has no exact solution, so the second case is the
// least-squares one the range projection gives.
#[test]
fn interleaved_components_solve_in_input_order() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let columns = [0u32, 2, 1, 3, 0, 2, 1, 3];
    let values = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    let factor = Builder::<f64>::new(Config::default())
        .build(CsrRef::new(&row_ptrs, &columns, &values, 4).expect("valid CSR"))
        .expect("interleaved factor");

    // Block {0,2} gets the 1st and 3rd entry, block {1,3} the 2nd and 4th.
    let cases = [
        (
            "zero-sum per block",
            [1.0, 2.0, -1.0, -2.0],
            [0.5, 1.0, -0.5, -1.0],
        ),
        (
            "inconsistent",
            [3.0, 5.0, -1.0, -2.0],
            [1.0, 1.75, -1.0, -1.75],
        ),
    ];
    for (label, rhs, expected) in cases {
        let solution = factor.solve(&rhs).expect("solve");
        for (got, want) in solution.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{label}: {solution:?}");
        }
    }
}

/// Every star has degree two, so AC is exact and the residual is round-off. The
/// fixtures above are too small to swap-remove.
#[test]
fn moved_components_keep_their_edges_through_fill_and_removal() {
    const N: u32 = 16;
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for v in 0..N {
        let mut row = [((v + N - 2) % N, -1.0), (v, 2.0), ((v + 2) % N, -1.0)];
        row.sort_unstable_by_key(|&(column, _)| column);
        columns.extend(row.iter().map(|&(column, _)| column));
        values.extend(row.iter().map(|&(_, value)| value));
        row_ptrs.push(columns.len() as u32);
    }

    // Zero-sum within each cycle, so the singular system is consistent.
    let rhs: Vec<f64> = (0..N).map(|v| if v < N / 2 { 1.0 } else { -1.0 }).collect();
    for seed in 0..4u64 {
        let factor = build(
            Config {
                seed,
                ..Config::default()
            },
            &row_ptrs,
            &columns,
            &values,
        )
        .expect("double-cycle factor");
        assert_eq!(factor.n_steps(), (N - 2) as usize, "one pin per cycle");

        let x = factor.solve(&rhs).expect("solve");
        for row in 0..N as usize {
            let range = row_ptrs[row] as usize..row_ptrs[row + 1] as usize;
            let ax: f64 = columns[range.clone()]
                .iter()
                .zip(&values[range])
                .map(|(&column, value)| value * x[column as usize])
                .sum();
            let error = (ax - rhs[row]).abs();
            assert!(error < 1e-10, "seed={seed}: row {row} residual {error:.3e}");
        }
    }
}

#[test]
fn empty_and_singleton_systems_have_defined_solves() {
    let empty = build(Config::default(), &[0], &[], &[]).expect("empty factor");
    assert_eq!(empty.solve(&[]).expect("empty solve"), Vec::<f64>::new());
    assert_eq!(empty.n_steps(), 0);

    let zero = build(Config::default(), &[0, 1], &[0], &[0.0]).expect("zero singleton");
    assert_eq!(zero.solve(&[7.0]).expect("zero singleton solve"), vec![0.0]);

    let positive = build(Config::default(), &[0, 1], &[0], &[2.0]).expect("positive singleton");
    let solution = positive.solve(&[7.0]).expect("positive singleton solve");
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
            .expect("grid factor")
            .solve(&rhs)
            .expect("solve")
    };

    let reordered = CsrRef::new(&grid.row_ptrs, &reversed_columns, &reversed_values, grid.n)
        .expect("valid CSR");
    assert_eq!(solve(grid.as_csr().expect("valid CSR")), solve(reordered));
}

fn solve_path(rp: &[u32], ci: &[u32], vals: &[f64]) -> Vec<f64> {
    build(Config::default(), rp, ci, vals)
        .expect("path factor")
        .solve(&[1.0, 0.0, -1.0])
        .expect("solve")
}
