#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error, Factor};

/// The error the shape must be rejected with.
type Rejected<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], Error);
type Accepted<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64]);
type Solved<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], [f64; 2], [f64; 2]);

/// `n` follows from `rp`, so no case can disagree with its own row count.
fn build(config: Config, rp: &[u32], ci: &[u32], vals: &[f64]) -> Result<Factor<f64>, Error> {
    let n = (rp.len() - 1) as u32;
    let csr = CsrRef::new(rp, ci, vals, n).or_panic("structurally valid CSR");
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

/// Every end of the per-row surplus floor: 1e12-scale dominance must survive the
/// relative tolerance, a 5e-11 surplus must still count as dominance, and a surplus
/// real against its own row scale must not be discarded for being absolutely small.
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
            "surplus below near_zero",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1e-6 + 6e-15, -1e-6, -1e-6, 1e-6 + 6e-15],
            [1.0, 1.0],
            [1.0 / 6e-15, 1.0 / 6e-15],
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
            let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid CSR");
            let factor = Builder::<f64>::new(Config {
                split_merge,
                ..Config::default()
            })
            .build(csr)
            .expect("disconnected Laplacian must factor block-diagonally");
            assert_eq!(factor.n_steps(), k as usize, "one step per 2-node block");
            assert_eq!(factor.solve(&rhs).unwrap(), expected);
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
    .build(CsrRef::new(&row_ptrs, &columns, &values, 6).or_panic("valid CSR"))
    .or_panic("disconnected AC2 factor");

    let solution = factor
        .solve(&[1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
        .or_panic("solve");
    assert_eq!(solution, vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
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
        .build(CsrRef::new(&row_ptrs, &columns, &values, 4).or_panic("valid CSR"))
        .or_panic("interleaved factor");

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
        let solution = factor.solve(&rhs).or_panic("solve");
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
        .or_panic("double-cycle factor");
        assert_eq!(factor.n_steps(), (N - 2) as usize, "one pin per cycle");

        let x = factor.solve(&rhs).or_panic("solve");
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
