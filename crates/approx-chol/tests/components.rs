#[path = "common/factor.rs"]
mod factor;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/residual.rs"]
mod residual;
use factor::factor;
use laplacian_prop::{one_grounded_component_strategy, per_component_consistent_rhs};
use residual::relative_residual_over;

use approx_chol::{factorize_with, Backend, Config, CsrRef, Sddm};
use proptest::prelude::*;
use rstest::rstest;

fn csr<'a>(rp: &'a [u32], ci: &'a [u32], vals: &'a [f64]) -> CsrRef<'a> {
    CsrRef::new(rp, ci, vals, (rp.len() - 1) as u32).expect("valid CSR")
}

#[test]
fn empty_and_singleton_systems_have_defined_solves() {
    let empty = factor(Config::default(), csr(&[0], &[], &[])).expect("empty factor");
    assert_eq!(empty.solve(&[]).expect("empty solve"), Vec::<f64>::new());
    assert_eq!(empty.n_steps(), 0);

    let zero = factor(Config::default(), csr(&[0, 1], &[0], &[0.0])).expect("zero singleton");
    assert_eq!(zero.solve(&[7.0]).expect("zero singleton solve"), vec![0.0]);

    let positive =
        factor(Config::default(), csr(&[0, 1], &[0], &[2.0])).expect("positive singleton");
    let solution = positive.solve(&[7.0]).expect("positive singleton solve");
    assert!((solution[0] - 3.5).abs() < 1e-14);
}

#[test]
fn many_zero_singletons_factor_as_trivial_components() {
    let n = 128usize;
    let row_ptrs = vec![0u32; n + 1];
    let factor = factor(Config::default(), csr(&row_ptrs, &[], &[])).expect("zero components");
    assert_eq!(factor.n_steps(), 0);
    assert_eq!(factor.solve(&vec![1.0; n]).expect("solve"), vec![0.0; n]);
}

fn block_diagonal_paths(k: u32) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut vals) = (vec![0u32], Vec::new(), Vec::new());
    for b in 0..k {
        let (a, z) = (2 * b, 2 * b + 1);
        for row_vals in [[1.0, -1.0], [-1.0, 1.0]] {
            ci.extend([a, z]);
            vals.extend(row_vals);
            rp.push(ci.len() as u32);
        }
    }
    (rp, ci, vals)
}

/// Each 2-vertex component solves independently, contributing one elimination step.
#[test]
fn disconnected_laplacian_solves_per_component() {
    for k in [2u32, 3] {
        let (rp, ci, vals) = block_diagonal_paths(k);
        let rhs: Vec<f64> = (1..=k).flat_map(|b| [b as f64, -(b as f64)]).collect();
        let expected: Vec<f64> = rhs.iter().map(|value| value / 2.0).collect();

        for split_merge in [None, Some(2)] {
            let config = Config {
                split_merge,
                ..Config::default()
            };
            let factor = factor(config, csr(&rp, &ci, &vals)).expect("factor");
            assert_eq!(factor.n_steps(), k as usize, "one step per component");
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
    let config = Config {
        seed: 7,
        split_merge: Some(3),
        ..Config::default()
    };
    let factor = factor(config, csr(&row_ptrs, &columns, &values)).expect("AC2 factor");
    let b = [1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    assert_eq!(factor.solve(&b).expect("solve"), b);
}

#[test]
fn mixed_grounded_and_floating_components_solve_independently() {
    let (row_ptrs, columns) = ([0u32, 1, 3, 5], [0u32, 1, 2, 1, 2]);
    let values = [2.0, 1.0, -1.0, -1.0, 1.0];
    let factor = factor(Config::default(), csr(&row_ptrs, &columns, &values)).expect("factor");
    let solution = factor.solve(&[4.0, 1.0, -1.0]).expect("solve");
    assert!((solution[0] - 2.0).abs() < 1e-14);
    assert_eq!(&solution[1..], &[0.5, -0.5]);
}

/// Components `{0, 2}` and `{1, 3}` interleave, so the solve runs through a permutation;
/// a right-hand side that is not zero-sum per component gets the least-squares answer.
#[test]
fn interleaved_components_solve_in_input_order() {
    let (row_ptrs, columns) = ([0u32, 2, 4, 6, 8], [0u32, 2, 1, 3, 0, 2, 1, 3]);
    let values = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    let factor = factor(Config::default(), csr(&row_ptrs, &columns, &values)).expect("factor");

    let cases = [
        (
            "zero-sum per component",
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

/// Two grounded paths, `{0, 2}` and `{1, 3}`, interleaved: every component carries its
/// own ground slot, and the permutation must skip it on the way in and out. Paths are
/// trees, so even the approximate arm samples nothing and solves exactly.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn interleaved_grounded_components_solve_exactly(#[case] backend: Backend) {
    let (row_ptrs, columns) = ([0u32, 2, 4, 6, 8], [0u32, 2, 1, 3, 0, 2, 1, 3]);
    let values = [2.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 3.0];
    let csr = csr(&row_ptrs, &columns, &values);
    let sddm = Sddm::try_from(csr).expect("SDDM");
    assert!(matches!(sddm, Sddm::Grounded(_)));
    let config = Config {
        backend,
        ..Config::default()
    };
    let factor = factorize_with(sddm, config).expect("factor");
    assert_eq!(
        (factor.n(), factor.scratch_len()),
        (4, 6),
        "one ground slot per component"
    );

    let b = [1.0, -2.0, 3.0, 0.5];
    let x = factor.solve(&b).expect("solve");
    let residual = relative_residual_over(csr, &x, &b, 0..4);
    assert!(residual < 1e-14, "relative residual {residual:e}");
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
    let csr = csr(&row_ptrs, &columns, &values);
    for seed in 0..4u64 {
        let config = Config {
            seed,
            ..Config::default()
        };
        let factor = factor(config, csr).expect("double-cycle factor");
        assert_eq!(factor.n_steps(), (N - 2) as usize, "one pin per cycle");

        let x = factor.solve(&rhs).expect("solve");
        let residual = relative_residual_over(csr, &x, &rhs, 0..N as usize);
        assert!(residual < 1e-10, "seed={seed}: residual {residual:.3e}");
    }
}

/// A long path grounded at one end, where min-degree eliminates the ground first: the
/// solution is still measured from it, and a tree still solves exactly.
#[test]
fn a_ground_eliminated_early_still_anchors_the_solution() {
    let n = 64u32;
    let (mut row_ptrs, mut cols, mut vals) = (vec![0u32], Vec::new(), Vec::new());
    for v in 0..n {
        let mut row = Vec::new();
        if v > 0 {
            row.push((v - 1, -1.0));
        }
        let degree = f64::from(u8::from(v > 0) + u8::from(v + 1 < n));
        row.push((v, degree + if v == 0 { 0.25 } else { 0.0 }));
        if v + 1 < n {
            row.push((v + 1, -1.0));
        }
        for (col, value) in row {
            cols.push(col);
            vals.push(value);
        }
        row_ptrs.push(cols.len() as u32);
    }
    let csr = csr(&row_ptrs, &cols, &vals);
    let config = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };
    let factor = factor(config, csr).expect("factor");
    assert_eq!(
        factor.n_steps(),
        n as usize,
        "one slot per block stays free"
    );

    let b: Vec<f64> = (0..n).map(|v| f64::from(v % 7) - 3.0).collect();
    let x = factor.solve(&b).expect("solve");
    let residual = relative_residual_over(csr, &x, &b, 0..n as usize);
    assert!(residual < 1e-12, "relative residual {residual:e}");
}

proptest! {
    /// Only split grounded input is scanned per component, so the classes without surplus
    /// must come out floating and the one with surplus grounded.
    #[test]
    fn a_grounded_component_beside_floating_ones_solves_its_own_rows(
        ((row_ptrs, col_indices, values, n), parts) in one_grounded_component_strategy()
    ) {
        let rhs = per_component_consistent_rhs(n as usize, parts);
        let view = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect("valid CSR");
        let config = Config { seed: 11, ..Config::default() };
        let x = factor(config, view).expect("factorize").solve(&rhs).expect("solve");

        let residual = relative_residual_over(view, &x, &rhs, 0..rhs.len());
        prop_assert!(residual < 1e-9, "grounded component solved as floating: residual {residual:e}");
    }
}
