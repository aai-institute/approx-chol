#[path = "common/residual.rs"]
mod residual;
use residual::relative_residual_over;

use approx_chol::{factorize_with, Backend, Config, CsrRef, Sddm};
use rstest::rstest;

/// Two grounded paths, `{0, 2}` and `{1, 3}`, interleaved: every component carries its
/// own ground slot, and the permutation must skip it on the way in and out.
const ROW_PTRS: [u32; 5] = [0, 2, 4, 6, 8];
const COL_INDICES: [u32; 8] = [0, 2, 1, 3, 0, 2, 1, 3];
const VALUES: [f64; 8] = [2.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 3.0];

/// Paths are trees, so even the approximate arm samples nothing and solves exactly.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn interleaved_grounded_components_solve_exactly(#[case] backend: Backend) {
    let csr = CsrRef::new(&ROW_PTRS, &COL_INDICES, &VALUES, 4).expect("valid CSR");
    let sddm = Sddm::try_from(csr).expect("SDDM");
    assert!(matches!(sddm, Sddm::Grounded(_)));
    let factor = factorize_with(
        sddm,
        Config {
            backend,
            ..Config::default()
        },
    )
    .expect("factor");
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
    let csr = CsrRef::new(&row_ptrs, &cols, &vals, n).expect("valid CSR");
    let factor = factorize_with(
        Sddm::try_from(csr).expect("SDDM"),
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    )
    .expect("factor");
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
