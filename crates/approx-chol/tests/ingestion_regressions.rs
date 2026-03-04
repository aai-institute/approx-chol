#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef};

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
    factor
        .solve_into_with_projection(rhs, &mut work, false)
        .or_panic("solve_into_with_projection should succeed");
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
