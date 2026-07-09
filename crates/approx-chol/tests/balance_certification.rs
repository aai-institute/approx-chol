#[path = "common/panic_ok.rs"]
mod panic_ok;

use approx_chol::low_level::certify_balance;
use approx_chol::{CsrRef, Error};
use panic_ok::OrPanic;

type Csr = (Vec<u32>, Vec<u32>, Vec<f64>, u32);

fn certify(csr: &Csr) -> Result<Vec<i8>, Error> {
    let (rp, ci, vals, n) = csr;
    let csr = CsrRef::new(rp, ci, vals, *n).or_panic("valid csr");
    certify_balance(csr).map(|s| s.signs().to_vec())
}

fn assert_folds(csr: &Csr, signs: &[i8]) {
    let (rp, ci, vals, _) = csr;
    for row in 0..rp.len() - 1 {
        for k in rp[row] as usize..rp[row + 1] as usize {
            let col = ci[k] as usize;
            if row != col {
                let folded = signs[row] as f64 * vals[k] * signs[col] as f64;
                assert!(folded <= 0.0, "edge ({row},{col}) folds to {folded} > 0");
            }
        }
    }
}

#[test]
fn sign_free_input_short_circuits_to_trivial_signature() {
    let csr = (
        vec![0u32, 2, 5, 8, 10],
        vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        4,
    );
    assert_eq!(certify(&csr).or_panic("balanceable"), vec![1, 1, 1, 1]);
}

#[test]
fn balanceable_signed_input_yields_folding_signature() {
    // Path 0-1-2, edge 0-1 positive, edge 1-2 negative.
    let csr = (
        vec![0u32, 2, 5, 7],
        vec![0u32, 1, 0, 1, 2, 1, 2],
        vec![3.0, 1.0, 1.0, 3.0, -1.0, -1.0, 3.0],
        3,
    );
    let signs = certify(&csr).or_panic("balanceable");
    assert!(signs.iter().all(|&s| s == 1 || s == -1));
    assert_folds(&csr, &signs);
}

#[test]
fn frustrated_cycle_is_rejected_with_witness() {
    // 4-cycle with one positive edge: odd parity, unbalanceable.
    let csr = (
        vec![0u32, 3, 6, 9, 12],
        vec![0u32, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
        vec![
            4.0, 1.0, -1.0, 1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0,
        ],
        4,
    );
    let err = certify(&csr).expect_err("frustrated cycle must be rejected");
    let Error::FrustratedSigns { edge } = err else {
        panic!("expected FrustratedSigns, got {err:?}");
    };
    let (r, c) = edge;
    assert!(
        r < 4 && c < 4 && r != c,
        "witness edge out of range: {edge:?}"
    );
}
