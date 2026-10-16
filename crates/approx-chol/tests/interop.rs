#![cfg(any(feature = "sprs", feature = "faer"))]

#[cfg(feature = "faer")]
#[path = "interop/faer.rs"]
mod faer;
#[cfg(feature = "sprs")]
#[path = "interop/sprs.rs"]
mod sprs;

use approx_chol::input::{CsrRef, Input};
use approx_chol::{factorize, Error, Factor};
use num_traits::PrimInt;

/// The 4-vertex path 0-1-2-3 with unit weights.
const N: usize = 4;
const ROW_PTRS: [usize; 5] = [0, 2, 5, 8, 10];
const COL_INDICES: [usize; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
const VALUES: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

fn indices<I: PrimInt>(fixture: &[usize]) -> Vec<I> {
    let index = |&i: &usize| num_traits::cast(i).expect("fixture index fits");
    fixture.iter().map(index).collect()
}

fn assert_solves_fixture<'a, I, M>(matrix: M)
where
    I: PrimInt + Sync + 'a,
    M: Copy
        + TryInto<CsrRef<'a, f64, I>, Error = Error>
        + TryInto<Input<'a, f64, I>, Error = Error>,
{
    let csr: CsrRef<'a, f64, I> = matrix.try_into().expect("valid CSR view");
    assert_eq!(csr.n(), N);
    let b = [1.0, -1.0, 1.0, -1.0];
    let x = factorize(csr)
        .expect("the path is SDDM")
        .solve(&b)
        .expect("solve");
    // The matrix factors directly too, as `factorize(&matrix)` did before 0.7.
    let direct: Factor = factorize(matrix).expect("the path is SDDM");
    let bits = |x: &[f64]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&direct.solve(&b).expect("solve")), bits(&x));
    for (row, &b) in b.iter().enumerate() {
        let lx: f64 = (ROW_PTRS[row]..ROW_PTRS[row + 1])
            .map(|k| VALUES[k] * x[COL_INDICES[k]])
            .sum();
        assert!((lx - b).abs() < 1e-10, "row {row}: (Lx) = {lx}, b = {b}");
    }
    // The path floats, so the minimum-norm solution is the zero-mean one.
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    assert!(mean.abs() < 1e-10, "solution is not zero-mean");
}
