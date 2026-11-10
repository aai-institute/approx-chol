#![cfg(any(feature = "sprs", feature = "faer", feature = "nalgebra"))]

#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "sprs")]
mod sprs;

use approx_chol::factorize;
use approx_chol::input::{CsrRef, InputError};
use num_traits::PrimInt;

use crate::common::solve::Solve;

/// The 4-vertex path 0-1-2-3 with unit weights.
const N: usize = 4;
const ROW_PTRS: [usize; 5] = [0, 2, 5, 8, 10];
const COL_INDICES: [usize; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
const VALUES: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

fn assert_solves_fixture<'a, I, M>(matrix: M)
where
    I: PrimInt + Sync + 'a,
    M: TryInto<CsrRef<'a, f64, I>, Error = InputError>,
{
    let csr: CsrRef<'a, f64, I> = matrix.try_into().expect("valid CSR view");
    let factor = factorize(csr).expect("the path is SDDM");
    assert_eq!(factor.n(), N);
    let b = [1.0, -1.0, 1.0, -1.0];
    let x = factor.solve(&b).expect("solve");
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
