#![cfg(any(feature = "sprs", feature = "faer"))]

#[cfg(feature = "faer")]
#[path = "interop/faer.rs"]
mod faer;
#[cfg(feature = "sprs")]
#[path = "interop/sprs.rs"]
mod sprs;

use approx_chol_next::input::{Csr, CsrError};
use approx_chol_next::{factorize, Config, Factor, FromCsrError, ScalingPolicy, Sgdd};
use num_traits::PrimInt;

/// The 4-vertex path 0-1-2-3 with unit weights.
const N: usize = 4;
const ROW_PTRS: [usize; 5] = [0, 2, 5, 8, 10];
const COL_INDICES: [usize; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
const VALUES: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

/// What a caller writes: the conversion's error composes into `from_csr`'s through `?`.
fn factorize_matrix<'a, I, M>(matrix: M) -> Result<Factor, FromCsrError>
where
    I: PrimInt + Sync + 'a,
    M: TryInto<Csr<'a, f64, I>, Error = CsrError>,
{
    let sgdd = Sgdd::from_csr(matrix.try_into()?, ScalingPolicy::default())?;
    Ok(factorize(&sgdd, &Config::default()))
}

/// The whole converter contract: the view has the path's dimension, and the matrix factorizes and solves.
fn assert_view_and_factor_match_fixture<'a, I, M>(matrix: M)
where
    I: PrimInt + Sync + 'a,
    M: TryInto<Csr<'a, f64, I>, Error = CsrError> + Copy,
{
    let view: Csr<'a, f64, I> = matrix.try_into().expect("valid CSR view");
    assert_eq!(view.n(), N);

    let factor = factorize_matrix(matrix).expect("factorization should succeed");
    let x = factor.solve(&[1.0, -1.0, 1.0, -1.0]).expect("solve");
    assert!(x.iter().all(|x| x.is_finite()), "solution not finite");
    assert!(
        x.iter().any(|x| x.abs() > 1e-6),
        "solution is trivially zero"
    );
    // The path floats, so the minimum-norm solution is the zero-mean one.
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    assert!(mean.abs() < 1e-6, "solution is not zero-mean");
}
