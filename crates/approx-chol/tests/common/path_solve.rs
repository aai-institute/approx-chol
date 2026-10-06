//! Path-Laplacian assertions the sprs and faer suites run once per index type.

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error};
use num_traits::PrimInt;

/// The whole input-adapter contract: the view reports the fixture's shape, and the matrix factorizes and solves.
pub fn assert_view_and_factor_match_fixture<'a, I, M>(matrix: M)
where
    M: TryInto<CsrRef<'a, f64, I>> + Copy,
    <M as TryInto<CsrRef<'a, f64, I>>>::Error: core::fmt::Debug + Into<Error>,
    I: PrimInt + 'static,
{
    let view: CsrRef<'a, f64, I> = matrix.try_into().expect("valid CSR view");
    assert_eq!(view.n(), super::path::N as usize);
    assert_eq!(view.row_ptrs().len(), super::path::ROW_PTRS.len());
    assert_eq!(view.col_indices().len(), super::path::COL_INDICES.len());
    assert_eq!(view.values().len(), super::path::VALUES.len());

    let factor = Builder::<f64>::new(Config::default())
        .build(matrix)
        .expect("factorization should succeed");
    assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));

    let b = [1.0, -1.0, 1.0, -1.0];
    let mut work = vec![0.0; factor.n()];
    factor
        .solve_into(&b, &mut work)
        .expect("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()), "solution not finite");
    assert!(
        work.iter().any(|x| x.abs() > 1e-6),
        "solution is trivially zero"
    );
    // The floating block has no ground vertex, so the zero-mean representative is the answer.
    let mean = work.iter().sum::<f64>() / work.len() as f64;
    assert!(mean.abs() < 1e-6, "solution is not zero-mean");
}
