//! Shared assertions for the path-Laplacian fixture that the generic, sprs and
//! faer suites each run over their own index and value types.

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error, Factor};
use num_traits::{Float, FromPrimitive, PrimInt};

/// The whole input-adapter contract for one matrix carrying the path fixture: the
/// borrowed view reports the fixture's shape, and the same matrix factorizes and
/// solves through `Builder::build`. All three conversion suites assert exactly
/// this, so none of them owns a copy of it.
pub fn assert_view_and_factor_match_fixture<'a, T, I, M>(matrix: M, config: Config)
where
    M: TryInto<CsrRef<'a, T, I>> + Copy,
    <M as TryInto<CsrRef<'a, T, I>>>::Error: core::fmt::Debug + Into<Error>,
    I: PrimInt + 'static,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static,
{
    let view: CsrRef<'a, T, I> = matrix.try_into().expect("valid CSR view");
    assert_eq!(view.n(), super::path::N as usize);
    assert_eq!(view.row_ptrs().len(), super::path::ROW_PTRS.len());
    assert_eq!(view.col_indices().len(), super::path::COL_INDICES.len());
    assert_eq!(view.values().len(), super::path::VALUES.len());

    let factor = Builder::<T>::new(config)
        .build(matrix)
        .expect("factorization should succeed");
    assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));
    assert_solves_path_rhs(&factor);
}

/// Solve the alternating-sign RHS on the 4-node path fixture and assert the
/// result is finite, not the trivial zero vector, and the zero-mean representative
/// the fixture's floating block has no ground vertex to pick for it.
pub fn assert_solves_path_rhs<T>(factor: &Factor<T>)
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static,
{
    let one = T::one();
    let b = [one, -one, one, -one];
    let mut work = vec![T::zero(); factor.n()];
    factor
        .solve_into(&b, &mut work)
        .expect("solve_into should succeed");

    assert!(work.iter().all(|x| x.is_finite()), "solution not finite");
    let min_signal = T::from_f64(1e-6).expect("1e-6 is representable");
    assert!(
        work.iter().any(|x| x.abs() > min_signal),
        "solution is trivially zero"
    );

    let count = T::from_usize(work.len()).expect("dimension is representable");
    let mean = work.iter().fold(T::zero(), |sum, &x| sum + x) / count;
    assert!(mean.abs() < min_signal, "solution is not zero-mean");
}
