//! Shared assertion for the path-Laplacian solve that the generic, sprs and faer
//! suites each run over their own index and value types.

use approx_chol::Factor;
use num_traits::{Float, FromPrimitive};

/// Solve the alternating-sign RHS on the 4-node path fixture and assert the
/// result is finite and not the trivial zero vector.
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
}
