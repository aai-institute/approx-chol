mod csr;
mod scaling;
mod surplus;
mod views;

use std::fmt::Debug;

use approx_chol::input::{CsrRef, UpperCsrRef};
use approx_chol::{factorize, factorize_with, ApproxInverse, Config, Error, Real, ScalingPolicy};

/// `n` follows from `row_ptrs`, so no case can disagree with its own row count.
fn reduce<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
) -> Result<ApproxInverse<T>, Error> {
    reduce_with(row_ptrs, columns, values, ScalingPolicy::default())
}

fn reduce_with<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
    policy: ScalingPolicy,
) -> Result<ApproxInverse<T>, Error> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, columns, values, n).expect("valid shape");
    factorize_with(csr, Config::default(), policy)
}

fn reduce_upper<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
) -> Result<ApproxInverse<T>, Error> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = UpperCsrRef::new(row_ptrs, columns, values, n).expect("valid shape");
    factorize(csr)
}

/// For input reducing to one balanced block: grounded exactly when it does not float.
fn grounded<T: Real>(factor: &ApproxInverse<T>) -> bool {
    assert_eq!(factor.blocks().len(), 1, "one block");
    !factor.blocks()[0].floats()
}

/// Error order is not the contract: a rejection need only name one of the input's defects.
fn assert_names_a_defect<E: PartialEq + Debug>(observed: E, defects: &[E]) {
    assert!(
        defects.contains(&observed),
        "{observed:?} names none of {defects:?}"
    );
}
