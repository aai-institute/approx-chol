#[path = "admission/csr.rs"]
mod csr;
#[path = "common/dense_residual.rs"]
mod dense_residual;
#[path = "common/gram.rs"]
mod gram;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/grid.rs"]
mod grid;
#[path = "admission/scaling.rs"]
mod scaling;
#[path = "admission/surplus.rs"]
mod surplus;
#[path = "admission/views.rs"]
mod views;

use std::fmt::Debug;

use approx_chol::input::{CsrRef, UpperCsrRef};
use approx_chol::{factorize, factorize_with, Config, Error, Factor, Real, ScalingPolicy};

/// `n` follows from `row_ptrs`, so no case can disagree with its own row count.
fn reduce<T: Real>(row_ptrs: &[u32], columns: &[u32], values: &[T]) -> Result<Factor<T>, Error> {
    reduce_with(row_ptrs, columns, values, ScalingPolicy::default())
}

fn reduce_with<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
    policy: ScalingPolicy,
) -> Result<Factor<T>, Error> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, columns, values, n).expect("valid shape");
    factorize_with(csr, Config::default(), policy)
}

fn reduce_upper<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
) -> Result<Factor<T>, Error> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = UpperCsrRef::new(row_ptrs, columns, values, n).expect("valid shape");
    factorize(csr)
}

/// For input reducing to one balanced block: grounded exactly when it does not float.
fn grounded<T: Real>(factor: &Factor<T>) -> bool {
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
