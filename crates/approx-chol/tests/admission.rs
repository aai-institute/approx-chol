#[path = "admission/csr.rs"]
mod csr;
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

use approx_chol::input::{Csr, UpperCsr};
use approx_chol::{FromCsrError, Real, ScalingPolicy, Sgdd};

/// `n` follows from `row_ptrs`, so no case can disagree with its own row count.
fn reduce<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
) -> Result<Sgdd<T>, FromCsrError> {
    reduce_with(row_ptrs, columns, values, ScalingPolicy::default())
}

fn reduce_with<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
    policy: ScalingPolicy,
) -> Result<Sgdd<T>, FromCsrError> {
    let csr = Csr::new(row_ptrs.len() - 1, row_ptrs, columns, values).expect("valid shape");
    Sgdd::from_csr(csr, policy)
}

fn reduce_upper<T: Real>(
    row_ptrs: &[u32],
    columns: &[u32],
    values: &[T],
) -> Result<Sgdd<T>, FromCsrError> {
    let csr = UpperCsr::new(row_ptrs.len() - 1, row_ptrs, columns, values).expect("valid shape");
    Sgdd::from_upper_csr(csr, ScalingPolicy::default())
}

/// For input reducing to one balanced block: grounded exactly when it does not float.
fn grounded<T: Real>(sgdd: &Sgdd<T>) -> bool {
    assert_eq!(sgdd.blocks().len(), 1, "one block");
    !sgdd.blocks()[0].floats()
}

/// Error order is not the contract: a rejection need only name one of the input's defects.
fn assert_names_a_defect<E: PartialEq + Debug>(observed: E, defects: &[E]) {
    assert!(
        defects.contains(&observed),
        "{observed:?} names none of {defects:?}"
    );
}
