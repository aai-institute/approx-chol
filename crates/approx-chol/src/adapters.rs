//! Other crates' types as this crate's, one module per crate behind its feature: sprs views, faer's sparse matrices and nalgebra-sparse matrices as [`CsrRef`](crate::input::CsrRef) views, checked square first; faer's dense matrices, nalgebra matrices and ndarray arrays as blocks of right-hand sides; faer's `Par` as a budget, and an [`ApproxInverse`](crate::ApproxInverse) as faer's operator.
//!
//! A column-major (CSC) matrix is read as the CSR of its transpose, which a symmetric one equals, so a storage defect names its column as the row.

#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "ndarray")]
mod ndarray;
#[cfg(feature = "sprs")]
mod sprs;

#[cfg(any(feature = "sprs", feature = "faer", feature = "nalgebra"))]
fn square(rows: usize, cols: usize) -> Result<usize, crate::input::InputError> {
    match rows == cols {
        true => Ok(rows),
        false => Err(crate::input::InputError::NotSquare { rows, cols }),
    }
}
