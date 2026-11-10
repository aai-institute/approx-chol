//! Other crates' types as this crate's, one module per crate behind its feature: sprs views, faer's sparse matrices and nalgebra-sparse matrices as [`CsrRef`](crate::input::CsrRef) views, checked square first; faer's dense matrices and nalgebra matrices as blocks of right-hand sides; faer's `Par` as a budget.
//!
//! A column-major (CSC) matrix is read as the CSR of its transpose, which a symmetric one equals, so a storage defect names its column as the row.

#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "sprs")]
mod sprs;

use crate::input::InputError;

fn square(rows: usize, cols: usize) -> Result<usize, InputError> {
    match rows == cols {
        true => Ok(rows),
        false => Err(InputError::NotSquare { rows, cols }),
    }
}
