//! Other crates' types as this crate's, one module per crate behind its feature: sprs views, faer's row-major sparse matrices and nalgebra-sparse `CsrMatrix`es as [`CsrRef`](crate::input::CsrRef) views, checked square first; faer's dense matrices, nalgebra matrices and ndarray arrays as blocks of right-hand sides; faer's `Par` as a budget, and an [`ApproxInverse`](crate::ApproxInverse) as faer's operator.
//!
//! An error names each sparse array by the source's own name for it. Only row-major (CSR) storage converts: a CSC read as a CSR is its transpose, which the symmetry check accepts within 8 ulps, so it would factor another matrix; a sprs view stored by columns is [`InputError::StoredByColumns`](crate::input::InputError::StoredByColumns), and each library converts its own CSC to CSR.

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
