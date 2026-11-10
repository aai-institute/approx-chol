//! faer's sparse matrices as [`CsrRef`] views, and its parallelism as a [`Threads`] budget.

use std::num::NonZeroUsize;

use faer::sparse::{SparseColMatRef, SparseRowMatRef};
use faer::Par;
use num_traits::PrimInt;

use super::square;
use crate::input::{CsrRef, InputError};
use crate::{Real, Threads};

impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: SparseRowMatRef<'a, I, T>) -> Result<Self, InputError> {
        rows(square(matrix.nrows(), matrix.ncols())?, matrix)
    }
}

impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<SparseColMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: SparseColMatRef<'a, I, T>) -> Result<Self, InputError> {
        rows(square(matrix.nrows(), matrix.ncols())?, matrix.transpose())
    }
}

/// The `n` rows of a square `matrix` as a [`CsrRef`].
fn rows<'a, T: Real, I: faer::Index + PrimInt>(
    n: usize,
    matrix: SparseRowMatRef<'a, I, T>,
) -> Result<CsrRef<'a, T, I>, InputError> {
    let symbolic = matrix.symbolic();
    let row_ptrs = symbolic.row_ptr();
    // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
    if let Some(row) = (0..n).find(|&i| symbolic.row_range(i).end != row_ptrs[i + 1].zx()) {
        return Err(InputError::RowGap { row });
    }
    // faer may keep spare capacity past the last row.
    let end = row_ptrs[n].zx();
    let (col_indices, values) = (&symbolic.col_idx()[..end], &matrix.val()[..end]);
    CsrRef::new(n, row_ptrs, col_indices, values)
}

/// faer's `Par::Seq` is `Count(1)`, `Par::Rayon(n)` is `Count(n)`.
impl From<Par> for Threads {
    fn from(par: Par) -> Self {
        Threads::Count(NonZeroUsize::new(par.degree()).unwrap_or(NonZeroUsize::MIN))
    }
}
