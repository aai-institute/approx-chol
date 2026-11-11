//! nalgebra-sparse `CsrMatrix`es as [`CsrRef`] views and nalgebra matrices as blocks of right-hand sides.

use super::square;
use crate::input::{CsrNames, CsrRef, InputError};
use crate::{MatrixMut, Real};

/// nalgebra-sparse's names of a `CsrMatrix`'s arrays, its accessors'.
const NAMES: CsrNames = CsrNames {
    row_ptrs: "row_offsets",
    col_indices: "col_indices",
    values: "values",
};

impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CsrMatrix<T>> for CsrRef<'a, T, usize> {
    type Error = InputError;

    fn try_from(matrix: &'a nalgebra_sparse::CsrMatrix<T>) -> Result<Self, InputError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let (row_offsets, col_indices, values) = matrix.csr_data();
        CsrRef::named(n, row_offsets, col_indices, values, NAMES)
    }
}

/// Every nalgebra matrix, owned or a view, of any shape and strides, staged through scratch like any block reached by its columns.
///
/// A contiguous owned matrix skips the staging copy as `ColumnMajor::new(m.as_mut_slice(), m.nrows())`, a vector as `m.as_mut_slice()`. nalgebra's inherent `rows` and `columns` shadow this trait's: call `MatrixMut::rows(&m)`.
impl<T, R, C, S> MatrixMut<T> for nalgebra::Matrix<T, R, C, S>
where
    T: Real,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorageMut<T, R, C>,
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn columns(&self) -> usize {
        self.ncols()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        let shape = (nalgebra::Dyn(self.nrows()), nalgebra::Const::<1>);
        let rows = into.len();
        nalgebra::DVectorViewMut::from_slice(into, rows)
            .copy_from(&self.generic_view((0, column), shape));
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.column_mut(column).copy_from_slice(from);
    }
}
