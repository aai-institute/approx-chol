use approx_chol::CsrRef;
use num_traits::{Float, PrimInt};

/// `||(Ax - b)[rows]|| / ||b[rows]||`, restricted to a row range so one block's
/// claim can be judged without the error the other blocks left.
pub fn relative_residual_over<T: Float, I: PrimInt>(
    csr: CsrRef<'_, T, I>,
    x: &[T],
    b: &[T],
    rows: core::ops::Range<usize>,
) -> T {
    let (row_ptrs, columns, values) = (csr.row_ptrs(), csr.col_indices(), csr.values());
    let (mut error, mut scale) = (T::zero(), T::zero());
    let at = |index: I| index.to_usize().expect("a CSR index fits usize");
    for (row, &target) in rows.clone().zip(&b[rows]) {
        let mut product = T::zero();
        for index in at(row_ptrs[row])..at(row_ptrs[row + 1]) {
            product = product + values[index] * x[at(columns[index])];
        }
        error = error + (product - target) * (product - target);
        scale = scale + target * target;
    }
    (error / scale).sqrt()
}
