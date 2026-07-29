#![allow(dead_code)]

use approx_chol::CsrRef;
use num_traits::Float;

/// `||(Ax - b)[rows]|| / ||b[rows]||`, restricted to a row range so one block's
/// claim can be judged without the error the other blocks left.
pub fn relative_residual_over<T: Float>(
    csr: CsrRef<'_, T, u32>,
    x: &[T],
    b: &[T],
    rows: core::ops::Range<usize>,
) -> T {
    let (row_ptrs, columns, values) = (csr.row_ptrs(), csr.col_indices(), csr.values());
    let (mut error, mut scale) = (T::zero(), T::zero());
    for (row, &target) in rows.clone().zip(&b[rows]) {
        let mut product = T::zero();
        for index in row_ptrs[row] as usize..row_ptrs[row + 1] as usize {
            product = product + values[index] * x[columns[index] as usize];
        }
        error = error + (product - target) * (product - target);
        scale = scale + target * target;
    }
    (error / scale).sqrt()
}
