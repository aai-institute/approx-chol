use num_traits::PrimInt;

use super::Part;
use crate::input::{Columns, Csr};
use crate::Real;

/// Proves which parts of one family alias exactly, so floating never rests on rounded totals alone.
pub(super) enum FloatingTest<'v, T, I> {
    /// A Gram's: `fᵀGf ≤ 10u·fᵀDf`, summed as squares per column of `V`.
    Observations(Columns<'v, T, I>),
    /// A CSR's: each row's surplus within the rounding its own additions can leave.
    RowFloor(Csr<'v, T, I>),
}

impl<T: Real, I: PrimInt> FloatingTest<'_, T, I> {
    /// One pass for the whole family, in `parts`' order.
    pub(super) fn floating(&self, parts: &[Part<T>]) -> Vec<bool> {
        todo!()
    }
}
