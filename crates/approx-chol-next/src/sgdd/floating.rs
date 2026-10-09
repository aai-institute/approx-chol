use num_traits::PrimInt;

use super::{Part, Reduced};
use crate::input::{Columns, Csr};
use crate::Real;

/// Proves which parts of one family alias exactly, so floating never rests on rounded totals alone.
pub(super) enum FloatingTest<'v, T, I> {
    /// A signless Laplacian's: zero surplus by shape, so every part floats.
    Signless,
    /// A Gram's: `fᵀGf ≤ 10u·fᵀDf`, summed as squares per column of `V`.
    Observations(Columns<'v, T, I>),
    /// A CSR's: each row's surplus within the rounding its own additions can leave.
    RowFloor(Csr<'v, T, I>),
}

impl<T: Real, I: PrimInt> FloatingTest<'_, T, I> {
    /// One pass over the family decides each ground before its part exists.
    pub(super) fn parts(&self, reduced: Vec<Reduced<T>>) -> Vec<Part<T>> {
        match self {
            Self::Signless => (reduced.into_iter())
                .map(|part| part.into_part(true))
                .collect(),
            Self::Observations(_) | Self::RowFloor(_) => todo!(),
        }
    }
}
