use num_traits::PrimInt;

use super::{into_sgdd, Signed, SymmetricBlock};
use crate::input::{Canonical, Csr, UpperCanonical, UpperCsr};
use crate::sgdd::block::{FromCsrError, ReduceError, Sgdd};
use crate::sgdd::decomposition::{of_csr, of_upper, Partitioned};
use crate::sgdd::scaling::ScalingPolicy;
use crate::Real;

impl<T: Real, I: PrimInt + Sync> Partitioned<Canonical<'_, T, I>> {
    /// Scales and signs each block's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        self.extract(|cells| {
            let Some(a) = SymmetricBlock::live(cells) else {
                return Ok(None);
            };
            let scaling = a.scale(policy)?;
            let block = match a.sign(scaling, policy)? {
                Signed::Folded(m, rows) => {
                    let grounding = rows.folded_surplus(&m.off_diagonal);
                    m.balanced(grounding)?
                }
                Signed::Frustrated(block) => block,
            };
            Ok(Some(block))
        })
        .map(into_sgdd)
    }
}

impl<T: Real> Sgdd<T> {
    /// Checks symmetry, then scales each block to dominance unless it already is.
    pub fn from_csr<I: PrimInt + Sync>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_csr(Canonical::of(csr)?)?.reduce(&policy)?)
    }

    /// As [`from_csr`](Self::from_csr), the triangle symmetric by storage: no mirror is checked.
    pub fn from_upper_csr<I: PrimInt + Sync>(
        csr: UpperCsr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_upper(UpperCanonical::of(csr)?)?.reduce(&policy)?)
    }
}
