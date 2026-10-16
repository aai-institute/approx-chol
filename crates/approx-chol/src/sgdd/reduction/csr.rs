use num_traits::PrimInt;

use super::{into_sgdd, Signed, SymmetricBlock};
use crate::input::{Canonical, CsrRef, UpperCsrRef};
use crate::sgdd::block::{ReduceError, Sgdd};
use crate::sgdd::decomposition::{of_csr, of_upper, Partitioned};
use crate::sgdd::scaling::{self, ScalingPolicy};
use crate::Error;
use crate::{Index, Real};

impl<T: Real, J: PrimInt + Sync, I: Index> Partitioned<Canonical<'_, T, J>, I> {
    /// Scales and signs each block's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T, I>, ReduceError> {
        self.extract(|cells| {
            let Some(a) = SymmetricBlock::live(cells) else {
                return Ok(None);
            };
            let relax =
                |a: &SymmetricBlock<T, I>| scaling::relax(&a.off_diagonal, &a.diagonal, policy);
            let block = match a.reduce(policy, relax)? {
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

impl<T: Real, I: Index> Sgdd<T, I> {
    /// Checks symmetry, then scales each block to dominance unless it already is.
    pub fn from_csr<J: PrimInt + Sync>(
        csr: CsrRef<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, Error> {
        Ok(of_csr(Canonical::of(csr)?)?.reduce(&policy)?)
    }

    /// As [`from_csr`](Self::from_csr), the triangle symmetric by storage: no mirror is checked.
    pub fn from_upper_csr<J: PrimInt + Sync>(
        csr: UpperCsrRef<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, Error> {
        Ok(of_upper(Canonical::of(csr.0)?)?.reduce(&policy)?)
    }
}
