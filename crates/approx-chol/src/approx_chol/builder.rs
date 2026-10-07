use super::config::{Backend, Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Permutation};
use crate::graph::{components, BlockVertices, Component, EdgeCount, Multi, Single};
use crate::sampling::CdfSampler;
use crate::sddm::Sddm;
use crate::types::Real;
use crate::{CsrError, CsrRef, Error, Factor, Fallback};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Clone)]
/// Factorization pipeline behind [`factorize_with`](crate::factorize_with).
pub struct Builder<T = f64> {
    config: Config,
    _scalar: core::marker::PhantomData<T>,
}

impl<T> Builder<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    #[must_use]
    /// Create a builder with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            _scalar: core::marker::PhantomData,
        }
    }

    /// Factorize any input fallibly convertible into [`CsrRef`].
    pub fn build<'a, I, M>(&self, sddm: M) -> Result<Factor<T>, Error>
    where
        I: PrimInt + 'a + 'static,
        M: TryInto<CsrRef<'a, T, I>>,
        <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
    {
        let csr = catch_unwind(AssertUnwindSafe(|| sddm.try_into()))
            .map_err(|_| Error::InvalidCsr(CsrError::InputConversionPanicked))?;
        let csr = csr.map_err(Into::into)?;
        self.build_validated(csr)
    }

    /// Multiplicity fixes layout and split together, so each arm is one algorithm end to end.
    fn build_validated<I: PrimInt>(&self, csr: CsrRef<'_, T, I>) -> Result<Factor<T>, Error> {
        let sddm = Sddm::try_from(csr)?;
        match self.config.split_factor() {
            None => self.factor_blocks::<Single>(&sddm, ()),
            Some(k) => self.factor_blocks::<Multi>(&sddm, k),
        }
        // The only scope holding both the caller's dimension and the finished factor.
        .inspect(|factor| debug_assert_eq!(factor.n(), sddm.n()))
    }

    fn factor_blocks<C: EdgeCount>(
        &self,
        sddm: &Sddm<T>,
        split: C::Split,
    ) -> Result<Factor<T>, Error> {
        if sddm.n() == 0 {
            return Ok(Factor::empty());
        }
        let mut factorizer = BlockFactorizer::<T, C>::new(self.config, split);
        let Some(layout) = components(sddm) else {
            let whole = Component::new(sddm, BlockVertices::whole(sddm.n()));
            let (block, fallback) = factorizer.factor(&whole)?;
            return Ok(Factor::from_blocks(
                None,
                vec![block],
                fallback.into_iter().collect(),
            ));
        };

        let mut blocks = Vec::with_capacity(layout.block_count());
        let mut fallbacks = Vec::new();
        // Scratch reused across blocks; each view refills the entries it names.
        let mut local_of = vec![0u32; sddm.n()];
        for vertices in layout.blocks() {
            let component = Component::new(sddm, BlockVertices::part(vertices, &mut local_of));
            let (block, fallback) = factorizer.factor(&component)?;
            blocks.push(block);
            fallbacks.extend(fallback);
        }
        Ok(Factor::from_blocks(
            Permutation::from_order(layout.into_order()),
            blocks,
            fallbacks,
        ))
    }
}

/// Per-block shared state; [`Config`] does not survive construction because the sampler replaces it.
struct BlockFactorizer<T: Real, C: EdgeCount> {
    backend: Backend,
    sampler: CdfSampler<T>,
    split: C::Split,
}

impl<T: Real, C: EdgeCount> BlockFactorizer<T, C> {
    fn new(config: Config, split: C::Split) -> Self {
        Self {
            sampler: CdfSampler::new(config.seed),
            backend: config.backend,
            split,
        }
    }

    fn factor(
        &mut self,
        component: &Component<'_, T>,
    ) -> Result<(Block<T>, Option<Fallback>), Error> {
        let (cholesky, fallback) = self.cholesky(component)?;
        let gauged = if component.is_grounded() {
            Block::Grounded(cholesky)
        } else {
            Block::Floating(cholesky)
        };
        Ok((gauged, fallback))
    }

    /// Routes first, so a component the dense backend claims never builds an elimination graph.
    fn cholesky(
        &mut self,
        component: &Component<'_, T>,
    ) -> Result<(Cholesky<T>, Option<Fallback>), Error> {
        // Every block restarts, so one block's draws never shift because another went exact.
        self.sampler.restart(component.first());

        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(component.eliminated()) {
            match exact::factor(component) {
                Ok(lower) => return Ok((Cholesky::Exact(lower), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason.at(component))?);
                }
            }
        }
        let adjacency = component.graph::<C>();
        let sequence = approximate::eliminate::<T, C>(adjacency, &mut self.sampler, self.split);
        Ok((Cholesky::Approximate(sequence), fallback))
    }
}
