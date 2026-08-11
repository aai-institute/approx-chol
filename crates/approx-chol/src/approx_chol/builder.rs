use super::config::{Backend, Config, Route};
use super::factorization::{
    approximate, exact, Anchor, Block, BlockDim, Cholesky, Fallback, Permutation,
};
use crate::graph::{BlockVertices, EdgeCount, Ingestion, Multi, Single};
use crate::sampling::CdfSampler;
use crate::types::Real;
use crate::{CsrError, CsrRef, Error, Factor};
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
        let narrowed = csr.narrow_indices()?;
        self.build_validated(narrowed.with_values(csr.values()))
    }

    /// The multiplicity decides layout and split together, so each arm names one
    /// algorithm end to end.
    fn build_validated(&self, sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error> {
        let original_n = sddm.n();
        let ingestion = Ingestion::of(sddm)?;
        match self.config.split_factor() {
            None => self.factor_blocks::<Single>(ingestion, ()),
            Some(k) => self.factor_blocks::<Multi>(ingestion, k),
        }
        // The only scope holding both the caller's dimension and the finished factor.
        .inspect(|factor| debug_assert_eq!(factor.original_n(), original_n))
    }

    fn factor_blocks<C: EdgeCount>(
        &self,
        mut ingestion: Ingestion<'_, T>,
        split: C::Split,
    ) -> Result<Factor<T>, Error> {
        if ingestion.n() == 0 {
            return Ok(Factor::empty());
        }
        let mut factorizer = BlockFactorizer::<T, C>::new(self.config, split);
        let Some(layout) = ingestion.take_layout() else {
            let whole = BlockVertices::whole(ingestion.n());
            let (block, fallback) = factorizer.factor(&mut ingestion, &whole)?;
            return Ok(Factor::from_blocks(
                None,
                vec![block],
                fallback.into_iter().collect(),
            ));
        };

        let mut blocks = Vec::with_capacity(layout.block_count());
        let mut fallbacks = Vec::new();
        // Scratch reused across blocks; each view refills the entries it names.
        let mut local_of = vec![0u32; ingestion.n()];
        for vertices in layout.blocks() {
            let view = BlockVertices::part(vertices, &mut local_of);
            let (block, fallback) = factorizer.factor(&mut ingestion, &view)?;
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

/// What every block shares, resolved — including the sampler each block restarts its
/// own stream from, which is why [`Config`] does not survive construction.
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

    /// Routing first, so a block the dense backend claims never has an elimination
    /// graph built for it — only a fallback from that arm, or the approximate route,
    /// reaches [`Ingestion::block_graph`].
    fn factor(
        &mut self,
        ingestion: &mut Ingestion<'_, T>,
        block: &BlockVertices<'_>,
    ) -> Result<(Block<T>, Option<Fallback>), Error> {
        // Restarts for every block, routed or not, so one block's draws never shift
        // because another was factored exactly.
        self.sampler.restart(block.first());

        let dim = BlockDim::of(block.len()).expect("a block has at least one vertex");
        let anchor = if ingestion.carries_ground(block) {
            Anchor::Ground
        } else {
            Anchor::Floating
        };
        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(dim) {
            match exact::factor(ingestion, block, dim) {
                Ok(lower) => return Ok((Block::new(dim, anchor, Cholesky::Exact(lower)), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason.at(block))?);
                }
            }
        }
        let graph = ingestion.block_graph::<C>(block);
        let diagonal = ingestion.take_block_diagonal(block);
        let sequence =
            approximate::eliminate::<T, C>(graph, diagonal, &mut self.sampler, self.split);
        Ok((
            Block::new(dim, anchor, Cholesky::Approximate(sequence)),
            fallback,
        ))
    }
}

#[cfg(test)]
mod tests;
