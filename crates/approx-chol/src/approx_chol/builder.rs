use super::config::{Backend, Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Permutation};
use crate::graph::{BlockVertices, EdgeCount, Ingestion, Multi, Single};
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
        let n = sddm.n();
        let ingestion = Ingestion::of(sddm);
        match self.config.split_factor() {
            None => self.factor_blocks::<Single>(ingestion, n, ()),
            Some(k) => self.factor_blocks::<Multi>(ingestion, n, k),
        }
        // The only scope holding both the caller's dimension and the finished factor.
        .inspect(|factor| debug_assert_eq!(factor.n(), n))
    }

    fn factor_blocks<C: EdgeCount>(
        &self,
        mut ingestion: Ingestion<T>,
        n: usize,
        split: C::Split,
    ) -> Result<Factor<T>, Error> {
        if ingestion.n() == 0 {
            return Ok(Factor::empty());
        }
        let mut factorizer = BlockFactorizer::<T, C>::new(self.config, split);
        let Some(layout) = ingestion.take_layout() else {
            let whole = BlockVertices::whole(ingestion.n());
            let (block, fallback) = factorizer.factor(&ingestion, &whole)?;
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
            let (block, fallback) = factorizer.factor(&ingestion, &view)?;
            blocks.push(block);
            fallbacks.extend(fallback);
        }
        // The ground is not an input vertex; its block keeps it as a slot of its own.
        let mut order = layout.into_order();
        order.retain(|&vertex| (vertex as usize) < n);
        Ok(Factor::from_blocks(
            Permutation::from_order(order),
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
        ingestion: &Ingestion<T>,
        block: &BlockVertices<'_>,
    ) -> Result<(Block<T>, Option<Fallback>), Error> {
        let (cholesky, fallback) = self.cholesky(ingestion, block)?;
        let gauged = if ingestion.carries_ground(block) {
            Block::Grounded(cholesky)
        } else {
            Block::Floating(cholesky)
        };
        Ok((gauged, fallback))
    }

    /// Routes first, so a block the dense backend claims never builds an elimination graph.
    fn cholesky(
        &mut self,
        ingestion: &Ingestion<T>,
        block: &BlockVertices<'_>,
    ) -> Result<(Cholesky<T>, Option<Fallback>), Error> {
        // Every block restarts, so one block's draws never shift because another went exact.
        self.sampler.restart(block.first());

        let eliminated = block
            .len()
            .checked_sub(1)
            .expect("a block has at least one vertex");
        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(eliminated) {
            match exact::factor(ingestion, block, eliminated) {
                Ok(lower) => return Ok((Cholesky::Exact(lower), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason.at(block))?);
                }
            }
        }
        let graph = ingestion.block_graph::<C>(block);
        let sequence = approximate::eliminate::<T, C>(graph, &mut self.sampler, self.split);
        Ok((Cholesky::Approximate(sequence), fallback))
    }
}
