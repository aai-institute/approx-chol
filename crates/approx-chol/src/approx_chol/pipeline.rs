use super::config::{Backend, Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Fallback, Permutation};
use crate::graph::{BlockVertices, EdgeCount, Ingestion, Multi, Single};
use crate::sampling::CdfSampler;
use crate::types::Real;
use crate::{Factor, Sddm, UnusablePivot};

/// The multiplicity decides layout and split together, so each arm names one algorithm
/// end to end.
pub(crate) fn factorize<T: Real>(
    sddm: Sddm<T>,
    config: Config,
) -> Result<Factor<T>, UnusablePivot> {
    let ingestion = Ingestion::of(sddm);
    match config.split_factor() {
        None => factor_blocks::<T, Single>(ingestion, config, ()),
        Some(k) => factor_blocks::<T, Multi>(ingestion, config, k),
    }
}

fn factor_blocks<T: Real, C: EdgeCount>(
    mut ingestion: Ingestion<T>,
    config: Config,
    split: C::Split,
) -> Result<Factor<T>, UnusablePivot> {
    if ingestion.n() == 0 {
        return Ok(Factor::empty());
    }
    let mut factorizer = BlockFactorizer::<T, C>::new(config, split);
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
    Ok(Factor::from_blocks(
        Permutation::from_order(layout.into_order()),
        blocks,
        fallbacks,
    ))
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
        ingestion: &Ingestion<T>,
        block: &BlockVertices<'_>,
    ) -> Result<(Block<T>, Option<Fallback>), UnusablePivot> {
        // Restarts for every block, routed or not, so one block's draws never shift
        // because another was factored exactly.
        self.sampler.restart(block.first());

        let grounded = ingestion.is_grounded(block);
        let anchored = if grounded {
            Block::Grounded
        } else {
            Block::Floating
        };
        // The exact arm pins the last slot: a floating block's own last vertex, a grounded
        // one's ground.
        let eliminated = block.len() - usize::from(!grounded);
        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(eliminated) {
            match exact::factor(ingestion, block, eliminated) {
                Ok(lower) => return Ok((anchored(Cholesky::Exact(lower)), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason.at(block))?);
                }
            }
        }
        let graph = ingestion.block_graph::<C>(block, grounded);
        let sequence = approximate::eliminate::<T, C>(graph, &mut self.sampler, self.split);
        Ok((anchored(Cholesky::Approximate(sequence)), fallback))
    }
}

#[cfg(test)]
mod tests;
