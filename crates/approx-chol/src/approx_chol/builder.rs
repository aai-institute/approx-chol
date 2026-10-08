use super::config::{Backend, Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Permutation};
use crate::graph::{Component, EdgeCount, Multi, Single};
use crate::sampling::CdfSampler;
use crate::sddm::Sddm;
use crate::types::Real;
use crate::{Factor, Fallback, UnusablePivot};

/// Multiplicity fixes layout and split together, so each arm is one algorithm end to end.
pub(crate) fn factor<T: Real>(sddm: Sddm<T>, config: Config) -> Result<Factor<T>, UnusablePivot> {
    // The only scope holding both the caller's dimension and the finished factor.
    let n = sddm.n();
    match config.split_factor() {
        None => factor_blocks::<T, Single>(sddm, config, ()),
        Some(k) => factor_blocks::<T, Multi>(sddm, config, k),
    }
    .inspect(|factor| debug_assert_eq!(factor.n(), n))
}

fn factor_blocks<T: Real, C: EdgeCount>(
    sddm: Sddm<T>,
    config: Config,
    split: C::Split,
) -> Result<Factor<T>, UnusablePivot> {
    let mut factorizer = BlockFactorizer::<T, C>::new(config, split);
    let mut fallbacks = Vec::new();
    let (blocks, order) = sddm.map_components(|component| {
        let (block, fallback) = factorizer.factor(component)?;
        fallbacks.extend(fallback);
        Ok(block)
    })?;
    Ok(Factor::from_blocks(
        order.and_then(Permutation::from_order),
        blocks,
        fallbacks,
    ))
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
    ) -> Result<(Block<T>, Option<Fallback>), UnusablePivot> {
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
    ) -> Result<(Cholesky<T>, Option<Fallback>), UnusablePivot> {
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
