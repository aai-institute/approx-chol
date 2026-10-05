use super::config::{Backend, Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Fallback, Permutation};
use crate::graph::{Component, Components, EdgeCount, Multi, Single};
use crate::sampling::CdfSampler;
use crate::types::Real;
use crate::{Factor, Sddm, UnusablePivot};

/// The multiplicity decides layout and split together, so each arm names one algorithm
/// end to end.
pub(crate) fn factorize<T: Real>(
    sddm: &Sddm<T>,
    config: Config,
) -> Result<Factor<T>, UnusablePivot> {
    match config.split_factor() {
        None => factor_components::<T, Single>(sddm, config, ()),
        Some(k) => factor_components::<T, Multi>(sddm, config, k),
    }
}

fn factor_components<T: Real, C: EdgeCount>(
    sddm: &Sddm<T>,
    config: Config,
    split: C::Split,
) -> Result<Factor<T>, UnusablePivot> {
    if sddm.n() == 0 {
        return Ok(Factor::empty());
    }
    let components = Components::of(sddm);
    let mut factorizer = BlockFactorizer::<T, C>::new(config, split);
    let mut blocks = Vec::with_capacity(components.len());
    let mut fallbacks = Vec::new();
    for component in components.iter() {
        let (block, fallback) = factorizer.factor(&component)?;
        blocks.push(block);
        fallbacks.extend(fallback);
    }
    let permutation = components.into_order().and_then(Permutation::from_order);
    Ok(Factor::from_blocks(permutation, blocks, fallbacks))
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

    /// Routing first, so a component the dense backend claims never has an elimination
    /// graph built for it — only a fallback from that arm, or the approximate route,
    /// reaches [`Component::graph`].
    fn factor(
        &mut self,
        component: &Component<'_, T>,
    ) -> Result<(Block<T>, Option<Fallback>), UnusablePivot> {
        // Restarts for every block, routed or not, so one block's draws never shift
        // because another was factored exactly.
        self.sampler.restart(component.first());

        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(component.eliminated()) {
            match exact::factor(component) {
                Ok(lower) => return Ok((Block::of(component, Cholesky::Exact(lower)), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason)?);
                }
            }
        }
        let sequence =
            approximate::eliminate::<T, C>(component.graph::<C>(), &mut self.sampler, self.split);
        Ok((
            Block::of(component, Cholesky::Approximate(sequence)),
            fallback,
        ))
    }
}

#[cfg(test)]
mod tests;
