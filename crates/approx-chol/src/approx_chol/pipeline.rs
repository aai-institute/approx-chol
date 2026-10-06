use super::config::{Config, Route};
use super::factorization::{approximate, exact, Block, Cholesky, Permutation};
use crate::graph::{Components, EdgeCount, Multi, Single, SplitFactor};
use crate::sampling::CdfSampler;
use crate::types::Real;
use crate::{Factor, Sddm, UnusablePivot};

/// The multiplicity fixes layout and split together, so each arm names one algorithm end to end.
pub(crate) fn factorize<T: Real>(
    sddm: &Sddm<T>,
    config: Config,
) -> Result<Factor<T>, UnusablePivot> {
    match config.split_merge.and_then(SplitFactor::new) {
        None => factor_components::<T, Single>(sddm, config, ()),
        Some(k) => factor_components::<T, Multi>(sddm, config, k),
    }
}

fn factor_components<T: Real, C: EdgeCount>(
    sddm: &Sddm<T>,
    Config { seed, backend, .. }: Config,
    split: C::Split,
) -> Result<Factor<T>, UnusablePivot> {
    let components = Components::of(sddm);
    let mut sampler = CdfSampler::new(seed);
    let mut blocks = Vec::with_capacity(components.len());
    let mut fallbacks = Vec::new();
    for component in components.iter() {
        // Restarts for every block so one block's draws never shift because another went exact.
        sampler.restart(component.first());
        // Routes first, so a block the dense backend claims never builds an elimination graph.
        if let Route::Exact { on_failure } = backend.route(component.eliminated()) {
            match exact::factor(&component) {
                Ok(lower) => {
                    blocks.push(Block::of(&component, Cholesky::Exact(lower)));
                    continue;
                }
                Err(reason) => fallbacks.push(on_failure.accept(reason)?),
            }
        }
        let sequence = approximate::eliminate::<T, C>(component.graph::<C>(), &mut sampler, split);
        blocks.push(Block::of(&component, Cholesky::Approximate(sequence)));
    }
    let permutation = components.into_order().and_then(Permutation::from_order);
    Ok(Factor::from_blocks(permutation, blocks, fallbacks))
}

#[cfg(test)]
mod tests;
