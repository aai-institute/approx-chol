use super::config::{Backend, Config, Route};
use super::factorization::{
    approximate, exact, Anchor, Block, BlockDim, Cholesky, Fallback, Permutation,
};
use crate::graph::{AdjListGraph, EdgeCount, GraphBuild, Multi, MultiEdgeGraph, Single, SlimGraph};
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
        match self.config.split_factor() {
            None => {
                let build = SlimGraph::<T>::from_sddm(sddm)?;
                self.build_graph::<Single>(build, original_n, ())
            }
            Some(k) => {
                let build = MultiEdgeGraph::<T>::from_sddm(sddm)?;
                self.build_graph::<Multi>(build, original_n, k)
            }
        }
    }

    fn build_graph<C: EdgeCount>(
        &self,
        build: GraphBuild<AdjListGraph<C, T>, T>,
        original_n: usize,
        split: C::Split,
    ) -> Result<Factor<T>, Error> {
        let GraphBuild {
            graph,
            diagonal,
            layout,
            ground,
        } = build;
        if graph.n() == 0 {
            return Ok(Factor::empty(original_n));
        }
        let mut factorizer = BlockFactorizer::<T, C>::new(self.config, split);
        let Some(layout) = layout else {
            let (block, fallback) = factorizer.factor(Component::whole(graph, diagonal, ground))?;
            return Ok(Factor::from_blocks(
                original_n,
                None,
                vec![block],
                fallback.into_iter().collect(),
            ));
        };

        let mut blocks = Vec::with_capacity(layout.block_count());
        let mut fallbacks = Vec::new();
        let mut partition = Partition::new(graph, diagonal, ground);
        for vertices in layout.blocks() {
            let (block, fallback) = factorizer.factor(partition.take(vertices))?;
            blocks.push(block);
            fallbacks.extend(fallback);
        }
        Ok(Factor::from_blocks(
            original_n,
            Permutation::from_order(layout.into_order()),
            blocks,
            fallbacks,
        ))
    }
}

/// One block to factor, and how its local vertices name global ones.
struct Component<'v, C, T: Real> {
    graph: AdjListGraph<C, T>,
    diagonal: Vec<T>,
    anchor: Anchor,
    /// `None` when the block is the whole graph, so a local vertex is already global.
    vertices: Option<&'v [u32]>,
}

impl<C: EdgeCount, T: Real> Component<'_, C, T> {
    /// A local vertex is already a global one, so `0..n` is never materialized.
    fn whole(graph: AdjListGraph<C, T>, diagonal: Vec<T>, ground: Option<u32>) -> Self {
        let last_vertex = (graph.n() - 1) as u32;
        Self {
            graph,
            diagonal,
            anchor: Anchor::of_block(ground, last_vertex),
            vertices: None,
        }
    }
}

/// The ingested graph, drained one connected component at a time. A connected input
/// never builds one, so it keeps skipping this scratch.
struct Partition<C, T: Real> {
    graph: AdjListGraph<C, T>,
    diagonal: Vec<T>,
    ground: Option<u32>,
    local_of: Vec<u32>,
}

impl<C: EdgeCount, T: Real> Partition<C, T> {
    fn new(graph: AdjListGraph<C, T>, diagonal: Vec<T>, ground: Option<u32>) -> Self {
        Self {
            local_of: vec![0u32; graph.n()],
            graph,
            diagonal,
            ground,
        }
    }

    fn take<'v>(&mut self, vertices: &'v [u32]) -> Component<'v, C, T> {
        let last_vertex = *vertices
            .last()
            .expect("a component has at least one vertex");
        Component {
            anchor: Anchor::of_block(self.ground, last_vertex),
            graph: self.graph.take_component(vertices, &mut self.local_of),
            diagonal: vertices
                .iter()
                .map(|&vertex| self.diagonal[vertex as usize])
                .collect(),
            vertices: Some(vertices),
        }
    }
}

/// What every block shares, resolved — including the one RNG stream a fixed seed
/// promises, which is why [`Config`] does not survive construction.
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
        component: Component<'_, C, T>,
    ) -> Result<(Block<T>, Option<Fallback>), Error> {
        let dim = BlockDim::of(component.graph.n()).expect("a block has at least one vertex");
        let anchor = component.anchor;
        let mut fallback = None;
        if let Route::Exact { on_failure } = self.backend.route(dim) {
            match exact::factor(&component.graph, &component.diagonal, dim) {
                Ok(lower) => return Ok((Block::new(dim, anchor, Cholesky::Exact(lower)), None)),
                Err(reason) => {
                    fallback = Some(on_failure.accept(reason.at(component.vertices))?);
                }
            }
        }
        let sequence = approximate::eliminate::<T, C>(
            component.graph,
            component.diagonal,
            &mut self.sampler,
            self.split,
        );
        Ok((
            Block::new(dim, anchor, Cholesky::Approximate(sequence)),
            fallback,
        ))
    }
}

#[cfg(test)]
mod tests;
