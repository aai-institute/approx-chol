use super::decomposition::{BlockFactor, EliminationSequence, Permutation, Pin};
use crate::graph::{AdjListGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::{DegreeDeltas, DynamicOrdering};
use crate::sampling::CdfSampler;
use crate::{ConfigError, CsrError, CsrRef, Error, Factor};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::clique_tree::SampledColumn;
use super::star::{Ac2StarBuilder, AcStarBuilder, StarBuilderVariant};
use super::Config;

/// Builder for approximate Cholesky factorization (Algorithm 8, Gao-Kyng-Spielman 2023).
///
/// Provides full control over the factorization pipeline, including
/// AC vs AC2 selection and seed control. `Builder::new(config).build(sddm)` is
/// what [`factorize_with`](crate::factorize_with) runs, and that is where the
/// worked example lives; most callers should prefer it or
/// [`factorize`](crate::factorize).
#[derive(Debug, Clone)]
pub struct Builder<T = f64> {
    config: Config,
    _scalar: core::marker::PhantomData<T>,
}

impl<T> Builder<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    /// Create a new builder with the given configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            config,
            _scalar: core::marker::PhantomData,
        }
    }

    /// Run approximate Cholesky factorization from any input fallibly convertible into
    /// [`CsrRef`].
    ///
    /// Performs a checked conversion of row pointers and column indices to owned
    /// `u32` storage; the values stay borrowed.
    ///
    /// # Errors
    ///
    /// As [`factorize_with`](crate::factorize_with).
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

    /// Assumes `sddm` already passed [`CsrRef::new`] validation (as
    /// [`build`](Self::build) guarantees); does not re-validate.
    fn build_validated(&self, sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error> {
        let original_n = sddm.n();
        Self::validate_config(self.config)?;
        match self.config.split_merge {
            None => {
                let build = SlimGraph::<T>::from_sddm(sddm)?;
                self.build_graph(build, original_n, AcStarBuilder::new)
            }
            Some(k) => {
                let mut build = MultiEdgeGraph::<T>::from_sddm(sddm)?;
                build.graph.mark_split_edges(k);
                let star = move |n| Ac2StarBuilder::new(n, k);
                self.build_graph(build, original_n, star)
            }
        }
    }

    /// `make_star` comes from the caller's single variant match and is applied per
    /// component, so nothing here can pair an AC star builder with a split
    /// multi-edge graph.
    fn build_graph<B: StarBuilderVariant<T>>(
        &self,
        build: GraphBuild<AdjListGraph<B::Count, T>, T>,
        original_n: usize,
        make_star: impl Fn(usize) -> B,
    ) -> Result<Factor<T>, Error> {
        let GraphBuild {
            mut graph,
            diagonal,
            components,
        } = build;
        let n = graph.n();
        if n == 0 {
            return Ok(Factor::empty(original_n));
        }
        // One stream for the whole factorization: the blocks are drawn from it in
        // component order, which is fixed by the input.
        let mut sampler = CdfSampler::<T>::new(self.config.seed);
        // Ground has the highest index, so it is the last vertex of its block.
        let ground_vertex = (n > original_n).then_some(original_n as u32);
        let Some(components) = components else {
            let block =
                self.build_from_graph(graph, diagonal, ground_vertex, &mut sampler, &make_star);
            return Ok(Factor::from_blocks(n, original_n, None, vec![block]));
        };

        let mut blocks = Vec::with_capacity(components.len());
        let mut local_of = vec![0u32; n];
        for vertices in &components {
            let ground = ground_vertex
                .filter(|vertex| vertices.last() == Some(vertex))
                .map(|_| (vertices.len() - 1) as u32);
            let component_graph = graph.take_component(vertices, &mut local_of);
            let component_diagonal = vertices
                .iter()
                .map(|&vertex| diagonal[vertex as usize])
                .collect();
            blocks.push(self.build_from_graph(
                component_graph,
                component_diagonal,
                ground,
                &mut sampler,
                &make_star,
            ));
        }
        let permutation = Permutation::from_forward(&components.concat());
        Ok(Factor::from_blocks(n, original_n, permutation, blocks))
    }

    fn validate_config(config: Config) -> Result<(), Error> {
        let Some(split_merge) = config.split_merge else {
            return Ok(());
        };
        if split_merge == 0 {
            return Err(Error::InvalidConfig(
                ConfigError::SplitMergeMustBePositive { split_merge },
            ));
        }
        Ok(())
    }

    /// Algorithm 8 loop on a pre-built graph. The graph's multiplicity storage is
    /// the star builder's `Count`, so an AC builder over a split multi-edge graph
    /// does not compile.
    fn build_from_graph<B: StarBuilderVariant<T>>(
        &self,
        mut graph: AdjListGraph<B::Count, T>,
        mut diag: Vec<T>,
        ground: Option<u32>,
        sampler: &mut CdfSampler<T>,
        make_star: &impl Fn(usize) -> B,
    ) -> BlockFactor<T> {
        let n = graph.n();
        let mut star_builder = make_star(n);
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
        // The bucket layout scales with the multiplicity the graph was split at,
        // read from the same `Config` the split came from rather than threaded
        // alongside the star builder as a second copy that could disagree.
        let degree_scale = self.config.split_merge.unwrap_or(1) as usize;
        let mut ordering = DynamicOrdering::new(&degrees, degree_scale);
        let mut column = SampledColumn::<T>::new();
        let mut seq = EliminationSequence::with_capacity(n, degree_sum);
        let mut deltas = DegreeDeltas::new(n);
        let target_steps = n.saturating_sub(1);
        let mut steps_done = 0usize;
        while steps_done < target_steps {
            let Some(v) = ordering.next_vertex() else {
                break;
            };
            steps_done += 1;
            star_builder.build_star(&mut graph, v, &mut ordering);
            let star_entries = star_builder.entries();
            if star_entries.is_empty() {
                seq.record_isolated(v, diag[v]);
                graph.eliminate_vertex(v);
                continue;
            }

            star_builder.sample_column(diag[v], sampler, &mut column);
            seq.record_column(v, &column);

            graph.eliminate_vertex(v);
            for &(u, w) in star_entries {
                diag[u as usize] = diag[u as usize] - w;
            }

            // Batch this step's degree-estimate updates into one pq_move per
            // affected neighbor (net delta) instead of one per incident
            // fill/removal/merge event. Batching reorders equal-degree vertices
            // within a bucket, so the exact factor for a fixed seed can differ
            // from a per-edge version (quality unaffected; see CHANGELOG).
            column.apply_fill_in_delta(&mut graph, &mut diag, &mut deltas);
            star_builder.accumulate_removal_delta(&mut deltas);
            deltas.flush(&mut ordering);
        }

        // A grounded block pins its ground vertex; a floating one pins the single
        // vertex left un-eliminated after `n - 1` pops.
        let pin = match ground {
            Some(ground) => Pin::Ground(ground),
            None => Pin::Floating(ordering.next_vertex().unwrap_or(0) as u32),
        };
        BlockFactor::approx(n, pin, seq)
    }
}

#[cfg(test)]
mod tests;
