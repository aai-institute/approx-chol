use super::decomposition::{ComponentFactor, EliminationSequence, SingleFactor};
use crate::graph::{
    AdjListGraph, EdgeLike, EliminationGraph, GraphBuild, MultiEdgeGraph, SlimGraph,
};
use crate::ordering::{DegreeDeltas, DynamicOrdering};
use crate::sampling::{CdfSampler, WeightedSampler};
use crate::{ConfigError, CsrError, CsrRef, Error, Factor};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::clique_tree::SampledColumn;
use super::star::{Ac2StarBuilder, AcStarBuilder, StarBuilderVariant};
use super::Config;

/// Builder for approximate Cholesky factorization (Algorithm 8, Gao-Kyng-Spielman 2023).
///
/// Provides full control over the factorization pipeline, including
/// AC vs AC2 selection and seed control. Most users should prefer
/// [`factorize`](crate::factorize) or [`factorize_with`](crate::factorize_with).
///
/// # Examples
///
/// ```
/// use approx_chol::{Config, CsrRef};
/// use approx_chol::low_level::Builder;
///
/// let row_ptrs    = [0u32, 2, 5, 8, 10];
/// let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
///
/// let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
/// let factor = Builder::new(Config::default()).build(csr)?;
/// assert_eq!(factor.n(), 4);
/// # Ok::<(), approx_chol::Error>(())
/// ```
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
    /// Performs a checked conversion of row pointers and column indices to
    /// owned `u32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if conversion fails, conversion panics,
    /// CSR validation fails, or index conversion to `u32` fails.
    /// Returns [`Error::PositiveOffDiagonal`] if any off-diagonal entry is
    /// strictly positive (outside the SDDM/Laplacian class).
    /// Returns [`Error::InvalidConfig`] for invalid `split_merge`.
    /// Returns a structured numeric error for non-finite, asymmetric, or
    /// non-SDDM input.
    pub fn build<'a, I, M>(&self, sddm: M) -> Result<Factor<T>, Error>
    where
        I: PrimInt + 'a + 'static,
        M: TryInto<CsrRef<'a, T, I>>,
        <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
    {
        let csr = catch_unwind(AssertUnwindSafe(|| sddm.try_into()))
            .map_err(|_| Error::InvalidCsr(CsrError::InputConversionPanicked))?;
        let csr = csr.map_err(Into::into)?;
        let converted = csr.to_owned_u32()?;
        // Sole per-factorization CSR validation; graph ingestion trusts it.
        let converted_ref = converted.try_as_ref()?;
        self.build_validated(converted_ref)
    }

    fn build_validated(&self, sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error> {
        let original_n = sddm.n();
        Self::validate_config(self.config)?;
        let factor = match self.config.split_merge {
            None => {
                let build = SlimGraph::<T>::from_sddm(sddm)?;
                self.build_graph(build, original_n)
            }
            Some(k) => {
                let mut build = MultiEdgeGraph::<T>::from_sddm(sddm)?;
                build.graph.mark_split_edges(k);
                self.build_graph(build, original_n)
            }
        }?;
        debug_assert_eq!(factor.original_n(), original_n);
        Ok(factor)
    }

    fn build_graph<E: EdgeLike<T>>(
        &self,
        build: GraphBuild<AdjListGraph<E, T>, T>,
        original_n: usize,
    ) -> Result<Factor<T>, Error> {
        let GraphBuild {
            graph,
            diagonal,
            components,
        } = build;
        let n = graph.n();
        let Some(components) = components else {
            let vertices: Vec<u32> = (0..n as u32).collect();
            let use_dense = n == 0
                || (self.config.dense_threshold > 0 && original_n <= self.config.dense_threshold);
            let factor = if use_dense {
                let (matrix, pivots) = graph.dense_principal(&diagonal, &vertices);
                SingleFactor::dense(n, matrix, &pivots)?
            } else {
                let mut sampler = CdfSampler::<T>::new(self.config.seed);
                self.build_from_graph(graph, diagonal, &mut sampler)?
            };
            return Ok(Factor::single(original_n, factor));
        };

        let mut factors = Vec::with_capacity(components.len());
        let mut local_of = Vec::new();
        for vertices in components {
            let component_n = vertices
                .iter()
                .filter(|&&vertex| (vertex as usize) < original_n)
                .count();
            let use_dense =
                self.config.dense_threshold > 0 && component_n <= self.config.dense_threshold;
            let factor = if use_dense {
                let (matrix, pivots) = graph.dense_principal(&diagonal, &vertices);
                SingleFactor::dense(vertices.len(), matrix, &pivots)?
            } else {
                if local_of.is_empty() {
                    local_of.resize(n, usize::MAX);
                }
                let (component_graph, component_diagonal) =
                    graph.extract_component(&diagonal, &vertices, &mut local_of);
                let representative = vertices.first().copied().unwrap_or(0) as u64;
                let mut sampler =
                    CdfSampler::<T>::new(component_seed(self.config.seed, representative));
                self.build_from_graph(component_graph, component_diagonal, &mut sampler)?
            };
            factors.push(ComponentFactor { vertices, factor });
        }
        Ok(Factor::blocks(n, original_n, factors))
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

    /// Run factorization on a pre-built graph (fused pipeline path).
    pub(crate) fn build_from_graph<G: EliminationGraph<T>, S: WeightedSampler<T>>(
        &self,
        mut graph: G,
        diag: Vec<T>,
        sampler: &mut S,
    ) -> Result<SingleFactor<T>, Error> {
        let n = graph.n();
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
        let mut ordering = match self.config.split_merge {
            None => DynamicOrdering::new(n, degrees.into_iter()),
            Some(k) => DynamicOrdering::new_with_scale(n, degrees.into_iter(), k as usize),
        }
        .map_err(Error::InvalidCsr)?;
        self.factorize_with_ordering(&mut graph, diag, &mut ordering, degree_sum, sampler)
    }

    /// Dispatch on the clique-tree sampling variant (AC vs AC2).
    fn factorize_with_ordering<G: EliminationGraph<T>, S: WeightedSampler<T>>(
        &self,
        graph: &mut G,
        diag: Vec<T>,
        ordering: &mut DynamicOrdering,
        degree_sum: usize,
        sampler: &mut S,
    ) -> Result<SingleFactor<T>, Error> {
        let mut diag = diag;
        match self.config.split_merge {
            None => Ok(Self::factorize_with_variant(
                graph,
                &mut diag,
                ordering,
                degree_sum,
                sampler,
                AcStarBuilder::new(graph.n()),
            )),
            Some(k) => Ok(Self::factorize_with_variant(
                graph,
                &mut diag,
                ordering,
                degree_sum,
                sampler,
                Ac2StarBuilder::new(graph.n(), k),
            )),
        }
    }

    /// Algorithm 8 loop parameterized by a clique-tree sampling variant.
    fn factorize_with_variant<
        G: EliminationGraph<T>,
        W: WeightedSampler<T>,
        B: StarBuilderVariant<T>,
    >(
        graph: &mut G,
        diag: &mut [T],
        ordering: &mut DynamicOrdering,
        degree_sum: usize,
        sampler: &mut W,
        mut star_builder: B,
    ) -> SingleFactor<T> {
        let n = graph.n();
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
            if graph.is_empty(v) {
                seq.record_isolated(v, diag[v]);
                continue;
            }

            star_builder.build_star(graph, v, ordering);
            if star_builder.is_empty() {
                seq.record_isolated(v, diag[v]);
                graph.eliminate_vertex(v);
                continue;
            }

            let star_entries = star_builder.entries();
            star_builder.sample_column(diag[v], sampler, &mut column);
            seq.record_column(v, column.diagonal, &column.neighbors, &column.fractions);

            graph.eliminate_vertex(v);
            for &(u, w) in star_entries {
                diag[u as usize] = diag[u as usize] - w;
            }

            // Batch this step's degree-estimate updates into one pq_move per
            // affected neighbor (net delta) instead of one per incident
            // fill/removal/merge event. Batching reorders equal-degree vertices
            // within a bucket, so the exact factor for a fixed seed can differ
            // from a per-edge version (quality unaffected; see CHANGELOG).
            column.apply_fill_in_delta(graph, diag, &mut deltas);
            star_builder.accumulate_removal_delta(&mut deltas);
            deltas.flush(ordering);
        }

        SingleFactor::approx(n, seq)
    }
}

fn component_seed(seed: u64, representative: u64) -> u64 {
    let mut value = seed ^ representative.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests;
