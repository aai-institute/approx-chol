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
/// AC vs AC2 selection and seed control. Most users should prefer
/// [`factorize`](crate::factorize) or [`factorize_with`](crate::factorize_with).
///
/// # Examples
///
/// ```
/// use approx_chol::{Config, CsrRef};
/// use approx_chol::low_level::Builder;
/// # let row_ptrs    = [0u32, 2, 5, 8, 10];
/// # let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// # let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
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
                self.build_graph(build, original_n, AcStarBuilder::new, 1)
            }
            Some(k) => {
                let mut build = MultiEdgeGraph::<T>::from_sddm(sddm)?;
                build.graph.mark_split_edges(k);
                let star = move |n| Ac2StarBuilder::new(n, k);
                self.build_graph(build, original_n, star, k as usize)
            }
        }
    }

    /// `make_star` and `degree_scale` come from the caller's single variant match
    /// and are applied per component, so nothing here can pair an AC star builder
    /// with a split multi-edge graph.
    fn build_graph<B: StarBuilderVariant<T>>(
        &self,
        build: GraphBuild<AdjListGraph<B::Count, T>, T>,
        original_n: usize,
        make_star: impl Fn(usize) -> B,
        degree_scale: usize,
    ) -> Result<Factor<T>, Error> {
        let GraphBuild {
            graph,
            diagonal,
            components,
        } = build;
        let n = graph.n();
        if n == 0 {
            return Ok(Factor::empty(original_n));
        }
        // Ground has the highest index, so it is the last vertex of its block.
        let ground_vertex = (n > original_n).then_some(original_n as u32);
        let Some(components) = components else {
            let block = self.build_approximate(
                graph,
                diagonal,
                ground_vertex,
                self.config.seed,
                &make_star,
                degree_scale,
            );
            return Ok(Factor::from_blocks(n, original_n, None, vec![block]));
        };

        let mut forward: Vec<u32> = Vec::with_capacity(n);
        let mut blocks = Vec::with_capacity(components.len());
        let mut local_of = vec![usize::MAX; n];
        for vertices in components {
            let ground = ground_vertex
                .filter(|vertex| vertices.last() == Some(vertex))
                .map(|_| (vertices.len() - 1) as u32);
            let (component_graph, component_diagonal) =
                graph.extract_component(&diagonal, &vertices, &mut local_of);
            let representative = vertices.first().copied().unwrap_or(0) as u64;
            blocks.push(self.build_approximate(
                component_graph,
                component_diagonal,
                ground,
                component_seed(self.config.seed, representative),
                &make_star,
                degree_scale,
            ));
            forward.extend_from_slice(&vertices);
        }
        let permutation = Permutation::from_forward(&forward);
        Ok(Factor::from_blocks(n, original_n, permutation, blocks))
    }

    fn build_approximate<B: StarBuilderVariant<T>>(
        &self,
        graph: AdjListGraph<B::Count, T>,
        diagonal: Vec<T>,
        ground: Option<u32>,
        seed: u64,
        make_star: &impl Fn(usize) -> B,
        degree_scale: usize,
    ) -> BlockFactor<T> {
        let mut sampler = CdfSampler::<T>::new(seed);
        let star_builder = make_star(graph.n());
        self.build_from_graph(
            graph,
            diagonal,
            ground,
            &mut sampler,
            star_builder,
            degree_scale,
        )
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

    /// Algorithm 8 loop on a pre-built graph.
    ///
    /// The graph's multiplicity storage is the star builder's `Count`, so an AC
    /// builder over a split multi-edge graph does not compile — a pairing
    /// nothing checked while both were re-derived here from `Config`.
    fn build_from_graph<B: StarBuilderVariant<T>>(
        &self,
        mut graph: AdjListGraph<B::Count, T>,
        mut diag: Vec<T>,
        ground: Option<u32>,
        sampler: &mut CdfSampler<T>,
        mut star_builder: B,
        degree_scale: usize,
    ) -> BlockFactor<T> {
        let n = graph.n();
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
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
            seq.record_column(v, column.diagonal, column.neighbors(), column.fractions());

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

fn component_seed(seed: u64, representative: u64) -> u64 {
    let mut value = seed ^ representative.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests;
