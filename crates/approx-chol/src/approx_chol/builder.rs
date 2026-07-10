use super::decomposition::EliminationSequence;
use crate::graph::{EliminationGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::{DegreeDeltas, DynamicOrdering};
use crate::sampling::{CdfSampler, WeightedSampler};
use crate::{ConfigError, CsrError, CsrRef, Error, Factor};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::clique_tree::SampledColumn;
use super::star::{Ac2StarBuilder, AcStarBuilder, StarBuilderVariant};
use super::{Config, InputClass};
use crate::balance::certify_balance;

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
    /// Returns [`Error::InvalidConfig`] for invalid `split_merge`.
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
        let converted_ref = converted.try_as_ref()?;
        self.build_with_sampler(converted_ref, CdfSampler::<T>::new(self.config.seed))
    }

    /// Run approximate Cholesky factorization with a custom [`WeightedSampler`].
    pub(crate) fn build_with_sampler<S: WeightedSampler<T>>(
        &self,
        sddm: CsrRef<'_, T, u32>,
        sampler: S,
    ) -> Result<Factor<T>, Error> {
        let original_n = sddm.n();
        Self::validate_config(self.config)?;
        sddm.validate()?;

        let signs = self.folding_signature(sddm)?;
        let folded = signs.as_ref().map(|s| fold_values(sddm, s));
        let build_csr = match &folded {
            Some(values) => CsrRef::new(
                sddm.row_ptrs(),
                sddm.col_indices(),
                values,
                original_n as u32,
            )?,
            None => sddm,
        };

        let (mut factor, deficit) = match self.config.split_merge {
            None => {
                let GraphBuild {
                    graph,
                    diagonal: diag,
                    deficit,
                } = SlimGraph::<T>::from_sddm(build_csr)?;
                (self.build_from_graph(graph, diag, sampler)?, deficit)
            }
            Some(k) => {
                let GraphBuild {
                    mut graph,
                    diagonal: diag,
                    deficit,
                } = MultiEdgeGraph::<T>::from_sddm(build_csr)?;
                graph.mark_split_edges(k);
                (self.build_from_graph(graph, diag, sampler)?, deficit)
            }
        };
        factor.original_n = original_n;
        factor.deficit = deficit;
        if let Some(s) = signs {
            factor.congruence = Some(
                s.iter()
                    .map(|&sign| if sign >= 0 { T::one() } else { -T::one() })
                    .collect(),
            );
        }
        Ok(factor)
    }

    /// The ±1 fold signature, or `None` when folding would be the identity.
    fn folding_signature(&self, sddm: CsrRef<'_, T, u32>) -> Result<Option<Vec<i8>>, Error> {
        match self.config.assume {
            InputClass::Laplacian | InputClass::Sddm => Ok(None),
            InputClass::Auto | InputClass::Sdd | InputClass::HMatrix => {
                let signs = certify_balance(sddm)?.signs().to_vec();
                Ok(signs.iter().any(|&s| s < 0).then_some(signs))
            }
        }
    }

    fn validate_config(config: Config) -> Result<(), Error> {
        if let Some(split_merge) = config.split_merge {
            if split_merge == 0 {
                return Err(Error::InvalidConfig(
                    ConfigError::SplitMergeMustBePositive { split_merge },
                ));
            }
        }
        let scaling = config.scaling;
        if scaling.budget == 0 {
            return Err(Error::InvalidConfig(
                ConfigError::ScalingBudgetMustBePositive,
            ));
        }
        if !(scaling.slack.is_finite() && scaling.slack > 0.0) {
            return Err(Error::InvalidConfig(
                ConfigError::ScalingSlackMustBeFinitePositive,
            ));
        }
        Ok(())
    }

    /// Run factorization on a pre-built graph (fused pipeline path).
    pub(crate) fn build_from_graph<G: EliminationGraph<T>, S: WeightedSampler<T>>(
        &self,
        mut graph: G,
        diag: Vec<T>,
        sampler: S,
    ) -> Result<Factor<T>, Error> {
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
        sampler: S,
    ) -> Result<Factor<T>, Error> {
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
        mut sampler: W,
        mut star_builder: B,
    ) -> Factor<T> {
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
            star_builder.sample_column(diag[v], &mut sampler, &mut column);
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

        Factor {
            n,
            original_n: n,
            sequence: seq,
            congruence: None,
            deficit: None,
        }
    }
}

/// Fold each off-diagonal by `sᵢ·sⱼ`, flipping balanceable positives negative.
fn fold_values<T: num_traits::Float>(sddm: CsrRef<'_, T, u32>, signs: &[i8]) -> Vec<T> {
    let row_ptrs = sddm.row_ptrs();
    let col_indices = sddm.col_indices();
    let mut folded = sddm.values().to_vec();
    for row in 0..sddm.n() {
        for k in row_ptrs[row] as usize..row_ptrs[row + 1] as usize {
            if signs[row] != signs[col_indices[k] as usize] {
                folded[k] = -folded[k];
            }
        }
    }
    folded
}

#[cfg(test)]
mod tests;
