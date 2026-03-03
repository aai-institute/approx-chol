use super::decomposition::EliminationSequence;
use crate::graph::{EliminationGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::DynamicOrdering;
use crate::sampling::{CdfSampler, WeightedSampler};
use crate::{CsrRef, Error, Factor};
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

    /// Run approximate Cholesky factorization from any input convertible into
    /// [`CsrRef`].
    ///
    /// Performs a checked conversion of row pointers and column indices to
    /// owned `u32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if conversion panics, CSR validation
    /// fails, or index conversion to `u32` fails.
    /// Returns [`Error::InvalidConfig`] for invalid `split_merge`.
    pub fn build<'a, I: PrimInt + 'a + 'static, M: Into<CsrRef<'a, T, I>>>(
        &self,
        sddm: M,
    ) -> Result<Factor<T>, Error> {
        let csr = catch_unwind(AssertUnwindSafe(|| sddm.into()))
            .map_err(|_| Error::InvalidCsr("input conversion panicked"))?;
        let converted = csr.to_owned_u32()?;
        self.build_u32(converted.as_ref())
    }

    fn build_u32(&self, sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error> {
        self.build_with_sampler(sddm, CdfSampler::<T>::new(self.config.seed))
    }

    /// Run approximate Cholesky factorization with a custom [`WeightedSampler`].
    pub(crate) fn build_with_sampler<S: WeightedSampler<T>>(
        &self,
        sddm: CsrRef<'_, T, u32>,
        sampler: S,
    ) -> Result<Factor<T>, Error> {
        Self::validate_config(self.config)?;
        Self::validate(&sddm)?;
        match self.config.split_merge {
            None => {
                let GraphBuild {
                    graph,
                    diagonal: diag,
                    ..
                } = SlimGraph::<T>::from_sddm(sddm);
                Ok(self.build_from_graph(graph, diag, sampler))
            }
            Some(k) => {
                let GraphBuild {
                    mut graph,
                    diagonal: diag,
                    ..
                } = MultiEdgeGraph::<T>::from_sddm(sddm);
                graph.mark_split_edges(k);
                Ok(self.build_from_graph(graph, diag, sampler))
            }
        }
    }

    fn validate(csr: &CsrRef<'_, T, u32>) -> Result<(), Error> {
        csr.validate()
    }

    fn validate_config(config: Config) -> Result<(), Error> {
        let Some(split_merge) = config.split_merge else {
            return Ok(());
        };
        if split_merge == 0 {
            return Err(Error::InvalidConfig("split_merge must be >= 1"));
        }
        Ok(())
    }

    /// Run factorization on a pre-built graph (fused pipeline path).
    pub(crate) fn build_from_graph<G: EliminationGraph<T>, S: WeightedSampler<T>>(
        &self,
        mut graph: G,
        diag: Vec<T>,
        sampler: S,
    ) -> Factor<T> {
        let n = graph.n();
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
        let mut ordering = match self.config.split_merge {
            None => DynamicOrdering::new(n, degrees.into_iter()),
            Some(k) => DynamicOrdering::new_with_scale(n, degrees.into_iter(), k as usize),
        };
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
    ) -> Factor<T> {
        let mut diag = diag;
        match self.config.split_merge {
            None => Self::factorize_with_variant(
                graph,
                &mut diag,
                ordering,
                degree_sum,
                sampler,
                AcStarBuilder::new(graph.n()),
            ),
            Some(k) => Self::factorize_with_variant(
                graph,
                &mut diag,
                ordering,
                degree_sum,
                sampler,
                Ac2StarBuilder::new(graph.n(), k),
            ),
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
            seq.record_column(v, &column);

            graph.eliminate_vertex(v);
            for &(u, w) in star_entries {
                diag[u as usize] = diag[u as usize] - w;
            }

            column.apply_fill_in(graph, diag, ordering);
            star_builder.notify_eliminated(ordering, v);
        }

        Factor { n, sequence: seq }
    }
}

#[cfg(test)]
mod tests;
