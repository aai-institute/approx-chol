use super::decomposition::EliminationSequence;
use crate::graph::{EliminationGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::{DynamicOrdering, EliminationOrdering, StaticOrdering};
use crate::sampling::{CdfSampler, WeightedSampler};
use crate::{CsrRef, Error, Factor};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::clique_tree::{clique_tree_sample_column, clique_tree_sample_column_multi};
use super::sampled_column::SampledColumn;
use super::star::{Ac2StarBuilder, AcStarBuilder};
use super::{Config, Ordering};

trait StarBuilderVariant<T: num_traits::Float + Send + Sync + 'static> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    );
    fn is_empty(&self) -> bool;
    fn entries(&self) -> &[(u32, T)];
    fn counts(&self) -> Option<&[u32]>;
}

impl<T: num_traits::Float + Send + Sync + 'static> StarBuilderVariant<T> for AcStarBuilder<T> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        self.build_star(graph, v, ordering);
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        self.entries()
    }

    fn counts(&self) -> Option<&[u32]> {
        None
    }
}

impl<T: num_traits::Float + Send + Sync + 'static> StarBuilderVariant<T> for Ac2StarBuilder<T> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        self.build_star(graph, v, ordering);
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        self.entries()
    }

    fn counts(&self) -> Option<&[u32]> {
        Some(self.counts())
    }
}

fn sample_star_column<T, W>(
    entries: &[(u32, T)],
    counts: Option<&[u32]>,
    pivot_diag: T,
    sampler: &mut W,
    column: &mut SampledColumn<T>,
) where
    T: num_traits::Float + Send + Sync + 'static,
    W: WeightedSampler<T>,
{
    match counts {
        None => clique_tree_sample_column(entries, pivot_diag, sampler, column),
        Some(counts) => {
            clique_tree_sample_column_multi(entries, counts, pivot_diag, sampler, column)
        }
    }
}

fn notify_star_eliminated<T, O>(
    ordering: &mut O,
    v: usize,
    entries: &[(u32, T)],
    counts: Option<&[u32]>,
) where
    T: num_traits::Float + Send + Sync + 'static,
    O: EliminationOrdering<T>,
{
    match counts {
        None => ordering.notify_eliminated(v, entries),
        Some(counts) => {
            debug_assert_eq!(entries.len(), counts.len());
            for (&(u, _), &count) in entries.iter().zip(counts.iter()) {
                ordering.notify_neighbor_removed_n(u, count);
            }
        }
    }
}

/// Builder for approximate Cholesky factorization (Algorithm 8, Gao-Kyng-Spielman 2023).
///
/// Provides full control over the factorization pipeline, including
/// elimination ordering. Most users should prefer [`factorize`](crate::factorize)
/// or [`factorize_with`](crate::factorize_with).
///
/// # Examples
///
/// ```
/// use approx_chol::{Config, CsrRef};
/// use approx_chol::low_level::{Builder, Ordering};
///
/// let row_ptrs    = [0u32, 2, 5, 8, 10];
/// let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
///
/// let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
/// let factor = Builder::new(Config::default())
///     .ordering(Ordering::StaticAMD)
///     .build(csr)?;
/// assert_eq!(factor.n(), 4);
/// # Ok::<(), approx_chol::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Builder<T = f64> {
    config: Config,
    ordering: Ordering,
    _scalar: core::marker::PhantomData<T>,
}

impl<T> Builder<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    /// Create a new builder with the given configuration.
    ///
    /// The elimination ordering defaults to [`Ordering::DynamicPQ`].
    /// Use [`ordering`](Self::ordering) to override.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            config,
            ordering: Ordering::DynamicPQ,
            _scalar: core::marker::PhantomData,
        }
    }

    /// Set the elimination ordering strategy.
    #[must_use]
    pub fn ordering(mut self, ordering: Ordering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Run approximate Cholesky factorization on a CSR SDDM matrix using `u32` indices.
    pub fn build(&self, sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error> {
        self.build_with_sampler(sddm, CdfSampler::<T>::new(self.config.seed))
    }

    /// Run approximate Cholesky factorization on any index type convertible to `u32`.
    ///
    /// Uses a zero-copy fast path when `I = u32`; otherwise performs checked
    /// index conversion to owned `u32` storage.
    pub fn build_generic<I: PrimInt + 'static>(
        &self,
        sddm: CsrRef<'_, T, I>,
    ) -> Result<Factor<T>, Error> {
        let converted = sddm.to_u32_fast_or_owned()?;
        self.build(converted.as_ref())
    }

    /// Run approximate Cholesky factorization from any input convertible into
    /// [`CsrRef`].
    ///
    /// This preserves the zero-copy path for `u32`-indexed inputs.
    /// For panic-free matrix-adapter paths, prefer fallible conversion methods
    /// on [`CsrRef`] and call [`Self::build_generic`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if conversion panics, CSR
    /// validation fails, or index conversion to `u32` fails.
    /// Returns [`Error::InvalidConfig`] for invalid `split_merge`.
    pub fn build_from<'a, I: PrimInt + 'a + 'static, M: Into<CsrRef<'a, T, I>>>(
        &self,
        sddm: M,
    ) -> Result<Factor<T>, Error> {
        let csr = catch_unwind(AssertUnwindSafe(|| sddm.into()))
            .map_err(|_| Error::InvalidCsr("input conversion panicked"))?;
        self.build_generic(csr)
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
            Some(sm) => {
                let GraphBuild {
                    mut graph,
                    diagonal: diag,
                    ..
                } = MultiEdgeGraph::<T>::from_sddm(sddm);
                graph.mark_split_edges(sm.split);
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
        if split_merge.split == 0 {
            return Err(Error::InvalidConfig("split_merge.split must be >= 1"));
        }
        if split_merge.merge == 0 {
            return Err(Error::InvalidConfig("split_merge.merge must be >= 1"));
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
        match self.ordering {
            Ordering::StaticAMD => {
                let (mut ordering, degree_sum) = StaticOrdering::from_graph::<T, _>(&mut graph);
                self.factorize_with_ordering(&mut graph, diag, &mut ordering, degree_sum, sampler)
            }
            Ordering::DynamicPQ => {
                let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
                let degree_sum: usize = degrees.iter().sum();
                let degree_scale = self
                    .config
                    .split_merge
                    .map_or(1usize, |sm| sm.split as usize);
                let mut ordering =
                    DynamicOrdering::new_with_scale(n, degrees.into_iter(), degree_scale);
                self.factorize_with_ordering(&mut graph, diag, &mut ordering, degree_sum, sampler)
            }
        }
    }

    /// Dispatch on the clique-tree sampling variant (AC vs AC2).
    fn factorize_with_ordering<
        G: EliminationGraph<T>,
        S: WeightedSampler<T>,
        O: EliminationOrdering<T>,
    >(
        &self,
        graph: &mut G,
        diag: Vec<T>,
        ordering: &mut O,
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
            Some(sm) => Self::factorize_with_variant(
                graph,
                &mut diag,
                ordering,
                degree_sum,
                sampler,
                Ac2StarBuilder::new(graph.n(), sm.merge),
            ),
        }
    }

    /// Algorithm 8 loop parameterized by a clique-tree sampling variant.
    fn factorize_with_variant<
        G: EliminationGraph<T>,
        W: WeightedSampler<T>,
        O: EliminationOrdering<T>,
        B: StarBuilderVariant<T>,
    >(
        graph: &mut G,
        diag: &mut [T],
        ordering: &mut O,
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
            let star_counts = star_builder.counts();
            sample_star_column(
                star_entries,
                star_counts,
                diag[v],
                &mut sampler,
                &mut column,
            );
            seq.record_column(v, &column);

            graph.eliminate_vertex(v);
            for &(u, w) in star_entries {
                diag[u as usize] = diag[u as usize] - w;
            }

            column.apply_fill_in(graph, diag, ordering);
            notify_star_eliminated(ordering, v, star_entries, star_counts);
        }

        Factor { n, sequence: seq }
    }
}

#[cfg(test)]
mod tests;
