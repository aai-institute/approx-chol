use super::decomposition::EliminationSequence;
use crate::graph::{AdjListGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::{DegreeDeltas, DynamicOrdering};
use crate::sampling::CdfSampler;
use crate::{CsrError, CsrRef, Error, Factor};
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
        let (n, sequence) = match self.config.split_merge {
            0 => {
                let GraphBuild {
                    graph,
                    diagonal: diag,
                    ..
                } = SlimGraph::<T>::from_sddm(sddm)?;
                let n = graph.n();
                let star = AcStarBuilder::new(n);
                (n, self.build_from_graph(graph, diag, star, 1))
            }
            k => {
                let GraphBuild {
                    mut graph,
                    diagonal: diag,
                    ..
                } = MultiEdgeGraph::<T>::from_sddm(sddm)?;
                graph.mark_split_edges(k);
                let n = graph.n();
                let star = Ac2StarBuilder::new(n, k);
                (n, self.build_from_graph(graph, diag, star, k as usize))
            }
        };
        Ok(Factor {
            n,
            original_n,
            sequence,
        })
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
        mut star_builder: B,
        degree_scale: usize,
    ) -> EliminationSequence<T> {
        let n = graph.n();
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
        let mut ordering = DynamicOrdering::new(&degrees, degree_scale);
        let mut sampler = CdfSampler::<T>::new(self.config.seed);
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

            star_builder.sample_column(diag[v], &mut sampler, &mut column);
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

        seq
    }
}

#[cfg(test)]
mod tests;
