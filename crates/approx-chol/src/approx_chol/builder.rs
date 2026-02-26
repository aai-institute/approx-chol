use super::decomposition::EliminationSequence;
use crate::graph::{EliminationGraph, GraphBuild, MultiEdgeGraph, SlimGraph};
use crate::ordering::{DynamicOrdering, EliminationOrdering, StaticOrdering};
use crate::sampling::{CdfSampler, WeightedSampler};
use crate::types::Real;
use crate::{Error, Factor, CsrRef};
use num_traits::PrimInt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::sampled_column::SampledColumn;
use super::star_ac::AcStar;
use super::star_ac2::Ac2Star;
use super::{Config, Ordering, Star};

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
    pub fn build(
        &self,
        sddm: CsrRef<'_, T, u32>,
    ) -> Result<Factor<T>, Error> {
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
            return Err(Error::InvalidConfig(
                "split_merge.split must be >= 1",
            ));
        }
        if split_merge.merge == 0 {
            return Err(Error::InvalidConfig(
                "split_merge.merge must be >= 1",
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

    /// Dispatch on star type (AC vs AC2), then run the generic factorization loop.
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
            None => {
                let star = AcStar::<S, T>::new(graph.n(), sampler);
                Self::factorize(graph, &mut diag, ordering, degree_sum, star)
            }
            Some(sm) => {
                let star = Ac2Star::<S, T>::new(graph.n(), sm.merge, sampler);
                Self::factorize(graph, &mut diag, ordering, degree_sum, star)
            }
        }
    }

    /// Unified factorization loop for both AC and AC2.
    ///
    /// Eliminates up to `n - 1` vertices (`target_steps`); the last vertex is never
    /// eliminated because it would produce an empty star with no neighbors.
    /// The loop may also break early if the ordering is exhausted before reaching the target.
    /// `degree_sum` is the total degree of the initial graph, used as a capacity hint
    /// for the `EliminationSequence` allocation.
    fn factorize<G: EliminationGraph<T>, S: Star<T>, O: EliminationOrdering<T>>(
        graph: &mut G,
        diag: &mut [T],
        ordering: &mut O,
        degree_sum: usize,
        mut star: S,
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
            eliminate_vertex(v, graph, diag, ordering, &mut star, &mut column, &mut seq);
        }

        Factor { n, sequence: seq }
    }
}

/// Per-vertex elimination step: compress, sample, eliminate, fill-in.
fn eliminate_vertex<T: Real, G: EliminationGraph<T>, S: Star<T>, O: EliminationOrdering<T>>(
    v: usize,
    graph: &mut G,
    diag: &mut [T],
    ordering: &mut O,
    star: &mut S,
    column: &mut SampledColumn<T>,
    seq: &mut EliminationSequence<T>,
) {
    if graph.is_empty(v) {
        seq.record_isolated(v, diag[v]);
        return;
    }

    // Phase 1: Compress — gather and dedup live neighbors of v
    star.compress(graph, v, ordering);
    if star.is_empty() {
        seq.record_isolated(v, diag[v]);
        graph.eliminate_vertex(v);
        return;
    }

    // Phase 2: Sample — draw one column of the approximate factor
    star.sample(diag[v], column);
    seq.record_column(v, column);

    // Phase 3: Eliminate — remove v and subtract its edge weights from neighbor diags
    let star_entries = star.entries();
    graph.eliminate_vertex(v);
    for &(u, w) in star_entries {
        diag[u as usize] = diag[u as usize] - w;
    }

    // Phase 4: Fill-in — insert sampled fill edges, apply Schur complement diag updates
    column.apply_fill_in(graph, diag, ordering);
    star.notify_eliminated(ordering, v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMerge;

    /// Build a 4-node path graph Laplacian as raw CSR arrays.
    fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let indptr = vec![0u32, 2, 5, 8, 10];
        let indices = vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let data = vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
        (indptr, indices, data)
    }

    fn make_csr<'a>(
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
    ) -> CsrRef<'a, f64, u32> {
        CsrRef::new_unchecked(indptr, indices, data, (indptr.len() - 1) as u32)
    }

    #[test]
    fn test_ac_default_solve_roundtrip() {
        let (indptr, indices, data) = path_laplacian_4();
        let csr = make_csr(&indptr, &indices, &data);

        let builder = Builder::<f64>::new(Config::default());
        let factor = builder.build(csr).unwrap();
        assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));

        let b = [1.0, -1.0, 1.0, -1.0];
        let mut work = vec![0.0; factor.n()];
        factor.solve_into(&b, &mut work);
        assert!(work.iter().all(|x| x.is_finite()));
        assert!(work.iter().any(|x| x.abs() > 1e-10));
        let mean = work.iter().sum::<f64>() / work.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    /// Regression test for the AC2 n==1 diagonal bug.
    ///
    /// Constructs a 3-node augmented SDDM matrix where every vertex has
    /// exactly one off-diagonal neighbor but the diagonal is larger than the
    /// neighbor weight (due to augmentation). The old code incorrectly set
    /// `column.diagonal = entries[0].1` (the neighbor weight) instead of
    /// `pivot_diag` (the actual matrix diagonal), losing the augmentation
    /// mass and producing an incorrect factorization.
    ///
    /// Matrix (3 nodes, path graph 0–1–2, augmented diagonal):
    ///   A = [ 5.0  -1.0   0.0 ]
    ///       [-1.0   6.0  -1.0 ]
    ///       [ 0.0  -1.0   5.0 ]
    ///
    /// Every node has at most 2 neighbors, but nodes 0 and 2 have only 1
    /// neighbor each, and their diagonal (5.0) is much larger than their
    /// edge weight (1.0), making this a clear augmented case.
    #[test]
    fn test_ac2_n_eq_1_augmented_diagonal_regression() {
        // 3-node path graph 0-1-2 with diagonal augmentation.
        // Node 0: diag=5.0, edge to 1 with weight 1.0
        // Node 1: diag=6.0, edges to 0 and 2 with weight 1.0 each
        // Node 2: diag=5.0, edge to 1 with weight 1.0
        let indptr = vec![0u32, 2, 5, 7];
        let indices = vec![0u32, 1, 0, 1, 2, 1, 2];
        let data = vec![5.0f64, -1.0, -1.0, 6.0, -1.0, -1.0, 5.0];

        let csr = CsrRef::new(&indptr, &indices, &data, 3).expect("valid SDDM matrix");

        let config = Config {
            split_merge: Some(SplitMerge { split: 2, merge: 2 }),
            ..Default::default()
        };
        let builder = Builder::<f64>::new(config);

        // Should complete without panic (old code would produce NaN/Inf for
        // vertices with n==1 and pivot_diag > neighbor_weight).
        let factor = builder.build(csr).expect("AC2 factorization must succeed");
        // The factor may be larger than 3 due to Gremban augmentation (the
        // matrix is SDDM but not Laplacian, so an auxiliary vertex is added).
        assert!(
            factor.n() >= 3,
            "factor dimension should be at least the original matrix size"
        );

        // Solve using the factorization as a preconditioner application.
        // The RHS covers the original 3 nodes; the factorization may operate
        // on an augmented system (4 nodes due to Gremban's reduction).
        let b = [4.0f64, 4.0, 4.0];
        let mut work = vec![0.0f64; factor.n()];
        factor.solve_into_with_projection(&b, &mut work, false);

        // All entries must be finite — the old bug set column.diagonal to the
        // small edge weight (1.0) instead of pivot_diag (5.0), making the
        // Schur complement update degenerate and producing NaN/Inf.
        assert!(
            work.iter().all(|x| x.is_finite()),
            "AC2 solve produced non-finite output with n==1 augmented diagonal: {:?}",
            work
        );

        // The output must be non-trivially non-zero (the system has a unique
        // solution since A is strictly diagonally dominant).
        assert!(
            work.iter().any(|x| x.abs() > 1e-10),
            "AC2 solve produced trivially zero output: {:?}",
            work
        );
    }

    /// Additional regression: verify AC2 handles n==1 with split=2 edge
    /// replication, ensuring the augmentation mass is never lost across
    /// multiple seeds.
    #[test]
    fn test_ac2_n_eq_1_solve_produces_finite_for_multiple_seeds() {
        // Minimal 2-node SDDM: each node has exactly 1 neighbor,
        // diagonal (10.0) >> edge weight (1.0) — strong augmentation.
        //   A = [10.0  -1.0]
        //       [-1.0  10.0]
        let indptr = vec![0u32, 2, 4];
        let indices = vec![0u32, 1, 0, 1];
        let data = vec![10.0f64, -1.0, -1.0, 10.0];
        let b = [9.0f64, -9.0];

        for seed in 0..8u64 {
            let csr = CsrRef::new(&indptr, &indices, &data, 2).expect("valid SDDM");
            let config = Config {
                split_merge: Some(SplitMerge { split: 2, merge: 2 }),
                seed,
                ..Default::default()
            };
            let factor = Builder::<f64>::new(config)
                .build(csr)
                .unwrap_or_else(|e| panic!("AC2 factorization failed (seed={seed}): {e}"));

            let mut work = vec![0.0f64; factor.n()];
            factor.solve_into_with_projection(&b, &mut work, false);

            assert!(
                work.iter().all(|x| x.is_finite()),
                "AC2 produced non-finite output for seed={seed}: {:?}",
                work
            );
        }
    }

    /// Regression test: AC2 handles near-zero total weight without division-by-zero.
    ///
    /// Constructs an SDDM matrix with extremely small edge weights (1e-300) but
    /// normal diagonal. The AC2 path encounters near-zero `total_weight` in the
    /// star neighborhood and must skip fill sampling gracefully.
    #[test]
    fn test_ac2_near_zero_weight_star() {
        // 3-node path graph with tiny edge weights and normal diagonal.
        //   A = [ 2.0    -1e-300   0.0     ]
        //       [-1e-300  2.0     -1e-300   ]
        //       [ 0.0    -1e-300   2.0      ]
        let eps = 1e-300_f64;
        let indptr = vec![0u32, 2, 5, 7];
        let indices = vec![0u32, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0, -eps, -eps, 2.0, -eps, -eps, 2.0];

        let csr = CsrRef::new(&indptr, &indices, &data, 3).expect("valid SDDM matrix");

        let config = Config {
            split_merge: Some(SplitMerge { split: 2, merge: 2 }),
            ..Default::default()
        };
        let factor = Builder::<f64>::new(config)
            .build(csr)
            .expect("AC2 factorization must succeed with near-zero weights");

        assert!(factor.n() >= 3);

        let b = [1.0f64, -1.0, 1.0];
        let mut work = vec![0.0f64; factor.n()];
        factor.solve_into_with_projection(&b, &mut work, false);

        assert!(
            work.iter().all(|x| x.is_finite()),
            "AC2 solve produced non-finite output with near-zero edge weights: {:?}",
            work
        );
    }
}
