//! Approximate Cholesky of one block by randomized elimination (Algorithm 8),
//! the sequence it produces, and the per-step row kernels that solve with it.
//!
//! The randomized machinery is this backend's alone — [`ordering`] picks the next
//! vertex, [`star`] assembles its neighborhood, [`clique_tree`] samples the column
//! — so it lives here rather than beside the graph the exact backend also reads.

mod clique_tree;
mod ordering;
mod star;

pub use clique_tree::clique_tree_sample;

#[cfg(any(feature = "serde", test))]
use super::FactorError;
use clique_tree::{sample_column, SampledColumn};
use ordering::{DegreeDeltas, DynamicOrdering};
use star::StarBuilder;

use crate::graph::{AdjListGraph, EdgeCount};
use crate::sampling::CdfSampler;
use crate::types::Real;

/// The graph's multiplicity storage decides the split it is eliminated on, so an AC
/// factorization over a split multi-edge graph does not compile.
pub(crate) fn eliminate<T: Real, C: EdgeCount>(
    mut graph: AdjListGraph<C, T>,
    mut diag: Vec<T>,
    sampler: &mut CdfSampler<T>,
    split: C::Split,
) -> EliminationSequence<T> {
    let n = graph.n();
    let copies = C::split_edges(&mut graph, split);
    let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
    let degree_sum: usize = degrees.iter().sum();
    let mut star_builder = StarBuilder::<T, C>::new(n, copies);
    let mut ordering = DynamicOrdering::new(&degrees, copies as usize);
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
        let star = star_builder.star();
        if star.entries().is_empty() {
            seq.record_isolated(v, diag[v]);
            graph.eliminate_vertex(v);
            continue;
        }

        sample_column(star, diag[v], sampler, &mut column);
        seq.record_column(v, &column);

        graph.eliminate_vertex(v);
        for entry in star.entries() {
            let u = entry.neighbor as usize;
            diag[u] = diag[u] - entry.weight;
        }

        // Batch this step's degree-estimate updates into one pq_move per
        // affected neighbor (net delta) instead of one per incident
        // fill/removal/merge event. Batching reorders equal-degree vertices
        // within a bucket, so the exact factor for a fixed seed can differ
        // from a per-edge version (quality unaffected; see CHANGELOG).
        column.apply_fill_in_delta(&mut graph, &mut diag, &mut deltas);
        star.accumulate_removal_delta(&mut deltas);
        deltas.flush(&mut ordering);
    }

    seq
}

/// Zero-copy view of one elimination step: it eliminates `vertex` by splitting
/// its weight among neighbors according to `elimination_fractions`.
pub(super) struct EliminationStep<'a, T> {
    vertex: usize,
    inv_diag: T,
    neighbor_indices: &'a [u32],
    elimination_fractions: &'a [T],
}

/// Every index a kernel below touches is in bounds already: the caller asserts
/// `y.len() >= n` once per solve, and every vertex and neighbor is under `n` —
/// by construction from the builder, by [`EliminationSequence::validate_for_dim`]
/// from serde.
/// Neither kernel re-checks per step.
impl<'a, T: Real> EliminationStep<'a, T> {
    /// Forward elimination: scatter pivot weight to neighbors, then scale by D^{-1}.
    #[inline(always)]
    pub(super) fn apply_forward(&self, y: &mut [T]) {
        let vertex = self.vertex;
        let inv_diag = self.inv_diag;
        let n = self.neighbor_indices.len();
        let one = T::one();
        if n == 0 {
            y[vertex] = y[vertex] * inv_diag;
            return;
        }

        let mut yi = y[vertex];

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
        {
            let j = j as usize;
            y[j] = y[j] + f * yi;
            yi = yi * (one - f);
        }

        let j_last = self.neighbor_indices[n - 1] as usize;
        y[j_last] = y[j_last] + yi;
        y[vertex] = yi * inv_diag;
    }

    /// Backward substitution: gather neighbor contributions back to pivot.
    #[inline(always)]
    pub(super) fn apply_backward(&self, y: &mut [T]) {
        let vertex = self.vertex;
        let n = self.neighbor_indices.len();
        let one = T::one();
        if n == 0 {
            return;
        }

        let j_last = self.neighbor_indices[n - 1] as usize;
        let mut yi = y[vertex] + y[j_last];

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
            .rev()
        {
            yi = (one - f) * yi + f * y[j as usize];
        }

        y[vertex] = yi;
    }
}

/// Header for one elimination step: which vertex, the factor its pivot is scaled
/// by, and where its neighbor range ends. The range *starts* at the previous header's
/// `end`, so there is no second array that could disagree about step count,
/// about where step 0 begins, or about which diagonal belongs to which vertex.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepHeader<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    pub(crate) inv_diag: T,
}

/// Contiguous memory owner for a sequence of elimination steps.
///
/// A persisted sequence is a list of [`StepData`], each owning its own neighbors,
/// so the cumulative `end` offsets are *rebuilt* on load rather than trusted:
/// contiguous, non-decreasing and exhaustive by construction. The solve path
/// keeps the flat split arrays.
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned"),
        try_from = "Vec<StepData<T>>"
    )
)]
#[derive(Clone, Debug)]
pub(crate) struct EliminationSequence<T> {
    pub(crate) steps: Vec<StepHeader<T>>,
    pub(crate) neighbor_indices: Vec<u32>,
    pub(crate) elimination_fractions: Vec<T>,
}

/// Persisted shape of one step. Nesting the neighbors under the step they belong
/// to is what retires the range and trailing-storage checks.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct StepData<T> {
    vertex: u32,
    inv_diag: T,
    /// One `(neighbor, elimination fraction)` per factor nonzero in this step.
    neighbors: Vec<(u32, T)>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<Vec<StepData<T>>> for EliminationSequence<T> {
    type Error = FactorError;

    fn try_from(data: Vec<StepData<T>>) -> Result<Self, Self::Error> {
        let mut steps = Vec::with_capacity(data.len());
        let mut neighbor_indices = Vec::new();
        let mut elimination_fractions = Vec::new();
        for step in data {
            for (neighbor, fraction) in step.neighbors {
                neighbor_indices.push(neighbor);
                elimination_fractions.push(fraction);
            }
            let nnz = neighbor_indices.len();
            // The only range invariant the nesting cannot carry: `end` is a `u32`.
            let end =
                u32::try_from(nnz).map_err(|_| FactorError::NonzeroCountExceedsU32 { nnz })?;
            steps.push(StepHeader {
                vertex: step.vertex,
                end,
                inv_diag: step.inv_diag,
            });
        }
        Ok(Self {
            steps,
            neighbor_indices,
            elimination_fractions,
        })
    }
}

/// Mirrors [`StepData`] without materializing one: each step borrows its slice of
/// the flat arrays, so serializing allocates nothing regardless of `nnz`.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for EliminationSequence<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((0..self.steps.len()).map(|i| StepView(self, i)))
    }
}

#[cfg(feature = "serde")]
struct StepView<'a, T>(&'a EliminationSequence<T>, usize);

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for StepView<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let (sequence, i) = (self.0, self.1);
        let (start, end) = sequence.neighbor_range(i);
        let mut out = serializer.serialize_struct("StepData", 3)?;
        out.serialize_field("vertex", &sequence.steps[i].vertex)?;
        out.serialize_field("inv_diag", &sequence.steps[i].inv_diag)?;
        out.serialize_field(
            "neighbors",
            &PairedNeighbors(
                &sequence.neighbor_indices[start..end],
                &sequence.elimination_fractions[start..end],
            ),
        )?;
        out.end()
    }
}

#[cfg(feature = "serde")]
struct PairedNeighbors<'a, T>(&'a [u32], &'a [T]);

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for PairedNeighbors<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().zip(self.1))
    }
}

// Read-only accessors (no internal trait bounds).
impl<T> EliminationSequence<T> {
    #[inline(always)]
    pub(super) fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Half-open range of step `i` in the flat neighbor arrays.
    #[inline(always)]
    fn neighbor_range(&self, i: usize) -> (usize, usize) {
        let start = if i == 0 {
            0
        } else {
            self.steps[i - 1].end as usize
        };
        (start, self.steps[i].end as usize)
    }

    #[inline(always)]
    pub(super) fn step(&self, i: usize) -> EliminationStep<'_, T>
    where
        T: Copy,
    {
        let (start, end) = self.neighbor_range(i);
        EliminationStep {
            vertex: self.steps[i].vertex as usize,
            inv_diag: self.steps[i].inv_diag,
            neighbor_indices: &self.neighbor_indices[start..end],
            elimination_fractions: &self.elimination_fractions[start..end],
        }
    }

    /// Reject vertices and neighbors that would index outside a factor of
    /// dimension `n`. The ranges themselves need no check — they are rebuilt from
    /// the nested persisted form, never read off the wire.
    #[cfg(any(feature = "serde", test))]
    pub(super) fn validate_for_dim(&self, n: usize) -> Result<(), FactorError>
    where
        T: num_traits::Float,
    {
        for (i, step) in self.steps.iter().enumerate() {
            if (step.vertex as usize) >= n {
                return Err(FactorError::VertexOutOfBounds {
                    step: i,
                    vertex: step.vertex,
                    n,
                });
            }
            let (start, end) = self.neighbor_range(i);
            for &j in &self.neighbor_indices[start..end] {
                if (j as usize) >= n {
                    return Err(FactorError::NeighborOutOfBounds {
                        step: i,
                        neighbor: j,
                        n,
                    });
                }
            }
            // A fraction splits one column's weight, so it is a proportion; an
            // `inv_diag` is a reciprocal the solve multiplies by.
            let fractions = &self.elimination_fractions[start..end];
            if !step.inv_diag.is_finite()
                || fractions
                    .iter()
                    .any(|f| !f.is_finite() || *f < T::zero() || *f > T::one())
            {
                return Err(FactorError::StepValueInvalid { step: i });
            }
        }
        Ok(())
    }
}

impl<T: Real> EliminationSequence<T> {
    fn with_capacity(n: usize, degree_sum: usize) -> Self {
        Self {
            steps: Vec::with_capacity(n),
            neighbor_indices: Vec::with_capacity(degree_sum),
            elimination_fractions: Vec::with_capacity(degree_sum),
        }
    }

    /// Close the current step at the running nonzero count. Overflow of the `u32`
    /// range end is unreachable for tractable inputs, so assert (in release too)
    /// rather than truncate silently.
    fn push_step(&mut self, vertex: usize, diagonal: T) {
        let nnz = self.neighbor_indices.len();
        assert!(
            nnz <= u32::MAX as usize,
            "factor nonzero count {nnz} exceeds u32 range capacity"
        );
        self.steps.push(StepHeader {
            vertex: vertex as u32,
            end: nnz as u32,
            // Only a reciprocal too large to represent is unusable. A merely small
            // pivot inverts fine, and standing one in for `one` drops the block's
            // scale outright rather than losing accuracy.
            inv_diag: match T::one() / diagonal {
                inverse if inverse.is_finite() => inverse,
                _ => T::one(),
            },
        });
    }

    /// Record an isolated vertex (no neighbors, clamped diagonal).
    fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.push_step(vertex, diagonal);
    }

    /// Record one sampled column. Taking the column itself rather than its three
    /// parts is what keeps a neighbor array from being stored against a fraction
    /// array of another length — the pairing [`SampledColumn`] maintains.
    fn record_column(&mut self, vertex: usize, column: &SampledColumn<T>) {
        let (neighbors, fractions) = column.pattern();
        self.neighbor_indices.extend_from_slice(neighbors);
        self.elimination_fractions.extend_from_slice(fractions);
        self.push_step(vertex, column.diagonal);
    }
}
