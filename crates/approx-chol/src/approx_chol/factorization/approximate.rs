//! Randomized elimination (Algorithm 8). The machinery is this backend's alone, so it
//! lives here rather than beside the graph the exact backend also reads.

mod clique_tree;
mod ordering;
mod star;

pub use clique_tree::CliqueTreeSampler;

#[cfg(any(feature = "serde", test))]
use super::FactorError;
use clique_tree::{sample_column, SampledColumn};
use ordering::{DegreeDeltas, DynamicOrdering};
use star::StarBuilder;

use crate::graph::{AdjListGraph, EdgeCount};
use crate::sampling::CdfSampler;
use crate::types::Real;

/// `C::Split` ties the split to the graph's multiplicity storage, so an AC
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

        // One pq_move per affected neighbor, not one per incident event. Batching
        // reorders equal-degree vertices, so a fixed seed's factor differs from a
        // per-edge version's (quality unaffected; see CHANGELOG).
        column.apply_fill_in_delta(&mut graph, &mut diag, &mut deltas);
        star.accumulate_removal_delta(&mut deltas);
        deltas.flush(&mut ordering);
    }

    seq
}

/// Zero-copy view of one elimination step.
pub(super) struct EliminationStep<'a, T> {
    vertex: usize,
    inv_diag: T,
    neighbor_indices: &'a [u32],
    elimination_fractions: &'a [T],
}

/// Neither kernel bounds-checks per step: the caller asserts `y.len() >= n` once per
/// solve, and every index is under `n` by construction or by
/// [`EliminationSequence::validate_for_dim`].
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

/// The neighbor range *starts* at the previous header's `end`, so no second array can
/// disagree about step count or about which diagonal belongs to which vertex.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepHeader<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    pub(crate) inv_diag: T,
}

/// The solve path keeps flat split arrays; a persisted sequence nests neighbors under
/// their step, so the `end` offsets are rebuilt on load rather than trusted.
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

/// Nesting the neighbors under their step is what retires the range and
/// trailing-storage checks.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct StepData<T> {
    vertex: u32,
    inv_diag: T,
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

/// Mirrors [`StepData`] without materializing one, so serializing allocates nothing
/// regardless of `nnz`.
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

    /// The ranges need no check — they are rebuilt from the nested persisted form,
    /// never read off the wire.
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

    /// Overflowing the `u32` range end is unreachable for tractable inputs, so assert
    /// in release too rather than truncate silently.
    fn push_step(&mut self, vertex: usize, diagonal: T) {
        let nnz = self.neighbor_indices.len();
        assert!(
            nnz <= u32::MAX as usize,
            "factor nonzero count {nnz} exceeds u32 range capacity"
        );
        self.steps.push(StepHeader {
            vertex: vertex as u32,
            end: nnz as u32,
            // A merely small pivot inverts fine; standing `one` in for it would drop
            // the block's scale outright rather than lose accuracy.
            inv_diag: match T::one() / diagonal {
                inverse if inverse.is_finite() => inverse,
                _ => T::one(),
            },
        });
    }

    fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.push_step(vertex, diagonal);
    }

    /// Takes the column, not its parts: [`SampledColumn`] is what keeps a neighbor
    /// array from being stored against a fraction array of another length.
    fn record_column(&mut self, vertex: usize, column: &SampledColumn<T>) {
        let (neighbors, fractions) = column.pattern();
        self.neighbor_indices.extend_from_slice(neighbors);
        self.elimination_fractions.extend_from_slice(fractions);
        self.push_step(vertex, column.diagonal);
    }
}
