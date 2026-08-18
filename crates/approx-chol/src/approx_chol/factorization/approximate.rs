//! Randomized elimination (Algorithm 8). The machinery is this backend's alone, so it
//! lives here rather than beside the graph the exact backend also reads.

mod clique_tree;
mod ordering;
mod star;

pub use clique_tree::CliqueTreeSampler;

#[cfg(any(feature = "serde", test))]
use super::block::BlockDim;
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
    let mut star_builder = StarBuilder::<T, C>::new(n, split);
    let mut ordering = DynamicOrdering::new(&degrees, copies as usize);
    let mut column = SampledColumn::<T>::new();
    let mut seq = SequenceBuilder::with_capacity(n, degree_sum);
    let mut deltas = DegreeDeltas::new(n);
    for _ in 0..n.saturating_sub(1) {
        let v = ordering
            .next_vertex()
            .expect("the queue holds every vertex of the block");
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

    // One step short of `n`, so the queue still holds the vertex no step eliminated —
    // the block's last, which is what the anchor pins, only when it has no other.
    seq.finish(
        ordering
            .next_vertex()
            .expect("the queue holds every vertex of the block") as u32,
    )
}

/// Independent lanes for the backward gather; float addition is not ours to reassociate.
const LANES: usize = 4;

/// Zero-copy view of one elimination step.
struct EliminationStep<'a, T> {
    vertex: usize,
    /// `D^{-1}` already scaled by what the neighbors left, so no pass rebuilds it.
    pivot_scale: T,
    neighbor_indices: &'a [u32],
    coefficients: &'a [T],
}

/// Neither kernel bounds-checks per step: the caller asserts `y.len() >= n` once per
/// solve, and every index is under `n` by construction or by
/// [`EliminationSequence::validate_values`].
impl<'a, T: Real> EliminationStep<'a, T> {
    /// Forward elimination: scatter pivot weight to neighbors, then scale by D^{-1}.
    #[inline(always)]
    fn apply_forward(&self, y: &mut [T]) {
        let pivot = y[self.vertex];
        for (&j, &c) in self.neighbor_indices.iter().zip(self.coefficients.iter()) {
            let j = j as usize;
            y[j] = y[j] + c * pivot;
        }
        y[self.vertex] = pivot * self.pivot_scale;
    }

    /// Four accumulators, not one: a lone `total + c * y[j]` would reinstate the serial
    /// dependency the composed coefficients exist to break.
    #[inline(always)]
    fn apply_backward(&self, y: &mut [T]) {
        let Some((&retained, _)) = self.coefficients.split_last() else {
            return;
        };
        let mut lanes = [retained * y[self.vertex], T::zero(), T::zero(), T::zero()];
        let mut neighbors = self.neighbor_indices.chunks_exact(LANES);
        let mut coefficients = self.coefficients.chunks_exact(LANES);
        for (js, cs) in neighbors.by_ref().zip(coefficients.by_ref()) {
            lanes[0] = lanes[0] + cs[0] * y[js[0] as usize];
            lanes[1] = lanes[1] + cs[1] * y[js[1] as usize];
            lanes[2] = lanes[2] + cs[2] * y[js[2] as usize];
            lanes[3] = lanes[3] + cs[3] * y[js[3] as usize];
        }
        let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (&j, &c) in neighbors.remainder().iter().zip(coefficients.remainder()) {
            total = total + c * y[j as usize];
        }
        y[self.vertex] = total;
    }
}

/// The neighbor range *starts* at the previous header's `end`, so no second array can
/// disagree about step count or about which diagonal belongs to which vertex.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepHeader<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    /// `D^{-1}` premultiplied by the step's retained fraction, so no pass rebuilds it.
    pub(crate) pivot_scale: T,
}

/// The solve path keeps flat split arrays; a persisted sequence nests neighbors under
/// their step, so the `end` offsets are rebuilt on load rather than trusted.
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned"),
        try_from = "SequenceData<T>"
    )
)]
#[derive(Clone, Debug)]
pub(crate) struct EliminationSequence<T> {
    pub(crate) steps: Vec<StepHeader<T>>,
    pub(crate) neighbor_indices: Vec<u32>,
    /// What each neighbor takes of the *original* pivot, not of what predecessors left.
    pub(crate) coefficients: Vec<T>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    pub(crate) uneliminated: u32,
}

/// Nesting the neighbors under their step is what retires the range and
/// trailing-storage checks.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct StepData<T> {
    vertex: u32,
    pivot_scale: T,
    neighbors: Vec<(u32, T)>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct SequenceData<T> {
    uneliminated: u32,
    steps: Vec<StepData<T>>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<SequenceData<T>> for EliminationSequence<T> {
    type Error = FactorError;

    fn try_from(data: SequenceData<T>) -> Result<Self, Self::Error> {
        let mut steps = Vec::with_capacity(data.steps.len());
        let mut neighbor_indices = Vec::new();
        let mut coefficients = Vec::new();
        for step in data.steps {
            for (neighbor, coefficient) in step.neighbors {
                neighbor_indices.push(neighbor);
                coefficients.push(coefficient);
            }
            let nnz = neighbor_indices.len();
            // The only range invariant the nesting cannot carry: `end` is a `u32`.
            let end =
                u32::try_from(nnz).map_err(|_| FactorError::NonzeroCountExceedsU32 { nnz })?;
            steps.push(StepHeader {
                vertex: step.vertex,
                end,
                pivot_scale: step.pivot_scale,
            });
        }
        Ok(Self {
            steps,
            neighbor_indices,
            coefficients,
            uneliminated: data.uneliminated,
        })
    }
}

/// Mirrors [`SequenceData`] without materializing one, so serializing allocates nothing
/// regardless of `nnz`.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for EliminationSequence<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut out = serializer.serialize_struct("SequenceData", 2)?;
        out.serialize_field("uneliminated", &self.uneliminated)?;
        out.serialize_field("steps", &StepsView(self))?;
        out.end()
    }
}

#[cfg(feature = "serde")]
struct StepsView<'a, T>(&'a EliminationSequence<T>);

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for StepsView<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((0..self.0.steps.len()).map(|i| StepView(self.0, i)))
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
        out.serialize_field("pivot_scale", &sequence.steps[i].pivot_scale)?;
        out.serialize_field(
            "neighbors",
            &PairedNeighbors(
                &sequence.neighbor_indices[start..end],
                &sequence.coefficients[start..end],
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
    fn n_steps(&self) -> usize {
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
    fn step(&self, i: usize) -> EliminationStep<'_, T>
    where
        T: Copy,
    {
        let (start, end) = self.neighbor_range(i);
        EliminationStep {
            vertex: self.steps[i].vertex as usize,
            pivot_scale: self.steps[i].pivot_scale,
            neighbor_indices: &self.neighbor_indices[start..end],
            coefficients: &self.coefficients[start..end],
        }
    }

    #[cfg(any(feature = "serde", test))]
    pub(super) fn pinned_dim(&self) -> BlockDim {
        BlockDim::pinning(self.n_steps())
    }

    /// The ranges need no check — they are rebuilt from the nested persisted form,
    /// never read off the wire.
    #[cfg(any(feature = "serde", test))]
    pub(super) fn validate_values(&self) -> Result<(), FactorError>
    where
        T: num_traits::Float,
    {
        let n = self.pinned_dim().total();
        // `substitute` writes this entry unchecked.
        if (self.uneliminated as usize) >= n {
            return Err(FactorError::UneliminatedVertexInvalid {
                vertex: self.uneliminated,
                n,
            });
        }
        let mut eliminated = vec![false; n];
        for (i, step) in self.steps.iter().enumerate() {
            if (step.vertex as usize) >= n {
                return Err(FactorError::VertexOutOfBounds {
                    step: i,
                    vertex: step.vertex,
                    n,
                });
            }
            if core::mem::replace(&mut eliminated[step.vertex as usize], true) {
                return Err(FactorError::VertexEliminatedTwice {
                    step: i,
                    vertex: step.vertex,
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
            // Composing the fractions leaves them proportions of the pivot, so they
            // keep the same bounds; `pivot_scale` scales one and is not itself bounded.
            let coefficients = &self.coefficients[start..end];
            if !step.pivot_scale.is_finite()
                || coefficients
                    .iter()
                    .any(|c| !c.is_finite() || *c < T::zero() || *c > T::one())
            {
                return Err(FactorError::StepValueInvalid { step: i });
            }
        }
        if eliminated[self.uneliminated as usize] {
            return Err(FactorError::UneliminatedVertexInvalid {
                vertex: self.uneliminated,
                n,
            });
        }
        Ok(())
    }
}

impl<T: Real> EliminationSequence<T> {
    /// `D^+` zeroes the uneliminated vertex, whose entry is the rounding residue of a
    /// zero-sum right-hand side — at the right-hand side's scale, so leaving it there
    /// annihilates the solution-scale entries the backward pass reads (#93).
    pub(super) fn substitute(&self, y: &mut [T]) {
        for index in 0..self.n_steps() {
            self.step(index).apply_forward(y);
        }
        y[self.uneliminated as usize] = T::zero();
        for index in (0..self.n_steps()).rev() {
            self.step(index).apply_backward(y);
        }
    }
}

/// A sequence cannot exist without naming the vertex its steps left behind.
struct SequenceBuilder<T> {
    steps: Vec<StepHeader<T>>,
    neighbor_indices: Vec<u32>,
    coefficients: Vec<T>,
}

impl<T: Real> SequenceBuilder<T> {
    fn with_capacity(n: usize, degree_sum: usize) -> Self {
        Self {
            steps: Vec::with_capacity(n),
            neighbor_indices: Vec::with_capacity(degree_sum),
            coefficients: Vec::with_capacity(degree_sum),
        }
    }

    fn finish(self, uneliminated: u32) -> EliminationSequence<T> {
        EliminationSequence {
            steps: self.steps,
            neighbor_indices: self.neighbor_indices,
            coefficients: self.coefficients,
            uneliminated,
        }
    }

    /// Overflowing the `u32` range end is unreachable for tractable inputs, so assert
    /// in release too rather than truncate silently.
    fn push_step(&mut self, vertex: usize, diagonal: T, retained: T) {
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
            pivot_scale: retained
                * match T::one() / diagonal {
                    inverse if inverse.is_finite() => inverse,
                    _ => T::one(),
                },
        });
    }

    fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.push_step(vertex, diagonal, T::one());
    }

    /// Takes the column, not its parts: [`SampledColumn`] is what keeps a neighbor
    /// array from being stored against a fraction array of another length.
    fn record_column(&mut self, vertex: usize, column: &SampledColumn<T>) {
        let (neighbors, fractions) = column.pattern();
        self.neighbor_indices.extend_from_slice(neighbors);
        let retained = self.push_coefficients(fractions);
        self.push_step(vertex, column.diagonal, retained);
    }

    /// Composes the sampled fractions once at build time so the solve does not rebuild
    /// the running product on every apply: neighbor `i` takes `f_i * prod(1 - f_k), k < i`
    /// of the original pivot. The last slot — a placeholder the fractions never used —
    /// takes what all of them left, which is also what scales the pivot itself.
    fn push_coefficients(&mut self, fractions: &[T]) -> T {
        let Some((_, leading)) = fractions.split_last() else {
            return T::one();
        };
        let mut retained = T::one();
        for &f in leading {
            self.coefficients.push(f * retained);
            retained = retained * (T::one() - f);
        }
        self.coefficients.push(retained);
        retained
    }
}
