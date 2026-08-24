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
use clique_tree::{sample_column, ColumnShares, SampledColumn};
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
        sample_column(star, diag[v], sampler, &mut column);
        seq.push_sampled(v, column.diagonal, column.shares());

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

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const LANES: usize = 4;

/// Zero-copy view of one elimination step.
struct EliminationStep<'a, T> {
    vertex: usize,
    pivot_scale: T,
    neighbor_indices: &'a [u32],
    coefficients: &'a [T],
}

/// Neither kernel bounds-checks per step: the caller asserts `y.len() >= n` once per
/// solve, and every index is under `n` by construction or by
/// [`EliminationSequence::validate_values`].
impl<'a, T: Real> EliminationStep<'a, T> {
    /// Forward elimination: scatter pivot weight to neighbors, then scale by `pivot_scale`.
    #[inline(always)]
    fn apply_forward(&self, y: &mut [T]) {
        let pivot = y[self.vertex];
        match *self.coefficients {
            [c] => {
                debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                let j = self.neighbor_indices[0] as usize;
                y[j] = y[j] + pivot;
            }
            _ => {
                for (&j, &c) in self.neighbor_indices.iter().zip(self.coefficients) {
                    let j = j as usize;
                    y[j] = y[j] + c * pivot;
                }
            }
        }
        y[self.vertex] = pivot * self.pivot_scale;
    }

    /// The lone-neighbor arm is for the many degree-one rows, not the arithmetic.
    #[inline(always)]
    fn apply_backward(&self, y: &mut [T]) {
        match *self.coefficients {
            [] => {}
            [c] => {
                debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                let j = self.neighbor_indices[0] as usize;
                y[self.vertex] = y[self.vertex] + y[j];
            }
            [.., retained] => {
                let mut lanes = [T::zero(); LANES];
                lanes[0] = retained * y[self.vertex];
                let mut neighbors = self.neighbor_indices.chunks_exact(LANES);
                let mut coefficients = self.coefficients.chunks_exact(LANES);
                for (js, cs) in neighbors.by_ref().zip(coefficients.by_ref()) {
                    for lane in 0..LANES {
                        lanes[lane] = lanes[lane] + cs[lane] * y[js[lane] as usize];
                    }
                }
                let mut total = lanes.iter().fold(T::zero(), |sum, &lane| sum + lane);
                for (&j, &c) in neighbors.remainder().iter().zip(coefficients.remainder()) {
                    total = total + c * y[j as usize];
                }
                y[self.vertex] = total;
            }
        }
    }
}

/// The neighbor range *starts* at the previous header's `end`, so no second array can
/// disagree about step count or about which diagonal belongs to which vertex.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepHeader<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    /// `D^{-1}` premultiplied by the retained fraction, so no pass rebuilds it.
    pub(crate) pivot_scale: T,
}

/// The solve path keeps flat split arrays; a persisted sequence nests neighbors under
/// their step, so the `end` offsets are rebuilt on load rather than trusted.
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"),
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
    /// `None` for an isolated pivot, which has no neighbor to give anything to.
    column: Option<ColumnData<Vec<(u32, T)>>>,
}

/// No share for the remainder, so a payload cannot overspend a column's pivot. One
/// declaration for both directions: `S` is owned pairs decoding, borrowed slices encoding.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct ColumnData<S> {
    shares: S,
    remainder: u32,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct SequenceData<T> {
    uneliminated: u32,
    steps: Vec<StepData<T>>,
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<SequenceData<T>> for EliminationSequence<T> {
    type Error = FactorError;

    fn try_from(data: SequenceData<T>) -> Result<Self, Self::Error> {
        let mut builder = SequenceBuilder::with_capacity(data.steps.len(), 0);
        for step in data.steps {
            // Already composed on the wire, and absent rather than empty for an isolated pivot.
            if let Some(column) = step.column {
                builder.push_column(column.shares.into_iter(), column.remainder);
            }
            builder
                .push_header(step.vertex, step.pivot_scale)
                .map_err(|nnz| FactorError::NonzeroCountExceedsU32 { nnz })?;
        }
        Ok(builder.finish(data.uneliminated))
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
        // The decoder subtracts the remainder's share back out, so writing it writes
        // a value nothing reads.
        let column =
            sequence.neighbor_indices[start..end]
                .split_last()
                .map(|(&remainder, shared)| ColumnData {
                    shares: PairedNeighbors(shared, &sequence.coefficients[start..end - 1]),
                    remainder,
                });
        out.serialize_field("column", &column)?;
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
            // Overspent shares leave the derived remainder negative.
            let coefficients = &self.coefficients[start..end];
            if !step.pivot_scale.is_finite()
                || coefficients
                    .iter()
                    .any(|c| !c.is_finite() || *c < T::zero())
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

impl<T: num_traits::Float> SequenceBuilder<T> {
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

    /// The one site deriving the remainder, so wire and sampler cannot disagree.
    fn push_column(
        &mut self,
        shares: impl ExactSizeIterator<Item = (u32, T)>,
        remainder: u32,
    ) -> T {
        self.neighbor_indices.reserve(shares.len() + 1);
        self.coefficients.reserve(shares.len() + 1);
        let mut left = T::one();
        for (neighbor, share) in shares {
            self.neighbor_indices.push(neighbor);
            self.coefficients.push(share);
            left = left - share;
        }
        self.neighbor_indices.push(remainder);
        self.coefficients.push(left);
        left
    }

    /// Closes the step over the column pushed since the last one; `Err` carries the `nnz`.
    fn push_header(&mut self, vertex: u32, pivot_scale: T) -> Result<(), usize> {
        let nnz = self.neighbor_indices.len();
        let end = u32::try_from(nnz).map_err(|_| nnz)?;
        self.steps.push(StepHeader {
            vertex,
            end,
            pivot_scale,
        });
        Ok(())
    }

    /// `None` is a pivot that keeps everything it retained.
    fn push_sampled(&mut self, vertex: usize, diagonal: T, column: Option<ColumnShares<'_, T>>) {
        let retained = match column {
            Some(c) => self.push_column(
                c.neighbors
                    .iter()
                    .copied()
                    .zip(c.coefficients.iter().copied()),
                c.remainder,
            ),
            None => T::one(),
        };
        // A small pivot inverts fine; `one` would drop the block's scale outright.
        let inverse = match T::one() / diagonal {
            inverse if inverse.is_finite() => inverse,
            _ => T::one(),
        };
        self.push_header(vertex as u32, retained * inverse)
            .unwrap_or_else(|nnz| panic!("factor nonzero count {nnz} exceeds u32 range capacity"));
    }
}
