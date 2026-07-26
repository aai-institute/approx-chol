//! Flat storage for the elimination sequence and its per-step row kernels.

use super::FactorError;
use crate::types::Real;

/// Zero-copy view of one elimination step: it eliminates `vertex` by splitting
/// its weight among neighbors according to `elimination_fractions`.
pub(crate) struct EliminationStep<'a, T> {
    pub(crate) vertex: usize,
    pub(crate) inv_diag: T,
    pub(crate) neighbor_indices: &'a [u32],
    pub(crate) elimination_fractions: &'a [T],
}

/// Every index a kernel below touches is in bounds already: the caller asserts
/// `y.len() >= n` once per solve, and every vertex and neighbor is under `n` —
/// by construction from the builder, by [`EliminationSequence::validate_for_dim`]
/// from serde.
/// Neither kernel re-checks per step.
impl<'a, T: num_traits::Float + Send + Sync + 'static> EliminationStep<'a, T> {
    /// Forward elimination: scatter pivot weight to neighbors, then scale by D^{-1}.
    #[inline(always)]
    pub(crate) fn apply_forward(&self, y: &mut [T]) {
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
    pub(crate) fn apply_backward(&self, y: &mut [T]) {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepHeader<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    pub(crate) inv_diag: T,
}

/// Contiguous memory owner for a sequence of elimination steps.
///
/// The two neighbor arrays cross the serde boundary as one array of pairs (see
/// [`SequenceData`]), so a persisted sequence cannot arrive with them
/// disagreeing about length; the solve path keeps them split.
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned"),
        from = "SequenceData<T>"
    )
)]
#[derive(Clone, Debug)]
pub(crate) struct EliminationSequence<T> {
    pub(crate) steps: Vec<StepHeader<T>>,
    pub(crate) neighbor_indices: Vec<u32>,
    pub(crate) elimination_fractions: Vec<T>,
}

/// Persisted shape of an [`EliminationSequence`]. Unzipping one pair array into
/// two columns is infallible, which is what retires the length check.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct SequenceData<T> {
    steps: Vec<StepHeader<T>>,
    /// One `(neighbor, elimination fraction)` per factor nonzero.
    neighbors: Vec<(u32, T)>,
}

#[cfg(feature = "serde")]
impl<T> From<SequenceData<T>> for EliminationSequence<T> {
    fn from(data: SequenceData<T>) -> Self {
        let (neighbor_indices, elimination_fractions) = data.neighbors.into_iter().unzip();
        Self {
            steps: data.steps,
            neighbor_indices,
            elimination_fractions,
        }
    }
}

/// Mirrors [`SequenceData`] field for field. Hand-written rather than a
/// `serde(into)` shadow so serializing neither clones the sequence nor forces a
/// `Clone` bound onto [`Factor`](super::Factor)'s serialize impl.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for EliminationSequence<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut out = serializer.serialize_struct("EliminationSequence", 2)?;
        out.serialize_field("steps", &self.steps)?;
        out.serialize_field("neighbors", &PairedNeighbors(self))?;
        out.end()
    }
}

#[cfg(feature = "serde")]
struct PairedNeighbors<'a, T>(&'a EliminationSequence<T>);

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for PairedNeighbors<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.0
                .neighbor_indices
                .iter()
                .zip(&self.0.elimination_fractions),
        )
    }
}

// Read-only accessors (no internal trait bounds).
impl<T> EliminationSequence<T> {
    #[inline(always)]
    pub(crate) fn n_steps(&self) -> usize {
        self.steps.len()
    }

    #[inline(always)]
    pub(crate) fn step(&self, i: usize) -> EliminationStep<'_, T>
    where
        T: Copy,
    {
        let step = &self.steps[i];
        let start = if i == 0 {
            0
        } else {
            self.steps[i - 1].end as usize
        };
        let end = step.end as usize;
        EliminationStep {
            vertex: step.vertex as usize,
            inv_diag: step.inv_diag,
            neighbor_indices: &self.neighbor_indices[start..end],
            elimination_fractions: &self.elimination_fractions[start..end],
        }
    }

    /// Check every structural invariant the solve path relies on, against a
    /// factor dimension `n`, so a deserialized (untrusted) factor is rejected
    /// before it can index storage out of bounds or silently return garbage.
    pub(crate) fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
        // Threading `start` through the loop makes the ranges contiguous and
        // non-decreasing by construction; only `start <= end <= nnz` is left to check.
        let nnz = self.neighbor_indices.len();
        let mut start = 0usize;
        for (i, step) in self.steps.iter().enumerate() {
            let end = step.end as usize;
            if start > end || end > nnz {
                return Err(FactorError::NeighborRangeInvalid {
                    step: i,
                    start,
                    end,
                    nnz,
                });
            }

            if (step.vertex as usize) >= n {
                return Err(FactorError::VertexOutOfBounds {
                    step: i,
                    vertex: step.vertex,
                    n,
                });
            }
            for &j in &self.neighbor_indices[start..end] {
                if (j as usize) >= n {
                    return Err(FactorError::NeighborOutOfBounds {
                        step: i,
                        neighbor: j,
                        n,
                    });
                }
            }
            start = end;
        }
        if start != nnz {
            return Err(FactorError::TrailingNeighborStorage {
                covered: start,
                nnz,
            });
        }
        Ok(())
    }
}

// Internal construction methods (pub(crate) only, Real bound is internal).
#[allow(private_bounds)]
impl<T: Real> EliminationSequence<T> {
    pub(crate) fn with_capacity(n: usize, degree_sum: usize) -> Self {
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
            // A pivot too small to invert is left unscaled, which *is* a scale
            // factor of one — storing it spares every use the special case.
            inv_diag: if diagonal.abs() > T::near_zero() {
                T::one() / diagonal
            } else {
                T::one()
            },
        });
    }

    /// Record an isolated vertex (no neighbors, clamped diagonal).
    pub(crate) fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.push_step(vertex, diagonal);
    }

    /// Record one sampled column (diagonal value plus its neighbor/fraction pattern).
    pub(crate) fn record_column(
        &mut self,
        vertex: usize,
        diagonal: T,
        neighbors: &[u32],
        fractions: &[T],
    ) {
        self.neighbor_indices.extend_from_slice(neighbors);
        self.elimination_fractions.extend_from_slice(fractions);
        self.push_step(vertex, diagonal);
    }
}
