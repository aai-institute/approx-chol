//! How a block's singular system is made solvable, and how the solution is read
//! back out of it.

use crate::types::{count_as_scalar, Real};

/// The peer of [`Cholesky`](super::cholesky::Cholesky): a block is one of each, and
/// the two are chosen independently — this one by augmentation, that one by policy.
///
/// Every block is a connected pure Laplacian, so its null space is exactly
/// `span{1}` and one variable has to be pinned. Which variable never varies — it is
/// the block's last — but whether pinning it *is* the answer does.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Anchor {
    /// The pinned variable is a Gremban ground vertex, which absorbs the null-space
    /// component exactly, so the anchored solution already solves the input system.
    Ground,
    /// No ground vertex, so the null-space component is projected out instead and
    /// the solution is fixed to the zero-mean representative.
    Floating,
}

impl Anchor {
    /// A block is grounded exactly when the ground vertex is its last one — the
    /// only place it can be, since ingestion appends it above every real vertex
    /// and a component lists its vertices in ascending order.
    pub(crate) fn of_block(ground: Option<u32>, last_vertex: u32) -> Self {
        if ground == Some(last_vertex) {
            Self::Ground
        } else {
            Self::Floating
        }
    }

    /// Put `values` into the block's range. Every block solves only a zero-sum
    /// right-hand side, and both arms deliver one.
    pub(super) fn prepare<T: Real>(self, values: &mut [T]) {
        match self {
            // The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`.
            Self::Ground => {
                let Some((pinned, rest)) = values.split_last_mut() else {
                    return;
                };
                *pinned = -rest.iter().fold(T::zero(), |sum, &value| sum + value);
            }
            // Nothing to absorb the null-space component, so project it out; an
            // inconsistent right-hand side then gives least squares.
            Self::Floating => project_zero_mean(values),
        }
    }

    /// Anchor the solution at zero in the pinned variable, and — when `canonical` —
    /// replace a floating block's arbitrary constant with the zero-mean choice.
    pub(super) fn recover<T: Real>(self, values: &mut [T], canonical: bool) {
        let Some(&pinned) = values.last() else {
            return;
        };
        for value in values.iter_mut() {
            *value = *value - pinned;
        }
        if canonical && self == Self::Floating {
            project_zero_mean(values);
        }
    }
}

fn project_zero_mean<T: Real>(values: &mut [T]) {
    let count = count_as_scalar::<T, _>(values.len());
    let mean = values.iter().fold(T::zero(), |sum, &value| sum + value) / count;
    for value in values.iter_mut() {
        *value = *value - mean;
    }
}
