use super::cholesky::Cholesky;
use super::gauge::{pin_ground, project_zero_mean, relative_to_ground};
use crate::graph::Component;
use crate::types::Real;

#[cfg(test)]
mod tests;

/// One connected component, whose cholesky eliminates every slot but one; the variant is
/// how that freedom is fixed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum Block<T> {
    /// `L_C + diag(surplus_C)`, factored as `L_C` augmented by a ground vertex in the
    /// last slot, which no input entry maps to and the solution is measured from.
    Grounded(Cholesky<T>),
    /// `L_C`, whose solution is the zero-mean one.
    Floating(Cholesky<T>),
}

impl<T> Block<T> {
    /// The component's variant, carried over to its factor.
    pub(crate) fn of(component: &Component<'_, T>, cholesky: Cholesky<T>) -> Self {
        match component {
            Component::Laplacian(_) => Self::Floating(cholesky),
            Component::Grounded { .. } => Self::Grounded(cholesky),
        }
    }

    fn cholesky(&self) -> &Cholesky<T> {
        match self {
            Self::Grounded(cholesky) | Self::Floating(cholesky) => cholesky,
        }
    }

    pub(super) fn eliminated(&self) -> usize {
        self.cholesky().eliminated()
    }

    /// Input vertices, which is every slot but a ground.
    pub(super) fn vertices(&self) -> usize {
        match self {
            Self::Grounded(cholesky) => cholesky.eliminated(),
            Self::Floating(cholesky) => cholesky.eliminated() + 1,
        }
    }

    pub(super) fn slots(&self) -> usize {
        self.cholesky().eliminated() + 1
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> Block<T> {
    pub(super) fn validate(&self) -> Result<(), super::FactorError> {
        self.cholesky().validate()
    }
}

impl<T: Real> Block<T> {
    /// `slots` holds the block's right-hand side, then its solution; a ground's entry
    /// on entry is never read.
    pub(super) fn solve(&self, slots: &mut [T]) {
        match self {
            Self::Grounded(cholesky) => {
                pin_ground(slots);
                cholesky.apply(slots);
                relative_to_ground(slots);
            }
            Self::Floating(cholesky) => {
                project_zero_mean(slots);
                cholesky.apply(slots);
                project_zero_mean(slots);
            }
        }
    }
}
