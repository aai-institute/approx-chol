//! `A` as its reduced components: what the factorization consumes.

use std::sync::Arc;

use super::congruence::Congruence;
use super::scaling::ScalingReport;
use crate::storage::{diagonal_in_range, CrossBlock, UpperAdjacency};
use crate::Real;

/// Where a component sits in `A`: its vertices, and the congruence between `A_c` and `M`.
#[derive(Debug)]
pub(crate) struct Frame<T> {
    vertices: Box<[u32]>,
    congruence: Congruence<T>,
}

impl<T: Real> Frame<T> {
    /// Panics unless the congruence scales exactly these vertices.
    pub(super) fn new(vertices: Box<[u32]>, congruence: Congruence<T>) -> Self {
        assert!(congruence.fits(vertices.len()), "one factor per vertex");
        Self {
            vertices,
            congruence,
        }
    }
}

impl<T> Frame<T> {
    pub(crate) fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    pub(crate) fn congruence(&self) -> &Congruence<T> {
        &self.congruence
    }
}

/// Whether a signature folds `M` (Harary), and `M`'s grounding.
#[derive(Clone, Debug)]
pub(crate) enum Balance<T> {
    /// Folded to a Z-matrix: singular exactly when ungrounded.
    Balanced(Grounding<T>),
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even ungrounded.
    Frustrated(Grounding<T>),
}

impl<T: Real> Balance<T> {
    pub(crate) fn grounding(&self) -> &Grounding<T> {
        match self {
            Self::Balanced(grounding) | Self::Frustrated(grounding) => grounding,
        }
    }

    /// The ground is a vertex too: its diagonal `Σs`, once per sheet, clears the same bound as any other.
    fn ground_in_range(&self) -> bool {
        let sheets = match self {
            Self::Balanced(_) => T::one(),
            Self::Frustrated(_) => T::one() + T::one(),
        };
        let total = |surplus: &[T]| surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        (self.grounding().surplus())
            .is_none_or(|surplus| diagonal_in_range(sheets * total(surplus)))
    }
}

/// The surplus `s = diag(M) − Σ|offdiag(M)|` closing `M` into an SDDM; none leaves it singular.
#[derive(Clone, Debug)]
pub(crate) struct Grounding<T>(Option<Box<[T]>>);

impl<T: Real> Grounding<T> {
    /// Panics on a negative or non-finite entry; ungrounded exactly when every entry is zero.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        let valid = |&s: &T| s.is_finite() && s >= T::zero();
        assert!(
            surplus.iter().all(valid),
            "a surplus is finite and nonnegative"
        );
        match surplus.iter().all(|&s| s == T::zero()) {
            true => Self(None),
            false => Self(Some(surplus.into())),
        }
    }

    pub(super) fn ungrounded() -> Self {
        Self(None)
    }

    pub(crate) fn surplus(&self) -> Option<&[T]> {
        self.0.as_deref()
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.0.is_some()
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.0.as_ref().is_none_or(|surplus| surplus.len() == n)
    }
}

/// A component's off-diagonal by shape: `A_c`'s signed one, or `M`'s once congruent.
#[derive(Clone, Debug)]
pub(crate) enum OffDiagonal<T> {
    General(UpperAdjacency<T>),
    /// Both sides nonempty, the larger eliminated first.
    Bipartite(Arc<CrossBlock<T>>),
}

impl<T: Real> OffDiagonal<T> {
    pub(super) fn n(&self) -> usize {
        match self {
            Self::General(adjacency) => adjacency.n(),
            Self::Bipartite(cross) => cross.n(),
        }
    }

    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    pub(super) fn abs_row_sums(&self) -> Vec<T> {
        match self {
            Self::General(adjacency) => adjacency.abs_row_sums(),
            Self::Bipartite(cross) => cross.abs_row_sums(),
        }
    }

    /// Entries per row in local order: the rounding a row sum can carry.
    pub(super) fn degrees(&self) -> Vec<usize> {
        match self {
            Self::General(adjacency) => adjacency.degrees(),
            Self::Bipartite(cross) => cross.degrees(),
        }
    }

    /// `F X F`; `None` where an entry leaves [`in_range`](crate::storage::in_range).
    pub(super) fn congruent(self, f: &Congruence<T>) -> Option<Self> {
        match f.factors(self.n()) {
            None => Some(self),
            Some(factors) => self.scaled(&factors),
        }
    }

    fn scaled(self, f: &[T]) -> Option<Self> {
        Some(match self {
            Self::General(adjacency) => Self::General(adjacency.scaled(f)?),
            Self::Bipartite(cross) => {
                Self::Bipartite(Arc::new(Arc::unwrap_or_clone(cross).scaled(f)?))
            }
        })
    }

    /// As [`scaled`](Self::scaled), with the new row sums from the same pass.
    pub(super) fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        Some(match self {
            Self::General(adjacency) => {
                let (adjacency, sums) = adjacency.scaled_abs_row_sums(f)?;
                (Self::General(adjacency), sums)
            }
            Self::Bipartite(cross) => {
                let (cross, sums) = Arc::unwrap_or_clone(cross).scaled_abs_row_sums(f)?;
                (Self::Bipartite(Arc::new(cross)), sums)
            }
        })
    }

    pub(super) fn nonpositive(&self) -> bool {
        match self {
            Self::General(adjacency) => adjacency.nonpositive(),
            Self::Bipartite(cross) => cross.nonpositive(),
        }
    }
}

/// One component of `A`, reduced: `M = F · A_c · F` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Component<T = f64> {
    frame: Arc<Frame<T>>,
    off_diagonal: OffDiagonal<T>,
    balance: Balance<T>,
}

impl<T: Real> Component<T> {
    /// Panics unless one row and one surplus per vertex; the ground's `Σs` must clear the bound any diagonal does.
    pub(super) fn new(
        frame: Frame<T>,
        off_diagonal: OffDiagonal<T>,
        balance: Balance<T>,
    ) -> Result<Self, ReduceError> {
        let n = frame.vertices().len();
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        assert!(balance.grounding().fits(n), "one surplus per row");
        if !balance.ground_in_range() {
            let vertex = frame.vertices()[0];
            return Err(ReduceError::OutOfRange { vertex });
        }
        Ok(Self {
            frame: Arc::new(frame),
            off_diagonal,
            balance,
        })
    }

    /// The component's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
    }

    /// Present iff this component's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.frame.congruence().report()
    }

    /// Shared with every factor of this component.
    pub(crate) fn frame(&self) -> &Arc<Frame<T>> {
        &self.frame
    }

    /// Whether `A_c` is proven singular, so a solve returns `A_c⁺b`.
    pub fn floats(&self) -> bool {
        matches!(&self.balance, Balance::Balanced(grounding) if !grounding.is_grounded())
    }

    /// `M`: its off-diagonal and its balance.
    pub(crate) fn m(&self) -> (&OffDiagonal<T>, &Balance<T>) {
        (&self.off_diagonal, &self.balance)
    }
}

/// `A` as its connected [`Component`]s, each reduced; vertices in none solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64> {
    n: usize,
    components: Vec<Component<T>>,
    /// Lone vertices with a zero diagonal, recorded so a solve zeroes them without a pass over `n`.
    dead: Box<[u32]>,
}

impl<T: Real> Sgdd<T> {
    pub(super) fn new(n: usize, components: Vec<Component<T>>, dead: Vec<u32>) -> Self {
        Self {
            n,
            components,
            dead: dead.into_boxed_slice(),
        }
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The components, each factored and solved on its own.
    pub fn components(&self) -> &[Component<T>] {
        &self.components
    }

    pub(crate) fn dead(&self) -> &[u32] {
        &self.dead
    }

    /// The worst component's report: present iff some scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        (self.components.iter())
            .filter_map(Component::scaling_report)
            .max_by(|a, b| a.violation.total_cmp(&b.violation))
    }
}

/// Why a component has no SDDM form; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// No positive diagonal scaling makes the component weakly dominant: `diag − |offdiag|` is not PSD.
    NotGdd {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// A scale or scaled entry left the representable range, or the component passes `1/ε` vertices.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// An assembled entry, diagonal, row sum or ground left the representable range.
    OutOfRange {
        /// The entry's vertex, in `A`'s numbering.
        vertex: u32,
    },
}

impl core::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotGdd { vertex } => write!(
                f,
                "the component of vertex {vertex} is not generalized diagonally dominant"
            ),
            Self::Unscalable { vertex } => write!(
                f,
                "scaling the component of vertex {vertex} leaves the representable range"
            ),
            Self::OutOfRange { vertex } => {
                write!(
                    f,
                    "an entry at vertex {vertex} is outside the representable range"
                )
            }
        }
    }
}

impl std::error::Error for ReduceError {}
