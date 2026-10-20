//! `A` as its reduced irreducible blocks: what the factorization consumes.

use super::congruence::{Congruence, Restriction};
use super::decomposition::Components;
use super::scaling::ScalingReport;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, Index, Real};

/// Whether a signature folds `M` (Harary), and `M`'s grounding.
#[derive(Clone, Debug)]
pub(crate) enum Balance<T> {
    /// Folded to a Z-matrix: singular exactly when ungrounded.
    Balanced(Grounding<T>),
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even ungrounded.
    Frustrated(Grounding<T>),
}

impl<T: Real> Balance<T> {
    fn grounding(&self) -> &Grounding<T> {
        match self {
            Self::Balanced(grounding) | Self::Frustrated(grounding) => grounding,
        }
    }
}

/// The surplus `s = diag(M) − Σ|offdiag(M)|` closing `M` into an SDDM; none is `σ = 0`, singular only when balanced.
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

    /// Row `i`'s ground edge; a zero surplus couples nothing, so it is none.
    pub(crate) fn edge(&self, i: usize) -> Option<T> {
        self.surplus().map(|s| s[i]).filter(|&s| s > T::zero())
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.0.is_some()
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.0.as_ref().is_none_or(|surplus| surplus.len() == n)
    }
}

/// One irreducible principal block `A_c` of `A`, reduced: `M = F · A_c · F` weakly diagonally dominant, eliminated side first.
#[derive(Clone, Debug)]
pub(crate) struct IrreducibleBlock<T = f64, I = u32> {
    restriction: Box<Restriction<T, I>>,
    off_diagonal: OffDiagonal<T, I>,
    balance: Balance<T>,
    report: Option<Box<ScalingReport>>,
}

impl<T: Real, I: Index> IrreducibleBlock<T, I> {
    /// Panics unless one factor, row and surplus per vertex; the ground's `Σs` must clear the bound any diagonal does, and `I` must index the factor's vertices.
    pub(super) fn new(
        vertices: Box<[I]>,
        congruence: Congruence<T>,
        off_diagonal: OffDiagonal<T, I>,
        balance: Balance<T>,
        report: Option<ScalingReport>,
    ) -> Result<Self, ReduceError> {
        let restriction = Restriction::new(vertices, congruence);
        let vertices = restriction.vertices();
        let n = vertices.len();
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        assert!(balance.grounding().fits(n), "one surplus per row");
        let sheets = match balance {
            Balance::Balanced(_) => {
                debug_assert!(off_diagonal.nonpositive(), "a balanced M is a Z-matrix");
                T::one()
            }
            Balance::Frustrated(_) => T::one() + T::one(),
        };
        let surplus = balance.grounding().surplus().unwrap_or_default();
        if let Some(i) = surplus.iter().position(|&s| !crate::in_range(s)) {
            let vertex = vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        // The ground is a vertex too: its diagonal `Σs`, once per sheet, clears the same bound as any other.
        let total = surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        if !diagonal_in_range(sheets * total) {
            let vertex = vertices[0].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        Ok(Self {
            restriction: Box::new(restriction),
            off_diagonal,
            balance,
            report: report.map(Box::new),
        })
    }

    /// Moved out, so each part is freed once its factor stops reading it.
    pub(crate) fn into_parts(self) -> BlockParts<T, I> {
        (
            self.restriction,
            self.off_diagonal,
            self.balance,
            self.report,
        )
    }
}

type BlockParts<T, I> = (
    Box<Restriction<T, I>>,
    OffDiagonal<T, I>,
    Balance<T>,
    Option<Box<ScalingReport>>,
);

/// `A`'s direct summands by kind, each factored and solved on its own.
#[derive(Clone, Debug)]
pub(crate) struct Summands<B, I, T> {
    /// Each coupled component's block.
    pub(crate) blocks: Vec<B>,
    /// Vertices no entry couples, each with its surplus `σ`, which is its whole pivot.
    pub(crate) lone: Vec<(I, T)>,
    /// One-vertex components whose pivot is zero: an in-place solve zeroes them.
    pub(crate) dead: Vec<I>,
}

/// `A` as its reduced summands; built only from extraction, so they cover `A` once.
#[derive(Clone, Debug)]
pub(crate) struct Sgdd<T = f64, I = u32> {
    n: usize,
    summands: Summands<IrreducibleBlock<T, I>, I, T>,
}

impl<T: Real, I: Index> Sgdd<T, I> {
    pub(super) fn new(components: Components<IrreducibleBlock<T, I>, I, T>) -> Self {
        let Components {
            partition,
            summands,
        } = components;
        Self {
            n: partition.n(),
            summands,
        }
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn into_summands(self) -> Summands<IrreducibleBlock<T, I>, I, T> {
        self.summands
    }
}

/// Why a block has no SDDM form; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// A diagonal entry is not positive, so no positive scaling makes its row dominant.
    NonpositiveDiagonal {
        /// The entry's vertex, in `A`'s numbering.
        vertex: usize,
    },
    /// A scale or scaled entry left the representable range.
    Unscalable {
        /// The block's first vertex in local order, in `A`'s numbering.
        vertex: usize,
    },
    /// The block passes `1/ε` vertices: floor weights in series would fall below `MIN_POSITIVE`.
    TooLarge {
        /// The block's first vertex, in `A`'s numbering.
        vertex: usize,
    },
    /// An assembled entry, diagonal, row sum or ground left the representable range.
    OutOfRange {
        /// The entry's vertex, in `A`'s numbering.
        vertex: usize,
    },
}

impl core::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonpositiveDiagonal { vertex } => {
                write!(f, "vertex {vertex} has a nonpositive diagonal")
            }
            Self::Unscalable { vertex } => write!(
                f,
                "scaling the block of vertex {vertex} leaves the representable range"
            ),
            Self::TooLarge { vertex } => {
                write!(f, "the block of vertex {vertex} passes 1/ε vertices")
            }
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
