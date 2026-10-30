//! One reduced irreducible block, its balance and grounding.

use super::congruence::{Congruence, Restriction, ScaledFactors};
use super::diagonal_in_range;
use super::scaling::ScalingReport;
use crate::storage::{OffDiagonal, Rows};
use crate::{Index, Real};

/// Whether a signature folds `M` (Harary), and `M`'s grounding.
#[derive(Clone, Debug)]
pub(crate) enum Balance<T> {
    /// Folded to a Z-matrix: singular exactly when ungrounded.
    Balanced(Grounding<T>),
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even ungrounded.
    Frustrated(Grounding<T>),
}

impl<T> Balance<T> {
    /// A balanced `M` folds onto one sheet; a frustrated one needs the cover's two.
    fn sheets(&self) -> Sheets {
        match self {
            Self::Balanced(_) => Sheets::One,
            Self::Frustrated(_) => Sheets::Two,
        }
    }

    fn split(self) -> (Sheets, Grounding<T>) {
        let sheets = self.sheets();
        match self {
            Self::Balanced(grounding) | Self::Frustrated(grounding) => (sheets, grounding),
        }
    }
}

/// How many copies of each vertex a matrix holds: two for a Gremban cover, interleaved by vertex.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sheets {
    One,
    Two,
}

impl Sheets {
    pub(crate) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
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

    pub(crate) fn ungrounded() -> Self {
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

/// A bipartite block's Schur complement onto its kept side, unformed: `C`, and the restriction onto the kept side's numbering.
pub(crate) struct BlockComplement<T, I> {
    /// The whole block's first vertex in `A`.
    first: usize,
    restriction: Box<Restriction<T, I>>,
    cross: Rows<T, I>,
    sheets: Sheets,
    grounding: Grounding<T>,
    report: Option<Box<ScalingReport>>,
}

impl<T: Real, I: Index> BlockComplement<T, I> {
    /// `block`'s restriction tailed onto its kept side, each kept vertex of `A` numbered `local(v)`.
    pub(super) fn new(block: IrreducibleBlock<T, I>, local: impl Fn(usize) -> I) -> Self {
        let (whole, off_diagonal, sheets, grounding, report) = block.into_parts();
        let OffDiagonal::Bipartite(cross) = off_diagonal else {
            unreachable!("a complement's layout leaves each block both sides")
        };
        let first = whole.vertices().vertex(0);
        // Tailed before `S`, so the eliminated side's `F` never shares its formation's peak.
        let restriction = Box::new(whole.tail(cross.n_rows(), local));
        Self {
            first,
            restriction,
            cross,
            sheets,
            grounding,
            report,
        }
    }

    /// Moved out, the whole block's first vertex first, as it keys the draws.
    pub(crate) fn into_parts(self) -> ComplementParts<T, I> {
        (
            self.first,
            self.restriction,
            self.cross,
            self.sheets,
            self.grounding,
            self.report,
        )
    }
}

type ComplementParts<T, I> = (
    usize,
    Box<Restriction<T, I>>,
    Rows<T, I>,
    Sheets,
    Grounding<T>,
    Option<Box<ScalingReport>>,
);

/// `M = F · A_c · F` over its vertices, its surplus still to be decided.
pub(super) struct Congruent<T, I> {
    vertices: Box<[I]>,
    congruence: Congruence<T>,
    off_diagonal: OffDiagonal<T, I>,
}

impl<T: Real, I: Index> Congruent<T, I> {
    /// `S A_c S`, unscaled.
    pub(super) fn signed(
        vertices: Box<[I]>,
        off_diagonal: OffDiagonal<T, I>,
        signs: Congruence<T>,
    ) -> Self {
        let off_diagonal = match signs.factors(off_diagonal.n()) {
            Some(s) => off_diagonal.signed(&s),
            None => off_diagonal,
        };
        Self {
            vertices,
            congruence: signs,
            off_diagonal,
        }
    }

    /// `S R A_c R S` with its `Σ_j |m_ij|`; `None` once a scaled entry leaves the range.
    pub(super) fn scaled(
        vertices: Box<[I]>,
        off_diagonal: OffDiagonal<T, I>,
        signs: Congruence<T>,
        scales: &[T],
    ) -> Option<(Self, Vec<T>)> {
        let factors: Box<[T]> = (scales.iter().enumerate())
            .map(|(i, &scale)| signs.factor(i) * scale)
            .collect();
        let (off_diagonal, abs) = off_diagonal.scaled_abs_row_sums(&factors)?;
        let congruence = Congruence::Scaled(ScaledFactors::new(factors));
        let m = Self {
            vertices,
            congruence,
            off_diagonal,
        };
        Some((m, abs))
    }

    /// An incidence Gram, assembled switched across a bipartition.
    pub(super) fn incidence(vertices: Box<[I]>, off_diagonal: OffDiagonal<T, I>) -> Self {
        let congruence = match &off_diagonal {
            OffDiagonal::Bipartite(c) => Congruence::NegateFrom(c.n_rows()),
            OffDiagonal::General(_) => Congruence::Identity,
        };
        Self {
            vertices,
            congruence,
            off_diagonal,
        }
    }

    /// In `A`'s numbering, in local order.
    pub(super) fn vertices(&self) -> &[I] {
        &self.vertices
    }

    pub(super) fn off_diagonal(&self) -> &OffDiagonal<T, I> {
        &self.off_diagonal
    }

    pub(super) fn factor(&self, i: usize) -> T {
        self.congruence.factor(i)
    }

    /// Panics unless one factor, row and surplus per vertex; the ground's `Σs` must clear the bound any diagonal does, and `I` must index the factor's vertices.
    pub(super) fn block(
        self,
        balance: Balance<T>,
        report: Option<ScalingReport>,
    ) -> Result<IrreducibleBlock<T, I>, ReduceError> {
        let Self {
            vertices,
            congruence,
            off_diagonal,
        } = self;
        let restriction = Restriction::new(vertices, congruence);
        let vertices = restriction.vertices();
        let n = vertices.len();
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        let (Balance::Balanced(grounding) | Balance::Frustrated(grounding)) = &balance;
        let sheets = balance.sheets();
        debug_assert!(
            sheets == Sheets::Two || off_diagonal.nonpositive(),
            "a balanced M is a Z-matrix"
        );
        assert!(grounding.fits(n), "one surplus per row");
        let surplus = grounding.surplus().unwrap_or_default();
        if let Some(i) = surplus.iter().position(|&s| !crate::in_range(s)) {
            let vertex = vertices.vertex(i);
            return Err(ReduceError::OutOfRange { vertex });
        }
        // The ground is a vertex too: its diagonal `Σs`, once per sheet, clears the same bound as any other.
        let total = surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        let ground = match sheets {
            Sheets::One => total,
            Sheets::Two => total + total,
        };
        if !diagonal_in_range(ground) {
            let vertex = vertices.vertex(0);
            return Err(ReduceError::OutOfRange { vertex });
        }
        Ok(IrreducibleBlock {
            restriction: Box::new(restriction),
            off_diagonal,
            balance,
            report: report.map(Box::new),
        })
    }
}

impl<T: Real, I: Index> IrreducibleBlock<T, I> {
    /// Vertices of `A` the block holds, each once however many sheets cover it.
    pub(crate) fn vertex_count(&self) -> usize {
        self.restriction.vertices().len()
    }

    /// Moved out, so each part is freed once its factor stops reading it.
    pub(crate) fn into_parts(self) -> BlockParts<T, I> {
        let (sheets, grounding) = self.balance.split();
        (
            self.restriction,
            self.off_diagonal,
            sheets,
            grounding,
            self.report,
        )
    }
}

type BlockParts<T, I> = (
    Box<Restriction<T, I>>,
    OffDiagonal<T, I>,
    Sheets,
    Grounding<T>,
    Option<Box<ScalingReport>>,
);

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
