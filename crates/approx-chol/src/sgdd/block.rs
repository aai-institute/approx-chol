//! One reduced irreducible block.

use super::congruence::{Congruence, Restriction, ScaledFactors, Selection};
use super::decomposition::{BlockAdjacency, RowSums, SymmetricBlock};
use super::scaling::ScalingReport;
use crate::storage::{in_range, Balance, Grounding, Index, OffDiagonal, Real, Rows, Sheets};

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
    vertices: Selection<I>,
    congruence: Congruence<T>,
    off_diagonal: OffDiagonal<T, I>,
}

impl<T: Real, I: Index> Congruent<T, I> {
    /// `S A_c S`, unscaled: signs keep every magnitude, so `A_c`'s rows are `M`'s.
    pub(super) fn signed(a: SymmetricBlock<T, I>, signs: Congruence<T>) -> (Self, RowSums<T>) {
        let SymmetricBlock {
            adjacency:
                BlockAdjacency {
                    vertices,
                    off_diagonal,
                },
            rows,
        } = a;
        let off_diagonal = match signs.factors(off_diagonal.n()) {
            Some(s) => off_diagonal.signed(&s),
            None => off_diagonal,
        };
        let m = Self {
            vertices,
            congruence: signs,
            off_diagonal,
        };
        (m, rows)
    }

    /// `S R A_c R S` with its rows; `None` once a scaled off-diagonal entry leaves the range.
    pub(super) fn scaled(
        a: SymmetricBlock<T, I>,
        signs: Congruence<T>,
        scales: &[T],
    ) -> Option<(Self, RowSums<T>)> {
        let SymmetricBlock {
            adjacency:
                BlockAdjacency {
                    vertices,
                    off_diagonal,
                },
            rows,
        } = a;
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
        let diagonal = (rows.diagonal.iter().zip(scales))
            .map(|(&a, &f)| f * a * f)
            .collect();
        Some((m, RowSums { diagonal, abs }))
    }

    /// An incidence Gram, assembled switched across a bipartition.
    pub(super) fn incidence(adjacency: BlockAdjacency<T, I>) -> Self {
        let BlockAdjacency {
            vertices,
            off_diagonal,
        } = adjacency;
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
    pub(super) fn vertices(&self) -> &Selection<I> {
        &self.vertices
    }

    pub(super) fn off_diagonal(&self) -> &OffDiagonal<T, I> {
        &self.off_diagonal
    }

    pub(super) fn factor(&self, i: usize) -> T {
        self.congruence.factor(i)
    }

    /// Panics unless one factor, row and surplus per vertex, and `I` must index the factor's vertices.
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
        debug_assert!(
            balance.sheets() == Sheets::Two || off_diagonal.nonpositive(),
            "a balanced M is a Z-matrix"
        );
        assert!(grounding.fits(n), "one surplus per row");
        let surplus = grounding.surplus().unwrap_or_default();
        if let Some(i) = surplus.iter().position(|&s| !in_range(s)) {
            let vertex = vertices.vertex(i);
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
    /// An assembled entry, diagonal, row sum or row's ground left the representable range.
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
