//! One reduced irreducible block.

use super::congruence::{Congruence, Lift, ScaledFactors, Selection};
use super::decomposition::{BlockAdjacency, RowSums, SymmetricBlock};
use super::scaling::ScalingReport;
use crate::storage::{
    in_range, Adjacency, Balance, Bipartite, Gauge, Grounded, Grounding, Index, OffDiagonal, Real,
};

/// One irreducible principal block `A_c` of `A`, reduced: `M = F · A_c · F` weakly diagonally dominant, eliminated side first.
#[derive(Clone, Debug)]
pub(crate) struct Reduced<T, I, P> {
    /// `A_c`'s first vertex in `A`: it keys every draw, so a complement draws as its whole block does.
    pub(crate) key: usize,
    pub(crate) lift: Box<Lift<T, I>>,
    pub(crate) matrix: Grounded<T, P>,
    pub(crate) balance: Balance,
    /// On the vertices Cholesky factors: a bipartite block's kept side.
    pub(crate) gauge: Gauge,
    pub(crate) report: Option<Box<ScalingReport>>,
}

/// A block of `A`, whichever its shape.
pub(crate) type ReducedComponent<T, I> = Reduced<T, I, OffDiagonal<T, I>>;

/// A bipartite block's Schur complement onto its kept side, unformed: `C`, its lift already on the kept side's numbering.
pub(crate) type ReducedComplement<T, I> = Reduced<T, I, Bipartite<T, I>>;

impl<T: Real, I: Index, P> Reduced<T, I, P> {
    /// Vertices of `A` the block holds, each once however many sheets cover it.
    pub(crate) fn vertex_count(&self) -> usize {
        self.lift.vertices().len()
    }

    pub(super) fn into_component(self) -> ReducedComponent<T, I>
    where
        P: Into<OffDiagonal<T, I>>,
    {
        let Grounded {
            adjacency,
            grounding,
        } = self.matrix;
        Reduced {
            matrix: Grounded {
                adjacency: adjacency.into(),
                grounding,
            },
            key: self.key,
            lift: self.lift,
            balance: self.balance,
            gauge: self.gauge,
            report: self.report,
        }
    }
}

impl<T: Real, I: Index> ReducedComplement<T, I> {
    /// The lift tailed onto the kept side, each kept vertex of `A` numbered `local(v)`; tailed before `S`, so the eliminated side's `F` never shares its formation's peak.
    pub(super) fn onto_kept(self, local: impl Fn(usize) -> I) -> Self {
        let from = self.matrix.adjacency.n_eliminated();
        Self {
            lift: Box::new(self.lift.tail(from, local)),
            ..self
        }
    }
}

/// `M = F · A_c · F` over its vertices, its surplus still to be decided.
pub(super) struct Congruent<T, I, P> {
    vertices: Selection<I>,
    congruence: Congruence<T>,
    off_diagonal: P,
}

impl<T: Real, I: Index, P: Adjacency<T, I>> Congruent<T, I, P> {
    /// `S A_c S`, unscaled: signs keep every magnitude, so `A_c`'s rows are `M`'s.
    pub(super) fn signed(a: SymmetricBlock<T, I, P>, signs: Congruence<T>) -> (Self, RowSums<T>) {
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
        a: SymmetricBlock<T, I, P>,
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
    pub(super) fn incidence(adjacency: BlockAdjacency<I, P>) -> Self {
        let BlockAdjacency {
            vertices,
            off_diagonal,
        } = adjacency;
        let congruence = match off_diagonal.bipartite() {
            Some(c) => Congruence::NegateFrom(c.n_eliminated()),
            None => Congruence::Identity,
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

    pub(super) fn off_diagonal(&self) -> &P {
        &self.off_diagonal
    }

    pub(super) fn factor(&self, i: usize) -> T {
        self.congruence.factor(i)
    }

    /// Panics unless one factor, row and surplus per vertex, and `I` must index the factor's vertices.
    pub(super) fn block(
        self,
        grounding: Grounding<T>,
        balance: Balance,
        report: Option<ScalingReport>,
    ) -> Result<Reduced<T, I, P>, ReduceError> {
        let Self {
            vertices,
            congruence,
            off_diagonal,
        } = self;
        let lift = Lift::new(vertices, congruence);
        let vertices = lift.vertices();
        let n = vertices.len();
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        debug_assert!(
            balance == Balance::Frustrated || off_diagonal.nonpositive(),
            "a balanced M is a Z-matrix"
        );
        assert!(grounding.fits(n), "one surplus per row");
        let surplus = grounding.surplus().unwrap_or_default();
        if let Some(i) = surplus.iter().position(|&s| !in_range(s)) {
            let vertex = vertices.vertex(i);
            return Err(ReduceError::OutOfRange { vertex });
        }
        let key = vertices.vertex(0);
        let eliminated = (off_diagonal.bipartite()).map_or(0, Bipartite::n_eliminated);
        let gauge = lift.gauge(eliminated);
        Ok(Reduced {
            key,
            lift: Box::new(lift),
            matrix: Grounded {
                adjacency: off_diagonal,
                grounding,
            },
            balance,
            gauge,
            report: report.map(Box::new),
        })
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
