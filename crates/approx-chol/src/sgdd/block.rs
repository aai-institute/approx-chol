//! One reduced irreducible block, its balance and grounding.

use std::ops::Range;

use super::congruence::{Congruence, ScaledFactors};
use super::scaling::{Clamp, ScaleVector};
use crate::storage::{OffDiagonal, Rows};
use crate::{Index, Real};

/// Whether a signature folds `M` (Harary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Balance {
    /// Folded to a Z-matrix: singular exactly when ungrounded.
    Balanced,
    /// No signature exists: definite even ungrounded, as a two-sheet cover on `[z; −z]` is.
    Frustrated,
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

    /// The surplus on `grounds`, the rows whose surplus is a ground, each other row's rounding zeroed; with none, every entry, as `settle` keeps them. Below the range floor (#163) a ground is out of range, `Err` naming its row, and any other entry is zeroed; zeroing all of a surplus is out of range at its first.
    pub(crate) fn kept(self, grounds: Option<&[bool]>) -> Result<Self, usize> {
        let Some(mut surplus) = self.0 else {
            return Ok(self);
        };
        let mut zeroed = None;
        for (i, s) in surplus.iter_mut().enumerate() {
            match (grounds.map(|g| g[i]), crate::in_range(*s)) {
                (Some(true), false) => return Err(i),
                (Some(false), _) | (None, false) => {
                    zeroed = zeroed.or((*s > T::zero()).then_some(i));
                    *s = T::zero();
                }
                (Some(true) | None, true) => {}
            }
        }
        match (Self::of(surplus.into_vec()), zeroed) {
            (kept, Some(i)) if !kept.is_grounded() => Err(i),
            (kept, _) => Ok(kept),
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
    congruence: Box<Congruence<T>>,
    off_diagonal: OffDiagonal<T, I>,
    balance: Balance,
    grounding: Grounding<T>,
    grounding_unproven: bool,
    /// Boxed, as most components have none.
    clamp: Option<Box<Clamp>>,
}

/// A bipartite block's Schur complement onto its kept side, unformed: `C`, and `F` on the kept side.
pub(crate) struct BlockComplement<T, I> {
    /// The whole block's first vertex in `A`.
    first: usize,
    congruence: Box<Congruence<T>>,
    cross: Rows<T, I>,
    balance: Balance,
    grounding: Grounding<T>,
    grounding_unproven: bool,
    /// The whole block's.
    clamp: Option<Box<Clamp>>,
}

/// A coupled component's block, which knows the vertices of its component it keeps.
pub(crate) trait KeptVertices {
    /// Its kept vertices in its component's local order: a tail, so the component ends where they do.
    fn kept(&self) -> Range<usize>;
}

impl<T: Real, I: Index> BlockComplement<T, I> {
    /// `block` onto its kept side; `first` is the whole block's first vertex in `A`.
    pub(super) fn new(block: IrreducibleBlock<T, I>, first: usize) -> Self {
        let (congruence, off_diagonal, balance, grounding, grounding_unproven, clamp) =
            block.into_parts();
        let OffDiagonal::Bipartite(cross) = off_diagonal else {
            unreachable!("a complement's layout leaves each block both sides")
        };
        // Tailed before `S`, so the eliminated side's `F` never shares its formation's peak.
        let congruence = Box::new(congruence.tail(cross.n_rows()));
        Self {
            first,
            congruence,
            cross,
            balance,
            grounding,
            grounding_unproven,
            clamp,
        }
    }

    /// Moved out, the whole block's first vertex first, as it keys the draws.
    pub(crate) fn into_parts(self) -> ComplementParts<T, I> {
        (
            self.first,
            self.congruence,
            self.cross,
            self.balance,
            self.grounding,
            self.grounding_unproven,
            self.clamp,
        )
    }
}

impl<T: Real, I: Index> KeptVertices for BlockComplement<T, I> {
    fn kept(&self) -> Range<usize> {
        let eliminated = self.cross.n_rows();
        eliminated..eliminated + self.cross.n_cols()
    }
}

type ComplementParts<T, I> = (
    usize,
    Box<Congruence<T>>,
    Rows<T, I>,
    Balance,
    Grounding<T>,
    bool,
    Option<Box<Clamp>>,
);

/// `M = F · A_c · F` over its vertices, its surplus still to be decided.
pub(super) struct Congruent<'p, T, I> {
    vertices: &'p [I],
    congruence: Congruence<T>,
    off_diagonal: OffDiagonal<T, I>,
}

impl<'p, T: Real, I: Index> Congruent<'p, T, I> {
    /// `S A_c S`, unscaled.
    pub(super) fn signed(
        vertices: &'p [I],
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

    /// `S R A_c R S` with its `Σ_j |m_ij|`; `Err` at the stored row of the first scaled entry that leaves the range.
    pub(super) fn scaled(
        vertices: &'p [I],
        off_diagonal: OffDiagonal<T, I>,
        signs: Congruence<T>,
        scales: &[T],
    ) -> Result<(Self, Vec<T>), usize> {
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
        Ok((m, abs))
    }

    /// An incidence Gram, assembled switched across a bipartition.
    pub(super) fn incidence(vertices: &'p [I], off_diagonal: OffDiagonal<T, I>) -> Self {
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
    pub(super) fn vertices(&self) -> &'p [I] {
        self.vertices
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
        balance: Balance,
        grounding: Grounding<T>,
        vector: ScaleVector,
        clamped: f64,
    ) -> IrreducibleBlock<T, I> {
        let Self {
            vertices,
            congruence,
            off_diagonal,
        } = self;
        let n = vertices.len();
        assert!(congruence.fits(n), "one factor per vertex");
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        debug_assert!(
            balance == Balance::Frustrated || off_diagonal.nonpositive(),
            "a balanced M is a Z-matrix"
        );
        assert!(grounding.fits(n), "one surplus per row");
        IrreducibleBlock {
            congruence: Box::new(congruence),
            off_diagonal,
            balance,
            grounding,
            grounding_unproven: vector.grounding_unproven(),
            clamp: vector.clamp(clamped).map(Box::new),
        }
    }
}

impl<T: Real, I: Index> IrreducibleBlock<T, I> {
    /// Moved out, so each part is freed once its factor stops reading it.
    pub(crate) fn into_parts(self) -> BlockParts<T, I> {
        (
            self.congruence,
            self.off_diagonal,
            self.balance,
            self.grounding,
            self.grounding_unproven,
            self.clamp,
        )
    }
}

impl<T: Real, I: Index> KeptVertices for IrreducibleBlock<T, I> {
    fn kept(&self) -> Range<usize> {
        0..self.off_diagonal.n()
    }
}

type BlockParts<T, I> = (
    Box<Congruence<T>>,
    OffDiagonal<T, I>,
    Balance,
    Grounding<T>,
    bool,
    Option<Box<Clamp>>,
);
