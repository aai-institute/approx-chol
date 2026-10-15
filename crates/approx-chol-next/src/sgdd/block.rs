//! `A` as its reduced irreducible blocks: what the factorization consumes.

use std::sync::Arc;

use super::congruence::Congruence;
use super::decomposition::indexable;
use super::scaling::ScalingReport;
use crate::input::CsrError;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, Index, Real};

/// `R = F Pᵀ`, so `M = R A Rᵀ`: `Pᵀ` picks the block's vertices of `A`, `F` is the congruence.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Debug)]
pub(crate) struct Restriction<T, I> {
    vertices: Box<[I]>,
    congruence: Congruence<T>,
}

impl<T: Real, I: Index> Restriction<T, I> {
    pub(crate) fn vertices(&self) -> &[I] {
        &self.vertices
    }

    /// `local ← Pᵀ x`.
    pub(crate) fn gather(&self, x: &[T], local: &mut [T]) {
        (local.iter_mut().zip(self.vertices.iter())).for_each(|(v, &i)| *v = x[i.index()]);
    }

    /// `x ← P local` on the block's vertices, the rest untouched.
    pub(crate) fn scatter(&self, local: &[T], x: &mut [T]) {
        (local.iter().zip(self.vertices.iter())).for_each(|(&v, &i)| x[i.index()] = v);
    }

    /// `F`'s entries from `from` on, `None` when every `|f|` is `1`.
    pub(crate) fn scales(&self, from: usize) -> Option<&[T]> {
        match &self.congruence {
            Congruence::Scaled(factors, _) => Some(&factors[from..]),
            _ => None,
        }
    }

    /// `x ← F x` on a gathered `x`, `F·1` projected out first when `M` is singular.
    pub(crate) fn restrict(&self, x: &mut [T], singular: bool) {
        if singular {
            self.congruence.project_out_null(x);
        }
        self.congruence.apply(x);
    }

    /// `x ← F x` back in `A_c`'s scale, `F·1` projected out last when `M` is singular.
    pub(crate) fn prolongate(&self, x: &mut [T], singular: bool) {
        self.congruence.apply(x);
        if singular {
            self.congruence.project_out_null(x);
        }
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

    /// Copies of each vertex its factor holds: a cover has two.
    fn sheets(&self) -> usize {
        match self {
            Self::Balanced(_) => 1,
            Self::Frustrated(_) => 2,
        }
    }

    /// The ground is a vertex too: its diagonal `Σs`, once per sheet, clears the same bound as any other.
    fn ground_in_range(&self) -> bool {
        let sheets = T::from(self.sheets()).expect("a sheet count is a float");
        let total = |surplus: &[T]| surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        (self.grounding().surplus())
            .is_none_or(|surplus| diagonal_in_range(sheets * total(surplus)))
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
pub struct IrreducibleBlock<T = f64, I = u32> {
    restriction: Arc<Restriction<T, I>>,
    off_diagonal: OffDiagonal<T, I>,
    balance: Balance<T>,
}

impl<T: Real, I: Index> IrreducibleBlock<T, I> {
    /// Panics unless one factor, row and surplus per vertex; the ground's `Σs` must clear the bound any diagonal does, and `I` must index the factor's vertices.
    pub(super) fn new(
        vertices: Box<[I]>,
        congruence: Congruence<T>,
        off_diagonal: OffDiagonal<T, I>,
        balance: Balance<T>,
    ) -> Result<Self, ReduceError> {
        let n = vertices.len();
        assert!(congruence.fits(n), "one factor per vertex");
        assert_eq!(off_diagonal.n(), n, "one row per vertex");
        assert!(balance.grounding().fits(n), "one surplus per row");
        if let Balance::Balanced(_) = balance {
            debug_assert!(off_diagonal.nonpositive(), "a balanced M is a Z-matrix");
        }
        // Each vertex once per sheet, and the ground.
        indexable::<I>(n.saturating_mul(balance.sheets()).saturating_add(1))?;
        let surplus = balance.grounding().surplus().unwrap_or_default();
        if let Some(i) = surplus.iter().position(|&s| !crate::in_range(s)) {
            let vertex = vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        if !balance.ground_in_range() {
            let vertex = vertices[0].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        let restriction = Restriction {
            vertices,
            congruence,
        };
        Ok(Self {
            restriction: Arc::new(restriction),
            off_diagonal,
            balance,
        })
    }

    /// The block's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[I] {
        &self.restriction.vertices
    }

    /// Present iff this block's scaling was clamped beyond the policy's tolerance or did not settle.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.restriction.congruence.report()
    }

    /// Shared with every factor of this block.
    pub(crate) fn restriction(&self) -> &Arc<Restriction<T, I>> {
        &self.restriction
    }

    /// Whether `A_c` is folded and solved as singular, `A_c⁺b`: before any clamp its `M` has `1ᵀM1 ≤ 5ε·tr M`, so a net deficit floats; a Laplacian's `g` or a signed graph's cancelled mass must be exactly `0`.
    pub fn floats(&self) -> bool {
        matches!(&self.balance, Balance::Balanced(grounding) if !grounding.is_grounded())
    }

    pub(crate) fn off_diagonal(&self) -> &OffDiagonal<T, I> {
        &self.off_diagonal
    }

    pub(crate) fn balance(&self) -> &Balance<T> {
        &self.balance
    }
}

/// `A` as its [`IrreducibleBlock`]s, each reduced; vertices in none solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64, I = u32> {
    n: usize,
    blocks: Vec<IrreducibleBlock<T, I>>,
    /// Lone vertices with a zero diagonal, recorded so a solve zeroes them without a pass over `n`.
    dead: Box<[I]>,
}

impl<T: Real, I: Index> Sgdd<T, I> {
    pub(super) fn new(n: usize, blocks: Vec<IrreducibleBlock<T, I>>, dead: Vec<I>) -> Self {
        Self {
            n,
            blocks,
            dead: dead.into_boxed_slice(),
        }
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The irreducible blocks, each factored and solved on its own.
    pub fn blocks(&self) -> &[IrreducibleBlock<T, I>] {
        &self.blocks
    }

    pub(crate) fn dead(&self) -> &[I] {
        &self.dead
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
    /// More vertices than the index type `I` holds below its `MAX`: all of `A`'s, or a block's factor's, a cover's two sheets and its ground counted.
    IndexOverflow {
        /// The vertices to index.
        count: usize,
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
            Self::IndexOverflow { count } => {
                write!(f, "{count} vertices exceed the index type")
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

/// Why a CSR did not become an [`Sgdd`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromCsrError {
    /// The arrays are not a symmetric matrix.
    Csr(CsrError),
    /// A block has no SDDM form.
    Reduce(ReduceError),
}

impl From<CsrError> for FromCsrError {
    fn from(error: CsrError) -> Self {
        Self::Csr(error)
    }
}

impl From<ReduceError> for FromCsrError {
    fn from(error: ReduceError) -> Self {
        Self::Reduce(error)
    }
}

impl core::fmt::Display for FromCsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cannot convert CSR to Sgdd")
    }
}

impl std::error::Error for FromCsrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Csr(error) => Some(error),
            Self::Reduce(error) => Some(error),
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Restriction);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Restriction<T, I> {
    fn checked(self) -> Option<Self> {
        let factors = match &self.congruence {
            Congruence::Signs(f) => f.iter().all(|f| f.abs() == T::one()),
            Congruence::Scaled(f, _) => f.iter().all(|f| f.is_finite() && *f != T::zero()),
            Congruence::Identity | Congruence::NegateFrom(_) => true,
        };
        let fits = !self.vertices.is_empty() && self.congruence.fits(self.vertices.len());
        (fits && factors).then_some(self)
    }
}
