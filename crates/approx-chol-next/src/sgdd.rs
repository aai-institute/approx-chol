//! `A` as its connected components, each reduced to `M = F · A_c · F`, decided once per input.

mod component;
mod congruence;
mod decomposition;
mod scaling;
mod signature;

use std::sync::Arc;

use num_traits::PrimInt;

pub use component::Component;
use component::Sdd;
use congruence::Congruence;
use decomposition::Decomposition;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::input::{BipartiteGram, BipartiteGraph, Csr, CsrError, FactorWidth2};
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
    fn new(vertices: Box<[u32]>, congruence: Congruence<T>) -> Self {
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
    /// Folded to a Z-matrix: `M` itself, singular exactly when ungrounded, which reduction allows only on proof.
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

/// What closes `L` into an SDDM: a surplus `s = diag(M) − Σ|offdiag|` with a positive total, or none, leaving it singular.
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

    fn ungrounded() -> Self {
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
    fn n(&self) -> usize {
        match self {
            Self::General(adjacency) => adjacency.n(),
            Self::Bipartite(cross) => cross.n(),
        }
    }

    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    fn abs_row_sums(&self) -> Vec<T> {
        match self {
            Self::General(adjacency) => adjacency.abs_row_sums(),
            Self::Bipartite(cross) => cross.abs_row_sums(),
        }
    }

    fn nonpositive(&self) -> bool {
        match self {
            Self::General(adjacency) => adjacency.nonpositive(),
            Self::Bipartite(cross) => cross.nonpositive(),
        }
    }

    /// Entries per row in local order: the rounding a row sum can carry.
    fn degrees(&self) -> Vec<usize> {
        match self {
            Self::General(adjacency) => adjacency.degrees(),
            Self::Bipartite(cross) => cross.degrees(),
        }
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

    /// Checks symmetry, then scales each component to dominance unless it already is; an unscaled row's surplus up to about `2tε·a_ii` (`t` its entries) counts as rounding, and a scaled component floats only if its clamped surplus vanishes.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(Decomposition::csr(csr)?.reduce(&policy)?)
    }

    /// `A = V Vᵀ`: scales, folds or covers each component as general, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        v: FactorWidth2<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Decomposition::gram(v)?.reduce(&policy)
    }

    /// `A = V W Vᵀ` on two disjoint sides: as [`from_gram`](Self::from_gram), a bipartite component then taking the block elimination.
    pub fn from_bipartite_gram<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Decomposition::bipartite_gram(gram)?.reduce(&policy)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless bipartite Laplacian has zero surplus by shape: every component floats untested.
    fn try_from(graph: BipartiteGraph<'a, T, I>) -> Result<Self, Self::Error> {
        Decomposition::signless(graph)?.reduce(&ScalingPolicy::default())
    }
}

/// One component `A_c` as the input gave it: signed, unscaled, in the input's shape.
struct Unreduced<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
    /// `Σ_j |a_ij|`, summed in `f64` while its rows are extracted, so dominance and surplus reread no value and a long `f32` row is not undercounted.
    abs_row_sums: Vec<T>,
    /// No positive off-diagonal: already a Z-matrix, so no signature is searched.
    nonpositive: bool,
    /// Entries per row: the rounding its sum can carry, read by the dominance test and the surplus band alike.
    degrees: Vec<usize>,
}

/// Who proves a component singular: its own rows, or its input once every component is reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proof {
    Rows,
    Input,
}

impl<T: Real> Unreduced<T> {
    /// Panics unless one vertex and one diagonal per row.
    fn new(vertices: Box<[u32]>, off_diagonal: OffDiagonal<T>, diagonal: Vec<T>) -> Self {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(diagonal.len(), off_diagonal.n(), "one diagonal per row");
        Self {
            vertices,
            abs_row_sums: off_diagonal.abs_row_sums(),
            nonpositive: off_diagonal.nonpositive(),
            degrees: off_diagonal.degrees(),
            off_diagonal,
            diagonal,
        }
    }

    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    fn reduce(self, policy: &ScalingPolicy, proof: Proof) -> Result<Congruent<T>, ReduceError> {
        within_series_floor::<T>(&self.vertices)?;
        // A view's zero diagonal beside a cell is its terms underflowing, not a non-GDD input.
        let lost = self.diagonal.iter().position(|&d| d == T::zero());
        if let (Proof::Input, Some(i)) = (proof, lost) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let scaling = scaling::dominance_scaling(&self, policy)?;
        let signs = match self.nonpositive {
            true => Some(Congruence::identity()),
            false => signature::signature(&self.off_diagonal),
        };
        self.into_congruent(scaling, signs, policy)
    }

    /// `F = S R` folds and scales `A_c`, or only scales a frustrated one.
    fn into_congruent(
        self,
        scaling: scaling::Scaling<T>,
        signs: Option<Congruence<T>>,
        policy: &ScalingPolicy,
    ) -> Result<Congruent<T>, ReduceError> {
        let vertex = self.vertices[0];
        let folds = signs.is_some();
        let signs = signs.unwrap_or_else(Congruence::identity);
        let (congruence, relaxation) = match scaling {
            scaling::Scaling::AlreadyDominant => (signs, None),
            scaling::Scaling::Scaled(relaxation) => {
                (signs.scaled(&relaxation.scales), Some(relaxation))
            }
        };
        let off_diagonal =
            (congruence.transform(self.off_diagonal)).ok_or(ReduceError::Unscalable { vertex })?;
        // Signs keep every magnitude, so only a scaling sums `M`'s rows again.
        let (rows, report) = match relaxation {
            None => {
                let rows = RowSums {
                    diagonal: self.diagonal,
                    abs: self.abs_row_sums,
                    surplus: RowSurplus::Banded(self.degrees),
                };
                // An unscaled deficit lies inside the rounding band `already_dominant` allowed.
                (rows, None)
            }
            Some(relaxation) => {
                let diagonal: Vec<T> = (self.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let abs = off_diagonal.abs_row_sums();
                let deficit = scaling::clamped_deficit(&diagonal, &abs);
                let rows = RowSums {
                    diagonal,
                    abs,
                    surplus: RowSurplus::Residue,
                };
                (rows, policy.certify(&relaxation, deficit))
            }
        };
        Ok(Congruent {
            frame: Frame::new(self.vertices, congruence),
            off_diagonal,
            folds,
            rows,
            report,
        })
    }
}

/// `M = F · A_c · F` before its grounding is decided: by its rows, or for a folded view component by its input.
pub(super) struct Congruent<T> {
    frame: Frame<T>,
    off_diagonal: OffDiagonal<T>,
    folds: bool,
    rows: RowSums<T>,
    report: Option<ScalingReport>,
}

/// `M`'s diagonal and `Σ_j |m_ij|`, read only when the rows decide the surplus.
struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
    surplus: RowSurplus,
}

/// How a row's `m_ii − Σ_j |m_ij|` reads as surplus.
enum RowSurplus {
    /// Unscaled: inside the band its entries per row allow, it is rounding and reads zero.
    Banded(Vec<usize>),
    /// Scaled: the scaling's residue, which no row bound tells from a real surplus, deficits clamped to zero.
    Residue,
}

impl<T: Real> Congruent<T> {
    pub(super) fn frame(&self) -> &Frame<T> {
        &self.frame
    }

    pub(super) fn folds(&self) -> bool {
        self.folds
    }

    /// The surplus `M`'s own rows show.
    pub(super) fn by_rows(self) -> Result<Component<T>, ReduceError> {
        let rows = self.rows.diagonal.iter().zip(&self.rows.abs);
        let surplus = match &self.rows.surplus {
            RowSurplus::Banded(degrees) => (rows.zip(degrees))
                .map(|((&d, &s), &degree)| scaling::unscaled_surplus(d, s, degree))
                .collect(),
            RowSurplus::Residue => rows.map(|(&d, &s)| (d - s).max(T::zero())).collect(),
        };
        self.grounded(Grounding::of(surplus))
    }

    /// The ground's `Σs` must clear the bound any diagonal does.
    pub(super) fn grounded(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        let vertex = self.frame.vertices()[0];
        let balance = match self.folds {
            true => Balance::Balanced(grounding),
            false => Balance::Frustrated(grounding),
        };
        if !balance.ground_in_range() {
            return Err(ReduceError::OutOfRange { vertex });
        }
        Ok(Component::new(
            self.frame,
            Sdd::new(self.off_diagonal, balance),
            self.report,
        ))
    }
}

/// Past `1/ε` vertices, floor weights composed in series fall below `MIN_POSITIVE` (f32: 2^23).
fn within_series_floor<T: Real>(vertices: &[u32]) -> Result<(), ReduceError> {
    match T::from(vertices.len()).is_none_or(|n| n * T::epsilon() >= T::one()) {
        true => Err(ReduceError::Unscalable {
            vertex: vertices[0],
        }),
        false => Ok(()),
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
    /// A scale or a scaled entry overflowed or fell nonzero below `MIN_POSITIVE/EPSILON`, or the component passes `1/ε` vertices.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// An assembled entry is non-finite or nonzero below `MIN_POSITIVE/EPSILON`, a diagonal, a row's `Σ|a_ij|` or the ground's `Σs` exceeds `MAX/4`, or a view's diagonal underflowed to zero.
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

/// Why a CSR did not become an [`Sgdd`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromCsrError {
    /// The arrays are not a symmetric matrix.
    Csr(CsrError),
    /// A component has no SDDM form.
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
