mod approximate;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::Multi;
use super::sddm::{descend, lifted, Sddm, SignedWdd};
#[cfg(feature = "serde")]
use super::sign::{Signed, Unsigned};
use crate::sgdd::{Balance, FactorTail};
use crate::{Batch, Index, MemStack, Real};
pub(super) use approximate::Eliminated;
use approximate::{ApproximateCholesky, CliqueTreeElimination};
pub(super) use dense::AsIs;

/// A task's buffers, reused across its blocks by either arm.
pub(crate) struct Workspace<T, I> {
    elimination: CliqueTreeElimination<T, I>,
    /// The dense arm's working matrix: allocated per block, it cost 8% of building 2–5 vertex blocks.
    dense: Vec<T>,
}

impl<T: Real, I: Index> Workspace<T, I> {
    pub(super) fn new(copies: Option<Multi>) -> Self {
        Self {
            elimination: CliqueTreeElimination::new(copies),
            dense: Vec::new(),
        }
    }
}

/// What a Cholesky factors: `M` as it is, factored densely; its SDDM; or frustrated `M` signed, both factored approximately.
pub(super) enum Arm<T, I> {
    AsIs(AsIs<T, I>),
    Sddm(Sddm<T, I>),
    Signed(SignedWdd<T, I>),
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T, I> {
    /// The matrix's dimension, a cover's two sheets counted.
    n: usize,
    triangular: Triangular<T, I>,
}

impl<T: Real, I: Index> Cholesky<T, I> {
    /// Dense on `M` as it is, approximate on its SDDM or signed; `tail` is `F` on its vertices, `None` when every `|f|` is `1`.
    pub(super) fn factor(
        arm: Arm<T, I>,
        draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        match arm {
            Arm::AsIs(as_is) => {
                let (factor, pinned) = dense::factor(&as_is, tail, &mut reused.dense);
                Self {
                    n: as_is.n(),
                    triangular: Triangular::Dense(factor.into(), pinned),
                }
            }
            Arm::Sddm(sddm) => {
                let (balance, n) = (sddm.balance(), sddm.n());
                let (steps, pinned) = reused.elimination.eliminate(sddm, draws, tail);
                let steps = Box::new(steps);
                let triangular = match balance {
                    Balance::Balanced => Triangular::Folded(steps, pinned),
                    Balance::Frustrated => Triangular::Cover(steps, pinned),
                };
                Self { n, triangular }
            }
            Arm::Signed(m) => {
                let n = m.n();
                let (steps, _) = reused.elimination.eliminate(m, draws, tail);
                Self {
                    n,
                    triangular: Triangular::Signed(Box::new(steps)),
                }
            }
        }
    }

    /// Pivots eliminated: every slot but a floating matrix's pinned one.
    pub(super) fn n_steps(&self) -> usize {
        self.n - usize::from(self.triangular.pinned().is_some())
    }

    /// Whether it solves a singular Laplacian: one floating sheet.
    pub(super) fn singular(&self) -> bool {
        self.triangular.sheets() == 1 && self.triangular.pinned().is_some()
    }

    /// Slots [`solve`](Self::solve) takes: none on one sheet, which substitutes on `x` in place; a cover lifts `x` to rows of their own.
    pub(super) fn scratch_slots(&self) -> usize {
        match self.triangular {
            Triangular::Dense(..) | Triangular::Folded(..) | Triangular::Signed(_) => 0,
            Triangular::Cover(..) => self.n,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve<V: Batch<T>>(&self, x: &mut [V], stack: &mut MemStack) {
        match &self.triangular {
            Triangular::Dense(factor, pinned) => dense::substitute(factor, *pinned, x),
            Triangular::Folded(steps, pinned) => steps.substitute(x, *pinned),
            Triangular::Signed(steps) => steps.substitute(x, None),
            Triangular::Cover(steps, pinned) => {
                let (mut rows, _) = stack.make_with(self.n, |i| lifted(x, i));
                steps.substitute(&mut rows, *pinned);
                descend(&rows, x);
            }
        }
    }
}

/// The triangular factor: dense on one sheet, or approximate on one, a cover's two, or signed on one; each but the signed with a floating matrix's one slot no pivot eliminates, zeroed between the sweeps.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
enum Triangular<T, I> {
    /// Packed unit lower `L` of `L D Lᵀ`, `D⁻¹` on its diagonal.
    Dense(Box<[T]>, Option<usize>),
    Folded(Box<ApproximateCholesky<T, I>>, Option<usize>),
    Cover(Box<ApproximateCholesky<T, I>>, Option<usize>),
    /// Frustrated `M` eliminated as it is, its shares of either sign; its ground takes what it cancels, so it never floats.
    Signed(Box<ApproximateCholesky<T, I>>),
}

impl<T, I> Triangular<T, I> {
    /// A floating matrix's pinned slot; a definite one pivots every slot.
    fn pinned(&self) -> Option<usize> {
        match self {
            Self::Dense(_, pinned) | Self::Folded(_, pinned) | Self::Cover(_, pinned) => *pinned,
            Self::Signed(_) => None,
        }
    }

    /// Copies of each vertex it factors: a dense factor's balance left none but its one sheet.
    fn sheets(&self) -> usize {
        match self {
            Self::Dense(..) | Self::Folded(..) | Self::Signed(_) => 1,
            Self::Cover(..) => 2,
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Cholesky);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Cholesky<T, I> {
    fn checked(self) -> Option<Self> {
        if self.n == 0 {
            return None;
        }
        let pinned = self.triangular.pinned();
        let fits = match &self.triangular {
            Triangular::Dense(dense, _) => dense::fits(dense, self.n_steps()),
            Triangular::Folded(steps, _) | Triangular::Cover(steps, _) => {
                steps.fits::<Unsigned>(self.n, pinned)
            }
            Triangular::Signed(steps) => steps.fits::<Signed<I>>(self.n, None),
        };
        (pinned.is_none_or(|pin| pin < self.n) && fits).then_some(self)
    }

    /// Whether it solves a block of `kept` vertices on its sheets.
    pub(super) fn solves(&self, kept: usize) -> bool {
        Some(self.n) == kept.checked_mul(self.triangular.sheets())
    }
}
