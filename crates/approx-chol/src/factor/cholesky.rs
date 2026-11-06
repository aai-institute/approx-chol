mod approximate;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::Multi;
use super::sddm::{descend, lift, Sddm};
use crate::sgdd::{FactorTail, Sheets};
use crate::{Batch, Index, Real};
use approximate::{ApproximateCholesky, CliqueTreeElimination};
pub(super) use dense::Signed;

/// A task's buffers, reused across its blocks by either arm.
pub(crate) struct Workspace<T, I> {
    elimination: CliqueTreeElimination<T, I>,
    /// The dense arm's working matrix: allocated per block, it cost 8% of building 2–5 vertex blocks.
    dense: Vec<T>,
}

impl<T: Real, I: Index> Workspace<T, I> {
    pub(super) fn new(split: Option<Multi>) -> Self {
        Self {
            elimination: CliqueTreeElimination::new(split),
            dense: Vec::new(),
        }
    }
}

/// What a Cholesky factors: `M` itself on the exact arm, its SDDM on the approximate one.
pub(super) enum Arm<T, I> {
    Exact(Signed<T, I>),
    Approximate(Sddm<T, I>),
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T, I> {
    sheets: Sheets,
    /// The matrix's dimension over its sheets.
    n: usize,
    /// A floating matrix's one slot no pivot eliminates, zeroed between the sweeps; a definite one pivots every slot.
    pinned: Option<usize>,
    triangular: Triangular<T, I>,
}

impl<T: Real, I: Index> Cholesky<T, I> {
    /// Dense on the exact arm, else approximate; `tail` is `F` on its vertices, `None` when every `|f|` is `1`.
    pub(super) fn factor(
        arm: Arm<T, I>,
        draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        match arm {
            Arm::Exact(signed) => {
                let (factor, pinned) = dense::factor(&signed, tail, &mut reused.dense);
                Self {
                    sheets: Sheets::One,
                    n: signed.n(),
                    pinned,
                    triangular: Triangular::Dense(factor.into()),
                }
            }
            Arm::Approximate(sddm) => {
                let (sheets, n) = (sddm.sheets(), sddm.n());
                let (steps, pinned) = reused.elimination.eliminate(sddm, draws, tail);
                Self {
                    sheets,
                    n,
                    pinned,
                    triangular: Triangular::Approximate(Box::new(steps)),
                }
            }
        }
    }

    /// Pivots eliminated: every slot but a floating matrix's pinned one.
    pub(super) fn n_steps(&self) -> usize {
        self.n - usize::from(self.pinned.is_some())
    }

    /// Whether it solves a singular Laplacian: one floating sheet.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && self.pinned.is_some()
    }

    /// One sheet substitutes on `x` in place; two lift `x` to rows of their own.
    pub(super) fn scratch_len(&self) -> usize {
        match self.sheets {
            Sheets::One => 0,
            Sheets::Two => self.n,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve<V: Batch<T>>(&self, x: &mut [V], scratch: &mut [V]) {
        let triangular = &self.triangular;
        match self.sheets {
            Sheets::One => triangular.substitute(x, self.pinned),
            Sheets::Two => {
                let rows = &mut scratch[..self.scratch_len()];
                lift(x, rows);
                triangular.substitute(rows, self.pinned);
                descend(rows, x);
            }
        }
    }
}

/// The triangular factor, dense or approximate.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
enum Triangular<T, I> {
    /// Packed unit lower `L` of `L D Lᵀ`, `D⁻¹` on its diagonal.
    Dense(Box<[T]>),
    Approximate(Box<ApproximateCholesky<T, I>>),
}

impl<T: Real, I: Index> Triangular<T, I> {
    fn substitute<V: Batch<T>>(&self, values: &mut [V], pinned: Option<usize>) {
        match self {
            Self::Dense(factor) => dense::substitute(factor, pinned, values),
            Self::Approximate(approximate) => approximate.substitute(values, pinned),
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
        let fits = match &self.triangular {
            Triangular::Dense(dense) => dense::fits(dense, self.n_steps()),
            Triangular::Approximate(approximate) => approximate.fits(self.n, self.pinned),
        };
        (self.pinned.is_none_or(|pin| pin < self.n) && fits).then_some(self)
    }

    /// Whether it solves a block of `kept` vertices on its sheets.
    pub(super) fn solves(&self, kept: usize) -> bool {
        Some(self.n) == self.sheets.count().checked_mul(kept)
    }
}
