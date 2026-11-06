mod approximate;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::Multi;
use super::sddm::{descend, first_copy, lift, Sddm};
use crate::sgdd::FactorTail;
use crate::storage::{compensated_sum, narrow, widen, Index, Real, Sheets};
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
    /// The ground is the appended slot, pinned so it reads zero; its right-hand side `−Σb` and the shift serve only factors saved before the ground was always pinned.
    grounded: bool,
    /// The matrix's dimension over its sheets, ground excluded.
    n: usize,
    /// The one slot no pivot eliminates, zeroed between the sweeps.
    pinned: usize,
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
        let (sheets, grounded, n, pinned, triangular) = match arm {
            Arm::Exact(signed) => {
                let (grounded, n) = (!signed.floats(), signed.n());
                // A definite `M`'s surplus is the ground's slot, pinned so the solve is `M`'s own.
                let pin = pinned(Sheets::One, grounded, n, tail);
                let factor =
                    Triangular::Dense(dense::factor(&signed, pin, &mut reused.dense).into());
                (Sheets::One, grounded, n, pin, factor)
            }
            Arm::Approximate(sddm) => {
                let (sheets, grounded, n) = (sddm.sheets(), sddm.grounded(), sddm.n());
                // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
                // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
                let pin = (grounded || tail.is_some_and(FactorTail::spans_wide))
                    .then(|| pinned(sheets, grounded, n, tail));
                let (steps, pin) = reused.elimination.eliminate(sddm, draws, pin);
                let triangular = Triangular::Approximate(Box::new(steps));
                (sheets, grounded, n, pin, triangular)
            }
        };
        Self {
            sheets,
            grounded,
            n,
            pinned,
            triangular,
        }
    }

    /// Pivots eliminated: every slot but the pinned one, which is the ground when there is one.
    pub(super) fn n_steps(&self) -> usize {
        self.n + usize::from(self.grounded) - 1
    }

    /// Whether it solves a singular Laplacian: one sheet, no ground.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && !self.grounded
    }

    /// A floating sheet substitutes on `x` in place; a grounded one appends the ground's slot beside the rows.
    pub(super) fn scratch_len(&self) -> usize {
        match (self.sheets, self.grounded) {
            (Sheets::One, false) => 0,
            (_, true) => self.n + 1,
            (Sheets::Two, false) => self.n,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        let triangular = &self.triangular;
        match (self.sheets, self.grounded) {
            (Sheets::One, false) => triangular.substitute(x, self.pinned),
            (Sheets::One, true) => {
                let rows = &mut scratch[..=x.len()];
                let (ground, values) = rows.split_last_mut().expect("the ground's slot");
                values.copy_from_slice(x);
                let total = compensated_sum(x, |_, b| widen(b));
                *ground = narrow::<T>(-total);
                triangular.substitute(rows, self.pinned);
                let (ground, values) = rows.split_last().expect("the ground's slot");
                (x.iter_mut().zip(values)).for_each(|(x, &v)| *x = v - *ground);
            }
            (Sheets::Two, _) => {
                let rows = &mut scratch[..self.scratch_len()];
                lift(x, rows);
                triangular.substitute(rows, self.pinned);
                descend(rows, x);
            }
        }
    }
}

/// The slot the solve pins: the ground, else the largest `|f|` on the first sheet, so the gauge's rounding is not amplified by `1/f`.
fn pinned<T: Real>(
    sheets: Sheets,
    grounded: bool,
    n: usize,
    tail: Option<FactorTail<'_, T>>,
) -> usize {
    match grounded {
        true => n,
        false => tail.map_or(n - 1, |tail| first_copy(sheets, tail.largest())),
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
    fn substitute(&self, values: &mut [T], pinned: usize) {
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
        let slots = self.n.checked_add(usize::from(self.grounded))?;
        let fits = match &self.triangular {
            Triangular::Dense(dense) => dense::fits(dense, slots),
            Triangular::Approximate(approximate) => approximate.fits(slots, self.pinned),
        };
        (self.pinned < slots && fits).then_some(self)
    }

    /// Whether it solves a block of `kept` vertices on its sheets.
    pub(super) fn solves(&self, kept: usize) -> bool {
        Some(self.n) == self.sheets.count().checked_mul(kept)
    }
}
