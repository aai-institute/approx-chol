mod approximate;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;

use super::config::Config;
use super::sddm::{descend, lift, Sddm, Sheets};
use crate::{largest, narrow};
use crate::{widen, Index, Real};
use approximate::ApproximateCholesky;
pub(super) use approximate::Elimination;

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T, I> {
    sheets: Sheets,
    /// The ground is the appended slot, right-hand side `−Σb`, every entry then shifted so it reads zero; else the pinned slot reads zero.
    grounded: bool,
    /// The matrix's dimension over its sheets, ground excluded.
    n: usize,
    /// The one slot no pivot eliminates, zeroed between the sweeps.
    pinned: usize,
    triangular: Triangular<T, I>,
}

impl<T: Real, I: Index> Cholesky<T, I> {
    /// Dense up to the config's dimension, else approximate; `factors` is `F` on its vertices, `None` when every `|f|` is `1`.
    pub(super) fn factor(
        sddm: Sddm<T, I>,
        draws: Xoshiro256PlusPlus,
        config: &Config,
        factors: Option<&[T]>,
        elimination: &mut Elimination<T, I>,
    ) -> Self {
        let (sheets, grounded, n) = (sddm.sheets(), sddm.grounded(), sddm.n());
        let (triangular, pinned) = match config.dense(n) {
            true => {
                let pin = pinned(&sddm, factors);
                (Triangular::Dense(dense::factor(&sddm, pin).into()), pin)
            }
            false => {
                // Ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
                let pin =
                    (spans_wide(factors) || grounds_wide(&sddm)).then(|| pinned(&sddm, factors));
                let (steps, pin) = elimination.factor(sddm, draws, pin);
                (Triangular::Approximate(Box::new(steps)), pin)
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

    /// Pivots eliminated: every slot, the ground's included, but the pinned one.
    pub(super) fn n_steps(&self) -> usize {
        self.n + usize::from(self.grounded) - 1
    }

    /// Whether it solves a singular Laplacian: one sheet, no ground.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && !self.grounded
    }

    /// A floating sheet substitutes on `x` in place; the approximate order may read the ground mid-recurrence, so it sits beside the rows.
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
                let total = crate::compensated_sum(x.iter().map(|&b| widen(b)));
                *ground = narrow::<T>(-total);
                triangular.substitute(rows, self.pinned);
                let (ground, values) = rows.split_last().expect("the ground's slot");
                (x.iter_mut().zip(values)).for_each(|(x, &v)| *x = v - *ground);
            }
            (Sheets::Two, _) => {
                let rows = &mut scratch[..self.scratch_len()];
                lift(x, rows);
                // A lift sums to zero, so the ground's right-hand side is zero and its shift cancels in the descent.
                rows[2 * x.len()..].fill(T::zero());
                triangular.substitute(rows, self.pinned);
                descend(rows, x);
            }
        }
    }
}

/// The slot the solve pins: the ground, else the largest `|f|` on the first sheet, so the gauge's rounding is not amplified by `1/f`.
fn pinned<T: Real, I: Index>(sddm: &Sddm<T, I>, factors: Option<&[T]>) -> usize {
    match sddm.grounded() {
        true => sddm.n(),
        false => factors.map_or(sddm.n() - 1, |f| sddm.sheets().count() * largest(f)),
    }
}

/// Whether `|f|` spans past `1/√ε`, where a solve pinned at a small `|f|` loses more than half its digits.
fn spans_wide<T: Real>(factors: Option<&[T]>) -> bool {
    factors.is_some_and(|f| {
        let (least, most) = (f.iter()).fold((T::infinity(), T::zero()), |(least, most), f| {
            (least.min(f.abs()), most.max(f.abs()))
        });
        least < most * T::epsilon().sqrt()
    })
}

/// A sink behind an edge past `1/ε` lighter than the heaviest would divide the ground's `−Σb` rounding.
fn grounds_wide<T: Real, I: Index>(sddm: &Sddm<T, I>) -> bool {
    let (n, mut least, mut most) = (sddm.n(), T::infinity(), T::zero());
    if sddm.grounded() && sddm.sheets() == Sheets::One {
        sddm.grounded_edges(|_, j, w| {
            most = most.max(w);
            if j.index() < n {
                least = least.min(w);
            }
        });
    }
    least < most * T::epsilon()
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
