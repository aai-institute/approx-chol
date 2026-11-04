mod approximate;
mod block_elimination;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use super::clique_tree::Multi;
use super::config::{Config, InnerParallelism};
use super::schur::BipartiteBlock;
use super::sddm::{descend, first_copy, lift, Sddm};
use crate::storage::{
    compensated_sum, narrow, widen, Adjacency, Balance, Bipartite, Gauge, Grounded, Index,
    OffDiagonal, Real, Sheets,
};
use approximate::{ApproximateCholesky, CliqueTreeElimination};
pub(super) use block_elimination::BlockElimination;
pub(super) use dense::Signed;

/// Whether `u32` indexes every slot of `n` vertices on the widest route, a cover's two sheets and its ground; the reduction is made before the route is known.
pub(crate) fn fits_u32(n: usize) -> bool {
    u32::try_from(n.saturating_mul(Sheets::Two.count()).saturating_add(1)).is_ok()
}

/// `log₂` of the multiply-adds a block elimination's task carries, below which a split costs more than it saves.
pub(super) const MULTIPLY_ADD_GRAIN_BITS: u32 = 17;

/// `Σ|t| ≤ 1` up to the terms' rounding: a substitution through such multipliers cannot overflow where its solution fits.
fn within_one<T: Real>(terms: impl Iterator<Item = T>) -> bool {
    let (sum, len) = terms.fold((0.0, 0), |(sum, len), t| (sum + widen(t.abs()), len + 1));
    sum <= 1.0 + 4.0 * (len + 2) as f64 * widen(T::epsilon())
}

/// One block's key, so no draw depends on thread scheduling or on how other blocks routed.
#[derive(Clone, Copy, Debug)]
pub(super) struct Streams {
    /// The seed and the block mixed once: every star would otherwise repeat them on its first draw's path.
    key: u64,
}

impl Streams {
    /// Keyed by the block's first vertex in `A`, so a complement eliminating the side its whole factor does draws as that factor does.
    fn of(first: usize, config: &Config) -> Self {
        Self {
            key: splitmix(splitmix(config.seed) ^ first as u64),
        }
    }

    /// The approximate Cholesky's one sequence: pivot `t`'s sample is conditioned on every earlier one.
    fn cholesky(self) -> Xoshiro256PlusPlus {
        rng(self, 0)
    }

    /// Star `eliminated`'s own sequence, by its index in the block, never by task.
    pub(super) fn star(self, eliminated: usize) -> Xoshiro256PlusPlus {
        rng(self, eliminated as u64 + 1)
    }
}

/// The only place a key becomes draws; stage `0` is the Cholesky's, `e + 1` star `e`'s.
fn rng(streams: Streams, stage: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(splitmix(streams.key ^ stage))
}

/// SplitMix64's finalizer: each part changes every bit of the key, so no two keys collide by addition.
fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A component's factorization: a bipartite one's block elimination, then the Cholesky of what it leaves.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize, I: Index + serde::Serialize",
        deserialize = "T: Real + serde::Deserialize<'de>, I: Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(super) struct Factorization<T, I> {
    pub(super) elimination: Option<Box<BlockElimination<T, I>>>,
    pub(super) cholesky: Cholesky<T, I>,
}

impl<T: Real, I: Index> Factorization<T, I> {
    /// Dense or approximate as `config` sizes it; a bipartite component eliminates its larger side first.
    pub(super) fn component(
        matrix: Grounded<T, OffDiagonal<T, I>>,
        balance: Balance,
        gauge: Gauge,
        key: usize,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let Grounded {
            adjacency,
            grounding,
        } = matrix;
        let streams = Streams::of(key, config);
        let sheets = balance.sheets();
        let draws = streams.cholesky();
        match adjacency {
            OffDiagonal::General(adjacency) => {
                let arm = match config.dense(adjacency.n()) {
                    true => Arm::Exact(Signed::new(adjacency, sheets, grounding)),
                    false => Arm::Approximate(Sddm::of_block(adjacency, sheets, grounding)),
                };
                Self {
                    elimination: None,
                    cholesky: Cholesky::factor(arm, draws, gauge, reused),
                }
            }
            OffDiagonal::Bipartite(cross) => {
                let factor = |s| Cholesky::factor(s, draws, gauge, reused);
                let block = BipartiteBlock::of(cross.into_cross(), sheets, grounding);
                let (elimination, cholesky) = block.eliminate(config, streams, factor);
                Self {
                    elimination: Some(Box::new(elimination)),
                    cholesky,
                }
            }
        }
    }

    /// The Schur complement onto the kept side, keyed as the whole component keys its draws.
    pub(super) fn complement(
        matrix: Grounded<T, Bipartite<T, I>>,
        balance: Balance,
        gauge: Gauge,
        key: usize,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let Grounded {
            adjacency: cross,
            grounding,
        } = matrix;
        let streams = Streams::of(key, config);
        let block = BipartiteBlock::of(cross.into_cross(), balance.sheets(), grounding);
        let s = block.complement(config, streams);
        Self {
            elimination: None,
            cholesky: Cholesky::factor(s, streams.cholesky(), gauge, reused),
        }
    }

    /// Every slot but one, eliminated ones and the Cholesky's alike.
    pub(super) fn n_steps(&self) -> usize {
        let eliminated = (self.elimination.as_deref()).map_or(0, BlockElimination::n_eliminated);
        eliminated + self.cholesky.n_steps()
    }

    pub(super) fn scratch_len(&self) -> usize {
        self.cholesky.scratch_len()
    }

    /// Work `InnerParallelism::Allow` may split per solve, `0` when nothing does.
    pub(super) fn splittable_work(&self) -> usize {
        (self.elimination.as_deref()).map_or(0, BlockElimination::splittable_work)
    }

    /// `x ← M⁻¹x`, or its pseudo-inverse's up to the gauge when `M` is singular.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T], inner: InnerParallelism) {
        match &self.elimination {
            None => self.cholesky.solve(x, scratch),
            Some(elimination) => {
                elimination.solve(x, inner, |kept| self.cholesky.solve(kept, scratch));
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Factorization<T, I> {
    /// Whether it solves a component of `n` vertices: the elimination's sides, then the Cholesky on what it keeps.
    pub(super) fn solves(&self, n: usize) -> bool {
        let kept = match &self.elimination {
            None => Some(n),
            Some(elimination) => n
                .checked_sub(elimination.n_eliminated())
                .filter(|&kept| kept == elimination.n_kept()),
        };
        kept.is_some_and(|kept| kept > 0 && self.cholesky.solves(kept))
    }

    /// Whether it solves a singular Laplacian, and whether an elimination splits that solve.
    pub(super) fn singular(&self) -> (bool, bool) {
        (self.cholesky.singular(), self.elimination.is_some())
    }
}

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
    /// Dense on the exact arm, else approximate; `gauge` on its vertices.
    pub(super) fn factor(
        arm: Arm<T, I>,
        draws: Xoshiro256PlusPlus,
        gauge: Gauge,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let (sheets, grounded, n, pinned, triangular) = match arm {
            Arm::Exact(signed) => {
                let (grounded, n) = (!signed.floats(), signed.n());
                // A definite `M`'s surplus is the ground's slot, pinned so the solve is `M`'s own.
                let pin = pinned(Sheets::One, grounded, n, gauge);
                let factor =
                    Triangular::Dense(dense::factor(&signed, pin, &mut reused.dense).into());
                (Sheets::One, grounded, n, pin, factor)
            }
            Arm::Approximate(sddm) => {
                let (sheets, grounded, n) = (sddm.sheets(), sddm.grounded(), sddm.n());
                // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
                // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
                let pin =
                    (grounded || gauge.spans_wide).then(|| pinned(sheets, grounded, n, gauge));
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
    #[cfg(feature = "serde")]
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
fn pinned(sheets: Sheets, grounded: bool, n: usize, gauge: Gauge) -> usize {
    match grounded {
        true => n,
        false => first_copy(sheets, gauge.peak),
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
