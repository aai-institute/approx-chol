mod approximate;
mod block_elimination;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use super::clique_tree::Multi;
use super::config::{Config, InnerParallelism};
use super::schur::BipartiteBlock;
use super::sddm::{descend, first_copy, lift, Sddm};
use super::wdd::AnyWdd;
use crate::sgdd::Null;
use crate::storage::{
    widen, Adjacency, Balance, Bipartite, Gauge, Grounded, Index, OffDiagonal, Real, Sheets,
};
use approximate::{ApproximateCholesky, CliqueTreeElimination};
pub(super) use block_elimination::BlockElimination;

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

/// A block's factorization: a bipartite one's block elimination, then the Cholesky of what it leaves.
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
        Grounded {
            adjacency,
            grounding,
        }: Grounded<T, OffDiagonal<T, I>>,
        balance: Balance,
        gauge: Gauge,
        key: usize,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let streams = Streams::of(key, config);
        let draws = streams.cholesky();
        match adjacency {
            OffDiagonal::General(adjacency) => {
                let arm = match config.dense(adjacency.n()) {
                    true => Arm::Dense(AnyWdd::new(
                        Grounded {
                            adjacency,
                            grounding,
                        },
                        balance,
                    )),
                    false => {
                        Arm::Approximate(Sddm::of_block(adjacency, balance.sheets(), grounding))
                    }
                };
                Self {
                    elimination: None,
                    cholesky: Cholesky::factor(arm, draws, gauge, reused),
                }
            }
            OffDiagonal::Bipartite(cross) => {
                let factor = |s| Cholesky::factor(s, draws, gauge, reused);
                let block = BipartiteBlock::of(cross.into_cross(), balance, grounding);
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
        Grounded {
            adjacency: cross,
            grounding,
        }: Grounded<T, Bipartite<T, I>>,
        balance: Balance,
        gauge: Gauge,
        key: usize,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let streams = Streams::of(key, config);
        let block = BipartiteBlock::of(cross.into_cross(), balance, grounding);
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
    /// Present iff it solves a component of `n` vertices: the elimination's sides, then the Cholesky on what it keeps.
    pub(super) fn checked(self, n: usize) -> Option<Self> {
        let kept = match &self.elimination {
            None => n,
            Some(elimination) => n
                .checked_sub(elimination.n_eliminated())
                .filter(|&kept| kept == elimination.n_kept())?,
        };
        (kept > 0 && self.cholesky.solves(kept)).then(|| Self {
            elimination: self.elimination.map(|e| Box::new(e.rebuilt())),
            cholesky: self.cholesky,
        })
    }
}

impl<T: Real, I: Index> Factorization<T, I> {
    /// What its lift must do about `F·1`, read off what it factored.
    pub(super) fn null(&self) -> Null {
        Null::of(self.cholesky.singular(), self.elimination.is_some())
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

/// What a Cholesky factors: `M` itself on the dense arm, its SDDM on the approximate one.
pub(super) enum Arm<T, I> {
    Dense(AnyWdd<T, I>),
    Approximate(Sddm<T, I>),
}

/// The elimination's slots over `M`'s `n` vertices: `n` floating, `n + 1` with the ground `g` appended, `2n` on a cover's sheets, `g` after them when grounded.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    Floating,
    Grounded,
    Cover { grounded: bool },
}

impl Layout {
    fn of(sheets: Sheets, grounded: bool) -> Self {
        match (sheets, grounded) {
            (Sheets::One, false) => Self::Floating,
            (Sheets::One, true) => Self::Grounded,
            (Sheets::Two, grounded) => Self::Cover { grounded },
        }
    }

    fn grounded(self) -> bool {
        matches!(self, Self::Grounded | Self::Cover { grounded: true })
    }

    fn sheets(self) -> Sheets {
        match self {
            Self::Floating | Self::Grounded => Sheets::One,
            Self::Cover { .. } => Sheets::Two,
        }
    }

    fn slots(self, n: usize) -> usize {
        self.sheets().count() * n + usize::from(self.grounded())
    }

    /// `M`'s vertices over `slots`, if they fit this layout.
    #[cfg(feature = "serde")]
    fn vertices(self, slots: usize) -> Option<usize> {
        let sheets = slots.checked_sub(usize::from(self.grounded()))?;
        let count = self.sheets().count();
        (sheets % count == 0 && sheets > 0).then_some(sheets / count)
    }

    /// The slot the solve pins: the ground, else the largest `|f|` on the first sheet, so the gauge's rounding is not amplified by `1/f`.
    fn pinned(self, n: usize, gauge: Gauge) -> usize {
        match self.grounded() {
            true => self.slots(n) - 1,
            false => first_copy(self.sheets(), gauge.peak),
        }
    }
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T, I> {
    layout: Layout,
    /// The one slot no pivot eliminates, zeroed between the sweeps; the ground when there is one.
    pinned: usize,
    triangular: Triangular<T, I>,
}

impl<T: Real, I: Index> Cholesky<T, I> {
    /// Dense on the dense arm, else approximate; `gauge` on its vertices.
    pub(super) fn factor(
        arm: Arm<T, I>,
        draws: Xoshiro256PlusPlus,
        gauge: Gauge,
        reused: &mut Workspace<T, I>,
    ) -> Self {
        let (layout, pinned, triangular) = match arm {
            Arm::Dense(wdd) => {
                let layout = Layout::of(Sheets::One, !wdd.floats());
                // A definite `M`'s surplus is the ground's slot, pinned so the solve is `M`'s own.
                let pin = layout.pinned(wdd.n(), gauge);
                let factor = dense::factor(&wdd, pin, &mut reused.dense);
                (layout, pin, Triangular::Dense(factor.into()))
            }
            Arm::Approximate(sddm) => {
                let (sheets, grounded) = (sddm.sheets(), sddm.grounded());
                let layout = Layout::of(sheets, grounded);
                let n = sddm.n() / sheets.count();
                // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
                // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
                let pin = (grounded || gauge.spans_wide).then(|| layout.pinned(n, gauge));
                let (steps, pin) = reused.elimination.eliminate(sddm, draws, pin);
                (layout, pin, Triangular::Approximate(Box::new(steps)))
            }
        };
        Self {
            layout,
            pinned,
            triangular,
        }
    }

    /// Pivots eliminated: every slot but the pinned one.
    pub(super) fn n_steps(&self) -> usize {
        self.triangular.slots() - 1
    }

    /// Whether it solves a singular Laplacian: one sheet, no ground.
    pub(super) fn singular(&self) -> bool {
        self.layout == Layout::Floating
    }

    /// A floating sheet substitutes on `x` in place; any other layout on its slots beside it.
    pub(super) fn scratch_len(&self) -> usize {
        match self.layout {
            Layout::Floating => 0,
            Layout::Grounded | Layout::Cover { .. } => self.triangular.slots(),
        }
    }

    /// Lifts to the slots, substitutes, and descends; the pinned ground reads zero, so its right-hand side is never read.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        let triangular = &self.triangular;
        match self.layout {
            Layout::Floating => triangular.substitute(x, self.pinned),
            Layout::Grounded => {
                let rows = &mut scratch[..=x.len()];
                rows[..x.len()].copy_from_slice(x);
                triangular.substitute(rows, self.pinned);
                x.copy_from_slice(&rows[..x.len()]);
            }
            Layout::Cover { .. } => {
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
    /// Every slot pivots but the pinned one.
    fn slots(&self) -> usize {
        match self {
            Self::Dense(factor) => dense::slots(factor),
            Self::Approximate(approximate) => approximate.n_steps() + 1,
        }
    }

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
    /// The slots its factor holds fit its layout, and a ground is the slot it pins.
    fn checked(self) -> Option<Self> {
        let slots = self.triangular.slots();
        self.layout.vertices(slots)?;
        let fits = match &self.triangular {
            Triangular::Dense(dense) => dense::fits(dense, slots),
            Triangular::Approximate(approximate) => approximate.fits(slots, self.pinned),
        };
        let pinned = match self.layout.grounded() {
            true => self.pinned == slots - 1,
            false => self.pinned < slots,
        };
        (pinned && fits).then_some(self)
    }

    /// Whether it solves a block of `kept` vertices on its sheets.
    pub(super) fn solves(&self, kept: usize) -> bool {
        self.layout.vertices(self.triangular.slots()) == Some(kept)
    }
}
