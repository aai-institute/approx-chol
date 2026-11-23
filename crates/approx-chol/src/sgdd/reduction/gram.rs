use std::cell::LazyCell;
use std::ops::Deref;
use std::sync::OnceLock;

use num_traits::PrimInt;

use super::{rows_where, Folded, Settling, Surplus};
use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::{Grounding, IrreducibleBlock};
use crate::sgdd::decomposition::{assemble_values, AdjacencyLayout, Cut, SymmetricBlock};
use crate::sgdd::forest::{Forest, Potential};
use crate::sgdd::scaling::{self, ScalingPolicy};
use crate::sgdd::{Components, Reduced};
use crate::{narrow, widen, BuildError, Index, Real, Threads};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` with values: scaled by its null vectors on demand, folded blocks dropping their rows for a pass over `V` that decides floating.
    pub(super) fn from_values<J: PrimInt + Sync, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        let (cuts, components) = assemble_values::<T, J, I, S>(cells, v)?.arrange()?;
        components.cut(|components| {
            // Built by whichever block first fails dominance: a dominant input pays only once a block's band fails.
            let nulls = OnceLock::<(Vec<f64>, Vec<I>)>::new();
            let live = |vertices, entries| SymmetricBlock { vertices, entries };
            let signed = cuts.extract_then(components, threads, live, |a, threads| {
                // `V` proves a scaling exists.
                a.reduce(policy, |a| {
                    // Serial: a rayon task stolen inside `get_or_init` could re-enter it.
                    let (u, _) = nulls.get_or_init(|| null_vectors(v, components));
                    // `NaN` where a component has no certificate.
                    let u = (a.vertices.iter()).map(|v| u[v.index()]);
                    let relax = || {
                        scaling::relax(
                            &a.entries.off_diagonal,
                            &a.entries.rows.diagonal,
                            policy,
                            threads,
                        )
                    };
                    (u.clone().all(f64::is_finite))
                        .then_some(u)
                        .and_then(|u| scaling::null_scaling(u, &a.entries.rows.diagonal))
                        .map_or_else(relax, Ok)
                })
            })?;
            // `u` freed before the column pass: only the certified set outlives the scaling.
            let read = nulls.into_inner().map(|(_, certified)| certified);
            let certified = LazyCell::new(|| {
                read.unwrap_or_else(|| null_vectors::<f64, _, _, _>(v, components).1)
            });
            // `ρ` in `u = u*(1 + ρ)` by vertex, walked again only once a null-scaled row is past its band.
            let rho = LazyCell::new(|| null_roundings(v, components));
            ground_or_float(v, signed, &certified, &rho)
        })
    }
}
/// Summed from `V`'s columns, with what one-ended columns state and the components whose cycles `V` certifies.
struct GramSurplus<'a, I, F> {
    /// Each row's `(M1)_i`.
    surplus: &'a [f64],
    /// Each row's ground stated by one-ended columns.
    stated: &'a [f64],
    /// Sorted lowest vertices of certified components, built on first use.
    certified: &'a LazyCell<Vec<I>, F>,
    /// Each vertex's `ρ` in its null vector `u = u*(1 + ρ)`, built on first use.
    rho: &'a dyn Deref<Target = Vec<f64>>,
}

impl<T: Real, I: Index, F: FnOnce() -> Vec<I>> Surplus<T, I> for GramSurplus<'_, I, F> {
    fn residues<'a>(
        &'a self,
        folded: &'a Folded<'_, T, I>,
    ) -> impl Iterator<Item = f64> + Clone + 'a {
        (folded.m.vertices().iter()).map(|v| self.surplus[v.index()])
    }

    fn stated(&self, folded: &Folded<'_, T, I>) -> Option<Box<[bool]>> {
        let stated = (folded.m.vertices().iter()).map(|v| self.stated[v.index()]);
        let diagonal = folded.rows.diagonal.iter().map(|&d| widen(d));
        // A stated ground sums without cancellation: the band of no entry.
        let past =
            (folded.vector).past_band(widen(T::epsilon()), stated, diagonal, (0, Vec::new))?;
        rows_where(folded.rows.diagonal.len(), past.map(|ratio| ratio > 0.0))
    }

    // A Gram's grounds are stated or break a cycle: a scaled block floats only on its certificate.
    fn certifier(&self) -> Option<impl Fn(&[I]) -> bool + '_> {
        Some(|vertices: &[I]| {
            (vertices.iter().min()).is_some_and(|v| self.certified.binary_search(v).is_ok())
        })
    }

    fn reported(&self, folded: &Folded<'_, T, I>) -> f64 {
        folded.beyond_null_rounding(|| &self.rho[..])
    }

    // Every rounded surplus ≤ 0 leaves nothing to ground: out of band, only past degree ~2·10⁸ (#299).
    fn grounds(&self, folded: &Folded<'_, T, I>) -> Grounding<T> {
        Grounding::of(
            (folded.m.vertices().iter())
                .map(|v| narrow(self.surplus[v.index()].max(0.0)))
                .collect(),
        )
    }
}

/// Each block in turn, a folded one settled by [`Folded::settle`] on one pass over `V`'s columns, which none takes when none is folded.
fn ground_or_float<'p, T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    settling: Cut<'p, I, T, Settling<'p, T, I>>,
    certified: &LazyCell<Vec<I>, impl FnOnce() -> Vec<I>>,
    rho: &dyn Deref<Target = Vec<f64>>,
) -> Result<Cut<'p, I, T, IrreducibleBlock<T, I>>, BuildError> {
    let blocks = settling.blocks();
    // By vertex, sized only when a block is folded: blocks are disjoint, so one slot per vertex holds its block, `f_i` and surplus.
    let folds = blocks.iter().any(|b| matches!(b, Settling::Pending(_)));
    let n = if folds { v.n() } else { 0 };
    let (mut block_of, mut factor) = (vec![I::max_value(); n], vec![0.0; n]);
    for (c, block) in blocks.iter().enumerate() {
        let Settling::Pending(Folded { m: block, .. }) = block else {
            continue;
        };
        for (slot, vertex) in block.vertices().iter().enumerate() {
            block_of[vertex.index()] = I::from_index(c);
            factor[vertex.index()] = widen(block.factor(slot));
        }
    }
    let mut surplus = vec![0.0f64; n];
    // One-ended columns: data, not a cancelling row sum.
    let mut stated = vec![0.0f64; n];
    // `ε·1ᵀM1` and `ε·tr M` per block, a frustrated one's unused.
    let (mut ones, mut trace) = (vec![0.0f64; blocks.len()], vec![0.0f64; blocks.len()]);
    if folds {
        let epsilon = widen(T::epsilon());
        v.for_each_column::<I>(
            #[inline(always)]
            |column| {
                if column.zero_weight() {
                    return;
                }
                let w = |x: f64, y: f64| column.bilinear(x, y);
                let end = |i: I, v_i: T| match block_of[i.index()] {
                    c if c == I::max_value() || v_i == T::zero() => None,
                    c => Some((i.index(), c.index(), factor[i.index()] * widen(v_i))),
                };
                match (end(column.a, column.v_a), end(column.b, column.v_b)) {
                    (Some((a, c, u_a)), Some((b, c_b, u_b))) if c == c_b => {
                        let p = u_a + u_b;
                        surplus[a] += w(u_a, p);
                        surplus[b] += w(u_b, p);
                        ones[c] += w(epsilon * p, p);
                        trace[c] += w(epsilon * u_a, u_a) + w(epsilon * u_b, u_b);
                    }
                    // One-ended, a column states a ground of its end's own square.
                    (a, b) => {
                        for (i, c, u) in [a, b].into_iter().flatten() {
                            surplus[i] += w(u, u);
                            stated[i] += w(u, u);
                            ones[c] += w(epsilon * u, u);
                            trace[c] += w(epsilon * u, u);
                        }
                    }
                }
            },
        );
    }
    drop((block_of, factor));
    // `|u_i·p| ≤ (u_i² + p²)/2`: a running sum past `f64::MAX` puts `m_ii` or `1ᵀM1` past range.
    let in_range = |&s: &f64| s.is_finite() && narrow::<T>(s.max(0.0)).is_finite();
    if let Some(vertex) = surplus.iter().position(|s| !in_range(s)) {
        return Err(BuildError::OutOfRange { vertex });
    }
    let gram = GramSurplus {
        surplus: &surplus,
        stated: &stated,
        certified,
        rho,
    };
    settling.try_map(|c, block| match block {
        Settling::Pending(folded) => folded.settle((ones[c], trace[c]), &gram),
        Settling::Settled(block) => Ok(block),
    })
}

/// Per vertex, `ρ` in `u = u*(1 + ρ)` of [`null_vectors`]' `u`: its walk again, on the same bits, each ratio tracked.
fn null_roundings<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    components: &Components<I, T>,
) -> Vec<f64> {
    let (u, _) = null_vectors::<Tracked, T, J, I>(v, components);
    // Collected afresh: in place, it would keep `u`'s twice-as-large allocation.
    u.iter().map(|u| u.rounding).collect()
}

/// Per vertex, a `u` orthogonal to its component's coupling columns, `NaN` where none is, and the sorted lowest vertices of the components that have one: one union-find pass; a cycle off by more than `√ε` relative dooms its component.
fn null_vectors<P, T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    components: &Components<I, T>,
) -> (Vec<P>, Vec<I>)
where
    P: Potential<Link = Ratio<P>> + From<f64> + Into<f64>,
    Ratio<P>: Into<P>,
{
    let (n, coupled) = (v.n(), components.coupled.len());
    // Lone and dead vertices share one doomed label: each is dead or definite, never a candidate.
    let mut component = vec![I::from_index(coupled); n];
    let mut doomed = vec![false; coupled + 1];
    doomed[coupled] = true;
    let mut lowest = Vec::with_capacity(coupled);
    for k in 0..coupled {
        let vertices = components.component(k);
        vertices
            .iter()
            .for_each(|v| component[v.index()] = I::from_index(k));
        lowest.push(vertices.iter().min().copied());
    }
    let mut alive = coupled;
    let tolerance = widen(T::epsilon()).sqrt();
    let mut forest = Forest::<I, P>::new(n);
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            if alive == 0 {
                return;
            }
            let Some(c) = coupling(&column, &component) else {
                return;
            };
            let (num, den) = (P::from(-widen(column.v_a)), P::from(widen(column.v_b)));
            let k: P = Ratio { num, den }.into();
            let (a, b) = (column.a.index(), column.b.index());
            if !doomed[c] && !join(&mut forest, a, b, k, tolerance) {
                doomed[c] = true;
                alive -= 1;
            }
        },
    );
    let u = (0..n)
        .map(|i| match doomed[component[i].index()] {
            true => P::from(f64::NAN),
            false => forest.find(i).1,
        })
        .collect();
    let mut certified: Vec<I> = (lowest.into_iter().zip(doomed))
        .filter_map(|(lowest, doomed)| lowest.filter(|_| !doomed))
        .collect();
    certified.sort_unstable();
    (u, certified)
}

/// A column's component when both ends are nonzero in it.
fn coupling<T: Real, I: Index>(column: &Column<T, I>, component: &[I]) -> Option<usize> {
    let c = component[column.a.index()];
    let couples = !column.zero_weight()
        && column.v_a != T::zero()
        && column.v_b != T::zero()
        && component[column.b.index()] == c;
    couples.then_some(c.index())
}

/// `u_b / u_a` across a link from `b` up to `a`, undivided: reversing swaps it exactly, and a closing cycle compares its halves.
pub(in crate::sgdd) struct Ratio<P> {
    num: P,
    den: P,
}

impl<P: Into<f64>> Ratio<P> {
    fn agrees(self, tolerance: f64) -> bool {
        let (num, den): (f64, f64) = (self.num.into(), self.den.into());
        (num - den).abs() <= tolerance * num.abs().max(den.abs())
    }
}

impl From<Ratio<f64>> for f64 {
    fn from(r: Ratio<f64>) -> Self {
        r.num / r.den
    }
}

impl Potential for f64 {
    const ROOT: Self = 1.0;
    type Link = Ratio<Self>;
    fn then(self, up: Self) -> Self {
        self * up
    }
    fn reversed(Ratio { num, den }: Ratio<Self>) -> Ratio<Self> {
        Ratio { num: den, den: num }
    }
}

/// Below it an operation's own rounding may underflow, so FMA no longer recovers it.
const TRACKED_FLOOR: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// A ratio of `u` as `f64` computes it, and its rounding to first order, `value = exact·(1 + rounding)`: FMA recovers each operation's own, `NaN` once one may have underflowed.
#[derive(Clone, Copy)]
struct Tracked {
    value: f64,
    rounding: f64,
}

impl From<f64> for Tracked {
    fn from(value: f64) -> Self {
        let rounding = 0.0;
        Self { value, rounding }
    }
}

impl From<Tracked> for f64 {
    fn from(x: Tracked) -> Self {
        x.value
    }
}

impl From<Ratio<Tracked>> for Tracked {
    fn from(Ratio { num, den }: Ratio<Tracked>) -> Self {
        let value = num.value / den.value;
        let own = match value.abs().min(num.value.abs()) >= TRACKED_FLOOR {
            true => value.mul_add(den.value, -num.value) / num.value,
            false => f64::NAN,
        };
        let rounding = num.rounding - den.rounding + own;
        Self { value, rounding }
    }
}

impl Potential for Tracked {
    const ROOT: Self = Self {
        value: 1.0,
        rounding: 0.0,
    };
    type Link = Ratio<Self>;
    fn then(self, up: Self) -> Self {
        let value = self.value * up.value;
        let own = match value.abs() >= TRACKED_FLOOR {
            true => -self.value.mul_add(up.value, -value) / value,
            false => f64::NAN,
        };
        let rounding = self.rounding + up.rounding + own;
        Self { value, rounding }
    }
    fn reversed(Ratio { num, den }: Ratio<Self>) -> Ratio<Self> {
        Ratio { num: den, den: num }
    }
}

/// Joins `u_b = k·u_a` in a forest whose potentials are `u_child / u_parent`; `false` on a cycle missing by more than `tolerance`, relative, or a ratio leaving the finite nonzero range.
fn join<I: Index, P>(forest: &mut Forest<I, P>, a: usize, b: usize, k: P, tolerance: f64) -> bool
where
    P: Potential<Link = Ratio<P>> + Into<f64>,
{
    // An overflowed `v_a/v_b` would compare `∞ ≤ tolerance·∞` and pass.
    if !k.into().is_finite() {
        return false;
    }
    let ((ra, xa), (rb, xb)) = (forest.find(a), forest.find(b));
    // `u_rb / u_ra = k·xa/xb`.
    let ratio = Ratio {
        num: k.then(xa),
        den: xb,
    };
    if ra == rb {
        return ratio.agrees(tolerance);
    }
    let (_, stored) = forest.link(ra, rb, ratio);
    let stored: f64 = stored.into();
    stored.is_finite() && stored != 0.0
}
