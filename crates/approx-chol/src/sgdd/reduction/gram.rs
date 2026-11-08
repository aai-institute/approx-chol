use std::cell::LazyCell;

use num_traits::PrimInt;

use super::{Folded, Settling, Surplus};
use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::{Grounding, IrreducibleBlock, ReduceError};
use crate::sgdd::decomposition::{assemble_values, AdjacencyLayout, SymmetricBlock};
use crate::sgdd::scaling::{self, ScalingPolicy, RATIO_BAND};
use crate::sgdd::{Components, Reduced, Summands};
use crate::{narrow, widen, Index, Real};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` with values: scaled by its null vectors on demand, folded blocks dropping their rows for a pass over `V` that decides floating.
    pub(super) fn from_values<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let (cuts, components) = assemble_values::<T, J, I, S>(cells, v)?.arrange()?;
        let live = cuts.extract(&components, |vertices, entries| {
            Ok(SymmetricBlock { vertices, entries })
        })?;
        // Built by the first block not already dominant: a dominant input pays only once a block's band fails.
        let mut nulls: Option<(Vec<f64>, Vec<I>)> = None;
        // Doubling, the folded blocks held up to 2x their size beside `live` (+12% peak at 99k blocks).
        let mut signed = Vec::with_capacity(live.len());
        for a in live {
            // `V` proves a scaling exists.
            signed.push(a.reduce(policy, |a| {
                let (u, _) = nulls.get_or_insert_with(|| null_vectors(v, &components));
                // `NaN` where a component has no certificate.
                let u = (a.vertices.iter()).map(|v| u[v.index()]);
                (u.clone().all(f64::is_finite))
                    .then(|| scaling::null_scaling(u, &a.entries.rows.diagonal))
                    .flatten()
                    .or_else(|| {
                        scaling::relax(&a.entries.off_diagonal, &a.entries.rows.diagonal, policy)
                    })
            })?);
        }
        // `u` freed before the column pass: only the certified set outlives the scaling.
        let read = nulls.map(|(_, certified)| certified);
        let certified = LazyCell::new(|| read.unwrap_or_else(|| null_vectors(v, &components).1));
        let blocks = ground_or_float(v, signed, &certified)?;
        drop(certified);
        Ok(Summands { components, blocks })
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
}

impl<T: Real, I: Index, F: FnOnce() -> Vec<I>> Surplus<T, I> for GramSurplus<'_, I, F> {
    // A Gram's grounds are stated or break a cycle: a scaled block floats only on its certificate.
    const RESIDUES_FLOAT_SCALED: bool = false;

    fn residues<'a>(
        &'a self,
        folded: &'a Folded<'_, T, I>,
    ) -> impl Iterator<Item = f64> + Clone + 'a {
        (folded.m.vertices().iter()).map(|v| self.surplus[v.index()])
    }

    fn states_ground(&self, folded: &Folded<'_, T, I>) -> bool {
        let epsilon = widen(T::epsilon());
        // A stated ground sums without cancellation, so the floor ignores degree.
        (folded.m.vertices().iter().zip(&folded.rows.diagonal))
            .any(|(v, &d)| self.stated[v.index()] > RATIO_BAND * epsilon * widen(d))
    }

    fn certifies(&self, vertices: &[I]) -> bool {
        (vertices.iter().min()).is_some_and(|v| self.certified.binary_search(v).is_ok())
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
fn ground_or_float<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    blocks: Vec<Settling<'_, T, I>>,
    certified: &LazyCell<Vec<I>, impl FnOnce() -> Vec<I>>,
) -> Result<Vec<IrreducibleBlock<T, I>>, ReduceError> {
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
        return Err(ReduceError::OutOfRange { vertex });
    }
    let gram = GramSurplus {
        surplus: &surplus,
        stated: &stated,
        certified,
    };
    (blocks.into_iter().zip(ones.into_iter().zip(trace)))
        .map(|(block, band)| match block {
            Settling::Pending(folded) => folded.settle(band, &gram),
            Settling::Settled(block) => Ok(block),
        })
        .collect()
}

/// Per vertex, a `u` orthogonal to its component's coupling columns, `NaN` where none is, and the sorted lowest vertices of the components that have one: one union-find pass; a cycle off by more than `√ε` relative dooms its component.
fn null_vectors<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    components: &Components<I, T>,
) -> (Vec<f64>, Vec<I>) {
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
    let mut forest = RatioForest::<I>::new(n, widen(T::epsilon()).sqrt());
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            if alive == 0 {
                return;
            }
            let Some(c) = coupling(&column, &component) else {
                return;
            };
            let k = -widen(column.v_a) / widen(column.v_b);
            if !doomed[c] && !forest.join(column.a.index(), column.b.index(), k) {
                doomed[c] = true;
                alive -= 1;
            }
        },
    );
    let u = (0..n)
        .map(|i| match doomed[component[i].index()] {
            true => f64::NAN,
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

/// Union-find whose links carry `u_child / u_parent`.
struct RatioForest<I> {
    parent: Vec<I>,
    /// `1` at a root.
    ratio: Vec<f64>,
    size: Vec<I>,
    /// How far a closing cycle may miss, relative.
    tolerance: f64,
}

impl<I: Index> RatioForest<I> {
    fn new(n: usize, tolerance: f64) -> Self {
        Self {
            parent: (0..n).map(I::from_index).collect(),
            ratio: vec![1.0; n],
            size: vec![I::one(); n],
            tolerance,
        }
    }

    /// Root and `u_i / u_root`, halving the path.
    fn find(&mut self, mut i: usize) -> (usize, f64) {
        let mut x = 1.0;
        while self.parent[i].index() != i {
            let up = self.parent[i].index();
            self.ratio[i] *= self.ratio[up];
            self.parent[i] = self.parent[up];
            x *= self.ratio[i];
            i = self.parent[i].index();
        }
        (i, x)
    }

    /// Joins `u_b = k·u_a`; `false` on a broken cycle or a ratio leaving the finite nonzero range.
    fn join(&mut self, a: usize, b: usize, k: f64) -> bool {
        // An overflowed `v_a/v_b` would compare `∞ ≤ tolerance·∞` and pass.
        if !k.is_finite() {
            return false;
        }
        let ((ra, xa), (rb, xb)) = (self.find(a), self.find(b));
        if ra == rb {
            let (want, have) = (k * xa, xb);
            return (want - have).abs() <= self.tolerance * want.abs().max(have.abs());
        }
        // `u_rb = (k·xa/xb)·u_ra`, inverted when `ra` goes under `rb`.
        let (big, small, ratio) = match self.size[ra] >= self.size[rb] {
            true => (ra, rb, k * xa / xb),
            false => (rb, ra, xb / (k * xa)),
        };
        self.parent[small] = I::from_index(big);
        self.ratio[small] = ratio;
        self.size[big] = self.size[big] + self.size[small];
        ratio.is_finite() && ratio != 0.0
    }
}
