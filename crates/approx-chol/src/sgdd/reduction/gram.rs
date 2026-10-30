use std::cell::LazyCell;

use num_traits::PrimInt;

use super::{within_band, Folded, Signed, SymmetricBlock};
use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::{Balance, Grounding, IrreducibleBlock, ReduceError};
use crate::sgdd::decomposition::{assemble_values, AdjacencyLayout, Components, Summands};
use crate::sgdd::scaling::{self, ScaleVector, ScalingPolicy, RATIO_BAND};
use crate::sgdd::Reduced;
use crate::{narrow, widen, Index, Real};

impl<T: Real, I: Index> Reduced<T, I> {
    /// `V` with values: scaled by its null vectors on demand, folded blocks dropping their rows for a pass over `V` that decides floating.
    pub(super) fn from_values<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let Components {
            partition,
            summands,
        } = assemble_values::<T, J, I, S>(cells, v)?
            .extract(|cells| Ok(SymmetricBlock::new(cells)))?;
        let Summands {
            blocks: live,
            lone,
            dead,
        } = summands;
        // Built by the first block not already dominant: a dominant input pays only once a block's band fails.
        let mut nulls: Option<(Vec<f64>, Certified<I>)> = None;
        // Doubling, the folded blocks held up to 2x their size beside `live` (+12% peak at 99k blocks).
        let (mut folded, mut frustrated) = (Vec::with_capacity(live.len()), Vec::new());
        for a in live {
            // `V` proves a scaling exists.
            let reduced = a.reduce(policy, |a| {
                let (u, _) = nulls
                    .get_or_insert_with(|| Cycles::of(v, partition.components()).null_vectors());
                // `NaN` where a component has no certificate.
                let u = (a.vertices.iter()).map(|&v| u[v.index()]);
                (u.clone().all(f64::is_finite))
                    .then(|| scaling::null_scaling(u, &a.rows.diagonal))
                    .flatten()
                    .or_else(|| scaling::relax(&a.off_diagonal, &a.rows.diagonal, policy))
            })?;
            match reduced {
                Signed::Folded(block) => folded.push(block),
                Signed::Frustrated(block) => frustrated.push(block),
            }
        }
        // `u` freed before the column pass: only the certified set outlives the scaling.
        let read = nulls.map(|(_, certified)| certified);
        let certified = LazyCell::new(|| {
            read.unwrap_or_else(|| Cycles::of(v, partition.components()).null_vectors().1)
        });
        let mut blocks = ground_or_float(v, folded, &certified)?;
        drop(certified);
        blocks.extend(frustrated);
        let summands = Summands { blocks, lone, dead };
        Ok(Components {
            partition,
            summands,
        }
        .into_reduced())
    }
}

/// Grounds a folded block on `s = M·1` unless, in band and stating no ground, its cycles close or, dominant, [`Folded::floats`] admits it.
fn ground_or_float<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    blocks: Vec<Folded<T, I>>,
    certified: &LazyCell<Certified<I>, impl FnOnce() -> Certified<I>>,
) -> Result<Vec<IrreducibleBlock<T, I>>, ReduceError> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    // By vertex: blocks are disjoint, so one slot per vertex holds its block, `f_i` and surplus.
    let mut block_of = vec![I::max_value(); v.n()];
    let mut factor = vec![0.0; v.n()];
    for (c, Folded { m: block, .. }) in blocks.iter().enumerate() {
        for (slot, &vertex) in block.vertices.iter().enumerate() {
            block_of[vertex.index()] = I::from_index(c);
            factor[vertex.index()] = widen(block.congruence.factor(slot));
        }
    }
    let mut surplus = vec![0.0f64; v.n()];
    // One-ended columns: data, not a cancelling row sum.
    let mut stated = vec![0.0f64; v.n()];
    // `ε·1ᵀM1` and `ε·tr M` per block.
    let (mut ones, mut trace) = (vec![0.0f64; blocks.len()], vec![0.0f64; blocks.len()]);
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
    drop((block_of, factor));
    // `|u_i·p| ≤ (u_i² + p²)/2`: a running sum past `f64::MAX` puts `m_ii` or `1ᵀM1` past range.
    let in_range = |&s: &f64| s.is_finite() && narrow::<T>(s.max(0.0)).is_finite();
    if let Some(vertex) = surplus.iter().position(|s| !in_range(s)) {
        return Err(ReduceError::OutOfRange { vertex });
    }
    let decided = blocks.into_iter().zip(ones).zip(trace);
    (decided.map(|((folded, ones), trace)| {
        let rows = (folded.m.vertices.iter()).map(|v| v.index());
        // A stated ground sums without cancellation, so the floor ignores degree.
        let stated_ground = (rows.clone().zip(&folded.rows.diagonal))
            .any(|(v, &d)| stated[v] > RATIO_BAND * epsilon * widen(d));
        let in_band = within_band::<T>(ones, trace);
        let dominant = matches!(folded.vector, ScaleVector::Identity);
        // A Gram's grounds are stated or break a cycle: closing all, `V` is singular within `√ε`.
        let floats = !stated_ground
            && in_band
            && ((dominant && folded.floats(rows.map(|v| surplus[v])))
                || certified.of(&folded.m.vertices));
        let block = folded.m;
        if floats {
            return block.block(Balance::Balanced(Grounding::ungrounded()));
        }
        let vertices = &block.vertices;
        let grounds: Vec<T> = (vertices.iter())
            .map(|&v| narrow(surplus[v.index()].max(0.0)))
            .collect();
        // Every rounded surplus ≤ 0 leaves nothing to ground: out of band, only past degree ~2·10⁸ (#299).
        let vertex = vertices[0].index();
        match Grounding::of(grounds) {
            grounding if grounding.is_grounded() => block.block(Balance::Balanced(grounding)),
            _ => Err(ReduceError::Unscalable { vertex }),
        }
    }))
    .collect()
}

/// Lowest vertices of the components whose cycles close within `√ε`, `u` in range, sorted.
struct Certified<I>(Vec<I>);

impl<I: Index> Certified<I> {
    /// Keyed by the lowest vertex, so neither order matters.
    fn of(&self, vertices: &[I]) -> bool {
        (vertices.iter().min()).is_some_and(|v| self.0.binary_search(v).is_ok())
    }
}

/// One union-find pass over `V`'s coupling columns: whether each component's cycles close.
struct Cycles<I> {
    forest: RatioForest<I>,
    component: Vec<I>,
    closes: Vec<bool>,
    /// `None` for a lone vertex, dead or definite, never a candidate.
    lowest: Vec<Option<I>>,
}

impl<I: Index> Cycles<I> {
    fn of<'c, T: Real, J: PrimInt>(
        v: FactorWidth2<'_, T, J>,
        components: impl Iterator<Item = &'c [I]>,
    ) -> Self
    where
        I: 'c,
    {
        let n = v.n();
        let mut component = vec![I::zero(); n];
        let (mut closes, mut lowest) = (Vec::new(), Vec::new());
        for (c, vertices) in components.enumerate() {
            vertices
                .iter()
                .for_each(|&v| component[v.index()] = I::from_index(c));
            let candidate = vertices.len() > 1;
            closes.push(candidate);
            lowest.push(vertices.iter().min().copied().filter(|_| candidate));
        }
        let mut alive = closes.iter().filter(|&&closes| closes).count();
        let mut forest = RatioForest::<I>::new(n, widen(T::epsilon()).sqrt());
        v.for_each_column::<I>(
            #[inline(always)]
            |column| {
                if alive == 0 {
                    return;
                }
                let Some(c) = coupling(&column, &component).filter(|&c| closes[c]) else {
                    return;
                };
                let k = -widen(column.v_a) / widen(column.v_b);
                if !forest.join(column.a.index(), column.b.index(), k) {
                    closes[c] = false;
                    alive -= 1;
                }
            },
        );
        Self {
            forest,
            component,
            closes,
            lowest,
        }
    }

    /// Per vertex, a `u` orthogonal to its component's coupling columns, `NaN` unless certified; a `u` past range uncertifies.
    fn null_vectors(mut self) -> (Vec<f64>, Certified<I>) {
        let n = self.component.len();
        let mut u: Vec<f64> = (0..n)
            .map(|i| match self.closes[self.component[i].index()] {
                true => self.forest.find(i).1,
                false => f64::NAN,
            })
            .collect();
        for (i, &x) in u.iter().enumerate() {
            if !(x.is_finite() && x != 0.0) {
                self.closes[self.component[i].index()] = false;
            }
        }
        (u.iter_mut().enumerate())
            .filter(|&(i, _)| !self.closes[self.component[i].index()])
            .for_each(|(_, x)| *x = f64::NAN);
        let mut certified: Vec<I> = (self.lowest.into_iter().zip(self.closes))
            .filter_map(|(lowest, closes)| lowest.filter(|_| closes))
            .collect();
        certified.sort_unstable();
        (u, Certified(certified))
    }
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

/// `u_child / u_parent`, `1` at a root, where `size` counts its tree.
#[derive(Clone, Copy)]
struct Link<I> {
    ratio: f64,
    parent: I,
    size: I,
}

/// Union-find over [`Link`]s, packed so a path step reads one line.
struct RatioForest<I> {
    links: Vec<Link<I>>,
    /// How far a closing cycle may miss, relative.
    tolerance: f64,
}

impl<I: Index> RatioForest<I> {
    fn new(n: usize, tolerance: f64) -> Self {
        let root = |i| Link {
            ratio: 1.0,
            parent: I::from_index(i),
            size: I::one(),
        };
        Self {
            links: (0..n).map(root).collect(),
            tolerance,
        }
    }

    /// Root and `u_i / u_root`, halving the path.
    fn find(&mut self, mut i: usize) -> (usize, f64) {
        let mut x = 1.0;
        loop {
            let link = self.links[i];
            let parent = link.parent.index();
            if parent == i {
                return (i, x);
            }
            let up = self.links[parent];
            let ratio = link.ratio * up.ratio;
            self.links[i] = Link {
                ratio,
                parent: up.parent,
                ..link
            };
            x *= ratio;
            i = up.parent.index();
        }
    }

    /// Joins `u_b = k·u_a`; `false` on a cycle missing past the tolerance or a ratio or product leaving `f64`'s range.
    fn join(&mut self, a: usize, b: usize, k: f64) -> bool {
        // An overflowed `v_a/v_b` would compare `∞ ≤ tolerance·∞` and pass.
        if !k.is_finite() {
            return false;
        }
        let ((ra, xa), (rb, xb)) = (self.find(a), self.find(b));
        if ra == rb {
            let (want, have) = (k * xa, xb);
            // A product past range, `0` or `∞`, compares `NaN` and fails.
            return (want - have).abs() / want.abs().max(have.abs()) <= self.tolerance;
        }
        // `u_rb = (k·xa/xb)·u_ra`, inverted when `ra` goes under `rb`.
        let (big, small, ratio) = match self.links[ra].size >= self.links[rb].size {
            true => (ra, rb, k * xa / xb),
            false => (rb, ra, xb / (k * xa)),
        };
        let size = self.links[small].size;
        self.links[big].size = self.links[big].size + size;
        self.links[small] = Link {
            ratio,
            parent: I::from_index(big),
            size,
        };
        ratio.is_finite() && ratio != 0.0
    }
}
