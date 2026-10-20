use std::cell::LazyCell;

use num_traits::PrimInt;

use super::{within_band, Congruent, Signed, SymmetricBlock};
use crate::input::{Column, End, Gram, IntoColumns};
use crate::sgdd::block::{Balance, Grounding, IrreducibleBlock, ReduceError, Summands};
use crate::sgdd::decomposition::{assemble_values, Components, Shape};
use crate::sgdd::scaling::{self, ScalingPolicy};
use crate::sgdd::Sgdd;
use crate::{narrow, widen, Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// `V` with values: scaled by its null vectors on demand, folded blocks dropping their rows for one pass over `V` that decides floating.
    pub(super) fn from_values<J: PrimInt, G: Shape, A: End<T>, B: End<T>>(
        v: Gram<'_, T, J, G, A, B>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let Components {
            partition,
            summands,
        } = assemble_values::<T, J, I, G, A, B>(v)?
            .extract(|cells| Ok(SymmetricBlock::new(cells)))?;
        let Summands {
            blocks: live,
            lone,
            dead,
        } = summands;
        // Built by the first block not already dominant, so a dominant input reads no extra pass.
        let nulls = LazyCell::new(|| null_vectors(v, partition.components()));
        // Doubling, the folded blocks held up to 2x their size beside `live` (+12% peak at 99k blocks).
        let (mut folded, mut frustrated) = (Vec::with_capacity(live.len()), Vec::new());
        for a in live {
            // `V` proves a scaling exists.
            let reduced = a.reduce(policy, |a| {
                let u = (a.vertices.iter()).map(|&v| nulls[v.index()]);
                (u.clone().all(f64::is_finite))
                    .then(|| scaling::null_scaling(u, &a.rows.diagonal))
                    .flatten()
                    .or_else(|| scaling::relax(&a.off_diagonal, &a.rows.diagonal, policy))
            })?;
            match reduced {
                Signed::Folded(m, _) => folded.push(m),
                Signed::Frustrated(block) => frustrated.push(block),
            }
        }
        let mut blocks = ground_or_float(v, folded)?;
        blocks.extend(frustrated);
        let summands = Summands { blocks, lone, dead };
        Ok(Sgdd::new(Components {
            partition,
            summands,
        }))
    }
}

/// Floats each folded block as [`RowSums::folded_surplus`](super::RowSums::folded_surplus) does, else grounds it on `s = M·1`.
fn ground_or_float<T: Real, I: Index>(
    v: impl IntoColumns<T>,
    blocks: Vec<Congruent<T, I>>,
) -> Result<Vec<IrreducibleBlock<T, I>>, ReduceError> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    // By vertex: blocks are disjoint, so one slot per vertex holds its block, `f_i` and surplus.
    let mut block_of = vec![I::max_value(); v.n()];
    let mut factor = vec![0.0; v.n()];
    for (c, block) in blocks.iter().enumerate() {
        for (slot, &vertex) in block.vertices.iter().enumerate() {
            block_of[vertex.index()] = I::from_index(c);
            factor[vertex.index()] = widen(block.congruence.factor(slot));
        }
    }
    let mut surplus = vec![0.0f64; v.n()];
    // `ε·1ᵀM1` and `ε·tr M` per block.
    let (mut ones, mut trace) = (vec![0.0f64; blocks.len()], vec![0.0f64; blocks.len()]);
    let epsilon = widen(T::epsilon());
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            if column.weight.is_zero() {
                return;
            }
            let w = |x: f64, y: f64| column.weight.bilinear(x, y);
            let end = |i: I, v_i: T| match block_of[i.index()] {
                c if c == I::max_value() => None,
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
                // A column straddling blocks keeps only its own end's square.
                (a, b) => {
                    for (i, c, u) in [a, b].into_iter().flatten() {
                        surplus[i] += w(u, u);
                        ones[c] += w(epsilon * u, u);
                        trace[c] += w(epsilon * u, u);
                    }
                }
            }
        },
    );
    drop((block_of, factor));
    let decided = blocks.into_iter().zip(ones).zip(trace);
    let decide = |((block, ones), trace): ((Congruent<T, I>, f64), f64)| {
        if within_band::<T>(ones, trace) {
            return block.block(Balance::Balanced(Grounding::ungrounded()));
        }
        let vertices = &block.vertices;
        let surplus: Vec<T> = (vertices.iter())
            .map(|&v| narrow(surplus[v.index()].max(0.0)))
            .collect();
        // Unproven, a surplus that vanishes in `T` is never a float.
        let vertex = vertices[0].index();
        match Grounding::of(surplus) {
            grounding if grounding.is_grounded() => block.block(Balance::Balanced(grounding)),
            _ => Err(ReduceError::Unscalable { vertex }),
        }
    };
    decided.map(decide).collect()
}

/// Per vertex, a `u` orthogonal to its component's coupling columns, `NaN` where none is: one union-find pass; a cycle off by more than `√ε` relative dooms its component.
fn null_vectors<'c, T: Real, I: Index>(
    v: impl IntoColumns<T>,
    components: impl Iterator<Item = &'c [I]>,
) -> Vec<f64> {
    let n = v.n();
    let mut component = vec![I::zero(); n];
    // A lone vertex is dead or definite, never a candidate.
    let mut doomed = Vec::new();
    for (c, vertices) in components.enumerate() {
        vertices
            .iter()
            .for_each(|&v| component[v.index()] = I::from_index(c));
        doomed.push(vertices.len() == 1);
    }
    let mut alive = doomed.iter().filter(|&&doomed| !doomed).count();
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
    (0..n)
        .map(|i| match doomed[component[i].index()] {
            true => f64::NAN,
            false => forest.find(i).1,
        })
        .collect()
}

/// A column's component when both ends are nonzero in it.
fn coupling<T: Real, I: Index>(column: &Column<T, I>, component: &[I]) -> Option<usize> {
    let c = component[column.a.index()];
    let couples = !column.weight.is_zero()
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
