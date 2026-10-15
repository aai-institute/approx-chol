use std::cell::LazyCell;

use num_traits::PrimInt;

use super::{Congruent, Signed, SymmetricBlock, SINGULAR_BAND};
use crate::input::{BipartiteGram, FactorWidth2, IntoColumns};
use crate::sgdd::block::{Grounding, IrreducibleBlock, ReduceError, Sgdd};
use crate::sgdd::decomposition::{assemble_bipartite_gram, assemble_general, Components};
use crate::sgdd::null_vectors::null_vectors;
use crate::sgdd::scaling::ScalingPolicy;
use crate::{narrow, widen, Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// `A = V Vᵀ`: scales, folds or covers each block as general, deciding floating from `V`.
    pub fn from_gram<J: PrimInt>(
        v: FactorWidth2<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let blocks = assemble_general(v)?.extract(|cells| Ok(SymmetricBlock::live(cells)))?;
        reduce_gram_blocks(v, blocks, &policy)
    }

    /// `A = V W Vᵀ` on two disjoint sides `[D_l C; Cᵀ D_r]`, as [`from_gram`](Self::from_gram).
    pub fn from_bipartite_gram<J: PrimInt>(
        gram: BipartiteGram<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        // Incidence on both sides folds by shape, so `policy` has nothing to scale.
        if let Some(laplacian) = gram.laplacian() {
            return Self::try_from(laplacian);
        }
        let blocks =
            assemble_bipartite_gram(gram)?.extract(|cells| Ok(SymmetricBlock::live(cells)))?;
        reduce_gram_blocks(gram, blocks, &policy)
    }
}

/// Scaled by `V`'s null vectors on demand; folded blocks drop their rows for one pass over `V`.
fn reduce_gram_blocks<T: Real, I: Index>(
    v: impl IntoColumns<T>,
    (partition, live, dead): Components<SymmetricBlock<T, I>, I>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T, I>, ReduceError> {
    // Built by the first block not already dominant, so a dominant input reads no extra pass.
    let nulls = LazyCell::new(|| null_vectors(v, partition.components()));
    let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
    for a in live {
        match a.reduce_gram(policy, &nulls)? {
            Signed::Folded(m, _) => folded.push(m),
            Signed::Frustrated(block) => frustrated.push(block),
        }
    }
    let mut blocks = ground_or_float(v, folded)?;
    blocks.extend(frustrated);
    Ok(Sgdd::new(partition.n(), blocks, dead))
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
        if ones <= SINGULAR_BAND * epsilon * trace {
            return block.balanced(Grounding::ungrounded());
        }
        let vertices = &block.vertices;
        let surplus: Vec<T> = (vertices.iter())
            .map(|&v| narrow(surplus[v.index()].max(0.0)))
            .collect();
        // Unproven, a surplus that vanishes in `T` is never a float.
        let vertex = vertices[0].index();
        match Grounding::of(surplus) {
            grounding if grounding.is_grounded() => block.balanced(grounding),
            _ => Err(ReduceError::Unscalable { vertex }),
        }
    };
    decided.map(decide).collect()
}
