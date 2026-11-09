//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (GKS 2023, Algorithm 8, stars by Algorithm 6).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{CliqueTreeSampler, Copies, Fill, Multi, Single};
use crate::factor::sign::Sign;
#[cfg(feature = "serde")]
use crate::factor::within_one;
use crate::sgdd::FactorTail;
use crate::{narrow, Batch, Index, Real};
use graph::EliminationGraph;
use ordering::MinDegreeOrder;

/// A task's AC or AC2 elimination; small blocks reuse its buffers (malloc was 17% of 20-vertex covers).
pub(crate) struct CliqueTreeElimination<T, I>(ByCopies<T, I>);

enum ByCopies<T, I> {
    Ac(Elimination<T, I, Single>),
    Ac2(Elimination<T, I, Multi>),
}

impl<T: Real, I: Index> CliqueTreeElimination<T, I> {
    pub(in crate::factor) fn new(copies: Option<Multi>) -> Self {
        Self(match copies {
            None => ByCopies::Ac(Elimination::new(Single)),
            Some(copies) => ByCopies::Ac2(Elimination::new(copies)),
        })
    }

    /// Every vertex of `M` in minimum-degree order, each star replaced by a sampled clique tree, but a floating `M`'s last: returned as the slot the solve zeroes.
    pub(super) fn eliminate<M: Eliminated<T, I>>(
        &mut self,
        m: M,
        draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        match &mut self.0 {
            ByCopies::Ac(elimination) => elimination.eliminate(m, draws, tail),
            ByCopies::Ac2(elimination) => elimination.eliminate(m, draws, tail),
        }
    }
}

/// What an elimination eliminates: its edges, the ground past its slots, how its weights' signs read, and the slot it leaves.
pub(in crate::factor) trait Eliminated<T: Real, I: Index> {
    type Sign: Sign<T, I>;

    /// Its slots; a ground is the vertex past them.
    fn n(&self) -> usize;

    /// Each edge once as `(i, j, −a_ij)`, the ground's as `(k, n, s_k)`.
    fn grounded_edges(&self, edge: impl FnMut(I, I, T));

    fn sign(&self) -> Self::Sign;

    /// The vertex no pivot eliminates: the ground, the vertex past the slots, or a floating matrix's gauge, by `tail`.
    fn pin(&self, tail: Option<FactorTail<'_, T>>) -> Option<usize>;
}

/// A block this small keeps its buffers for the task's next, its graph's slab never compacted; a larger one frees them.
const REUSED_UP_TO: usize = 1024;

/// One copy count's elimination: `C` reaches every edge, so it is fixed per task.
struct Elimination<T, I, C: Copies> {
    copies: C,
    graph: EliminationGraph<T, I, C>,
    clique: CliqueTreeSampler<T, I, C>,
    fill: Fill<T, I, C>,
    order: MinDegreeOrder<I>,
}

impl<T: Real, I: Index, C: Copies> Elimination<T, I, C> {
    fn new(copies: C) -> Self {
        Self {
            copies,
            graph: EliminationGraph::new(),
            clique: CliqueTreeSampler::new(copies),
            fill: Fill::new(),
            order: MinDegreeOrder::new(),
        }
    }

    fn eliminate<M: Eliminated<T, I>>(
        &mut self,
        m: M,
        mut draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        let (slots, pin) = (m.n(), m.pin(tail));
        self.graph
            .load(&m, slots + usize::from(pin == Some(slots)), self.copies);
        let sign = m.sign();
        drop(m);
        let (n, edge_ends) = (self.graph.n(), self.graph.edge_ends());
        self.order.load(&self.graph, self.copies, pin, edge_ends);
        let mut steps = ApproximateCholesky::with_capacity(n, edge_ends);
        self.pivot_all(&mut draws, &mut steps, slots, sign);
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        steps.neighbors.shrink_to_fit();
        steps.coefficients.shrink_to_fit();
        let last = pin.unwrap_or_else(|| self.order.next_vertex());
        let pinned = (last < slots).then_some(last);
        if n > REUSED_UP_TO {
            *self = Self::new(self.copies);
        }
        (steps, pinned)
    }

    #[inline(never)] // Inlined, its codegen moves with the buffer handling: grids +1-3%.
    fn pivot_all<S: Sign<T, I>>(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        steps: &mut ApproximateCholesky<T, I>,
        slots: usize,
        sign: S,
    ) {
        let Self {
            graph,
            clique,
            fill,
            order,
            ..
        } = self;
        for _ in 1..graph.n() {
            let v = order.next_vertex();
            fill.clear();
            let column = clique.eliminate(draws, graph.neighbors(v), fill, sign);
            steps.push(v, column, slots);
            graph.eliminate_vertex(v);
            for &(u, w, copies, weight) in fill.iter() {
                graph.push_pair(u, w, copies, weight);
            }
            order.update(graph, clique.neighbors());
            // Its own pass: chained onto the neighbors, it cost unsigned 16-vertex paths +5-8%.
            order.update(graph, clique.dropped());
        }
    }
}

/// A step's neighbors follow the previous step's, so the solve's running `usize` offset finds them at any size.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
struct StepHeader<T, I> {
    vertex: I,
    /// Below `n`, so `I` always holds it and a `u32` header stays 16 bytes.
    len: I,
    inverse_degree: T,
}

/// `LDLᵀ` with sampled columns, appended in pivot order: a serial recurrence, so shares are composed against the original pivot.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct ApproximateCholesky<T, I> {
    headers: Vec<StepHeader<T, I>>,
    neighbors: Vec<I>,
    coefficients: Vec<T>,
}

impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Growing to `entries` by doubling measured +1.5-2.5% on grids.
    fn with_capacity(n: usize, entries: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(entries),
            coefficients: Vec::with_capacity(entries),
        }
    }

    /// Each share `w/d` and `D⁻¹` rounded once from `f64`: no share is derived from the others, so none cancels (#275, #302); only `M`'s `slots` take one, not the ground past them.
    fn push(
        &mut self,
        vertex: usize,
        (d, shares): (f64, impl Iterator<Item = (I, f64)>),
        slots: usize,
    ) {
        let start = self.neighbors.len();
        for (neighbor, share) in shares.filter(|&(j, _)| j.index() < slots) {
            self.neighbors.push(neighbor);
            self.coefficients.push(narrow(share));
        }
        // Finite, as fill and `S` never weigh below `MIN_POSITIVE`.
        self.headers.push(StepHeader {
            vertex: I::from_index(vertex),
            len: I::from_index(self.neighbors.len() - start),
            inverse_degree: narrow(1.0 / d),
        });
    }

    /// Forward, zero a floating factor's `pinned` slot, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(super) fn substitute<V: Batch<T>>(&self, y: &mut [V], pinned: Option<usize>) {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        let mut end = 0;
        for header in headers {
            let span = end..end + header.len.index();
            end = span.end;
            let pivot = y[header.vertex.index()];
            match coefficients[span.clone()] {
                // A lone neighbor takes the whole pivot unless the ground shared it; a multiply on this serial chain cost paths +24%.
                [c] if c == T::one() => {
                    let j = neighbors[span.start].index();
                    y[j] = y[j].add(pivot);
                }
                _ => {
                    for (&j, &c) in neighbors[span.clone()].iter().zip(&coefficients[span]) {
                        y[j.index()] = y[j.index()].add(pivot.mul(c));
                    }
                }
            }
            y[header.vertex.index()] = pivot.mul(header.inverse_degree);
        }
        if let Some(pinned) = pinned {
            y[pinned] = V::zero();
        }
        for header in headers.iter().rev() {
            let span = end - header.len.index()..end;
            end = span.start;
            let v = header.vertex.index();
            match coefficients[span.clone()] {
                [] => {}
                [c] if c == T::one() => y[v] = y[v].add(y[neighbors[span.start].index()]),
                [c] => y[v] = y[v].add(y[neighbors[span.start].index()].mul(c)),
                _ => {
                    let mut lanes = [V::zero(); SPLIT];
                    lanes[0] = y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(SPLIT);
                    let mut cs = coefficients[span].chunks_exact(SPLIT);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..SPLIT {
                            lanes[lane] = lanes[lane].add(y[js[lane].index()].mul(cs[lane]));
                        }
                    }
                    let mut total = lanes.iter().fold(V::zero(), |sum, &lane| sum.add(lane));
                    for (&j, &c) in js.remainder().iter().zip(cs.remainder()) {
                        total = total.add(y[j.index()].mul(c));
                    }
                    y[v] = total;
                }
            }
        }
    }
}

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const SPLIT: usize = 4;

#[cfg(feature = "serde")]
impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Each slot but `pin` pivots once, giving only to slots not yet pivoted, shares `S` admits of at most its whole.
    pub(super) fn fits<S: Sign<T, I>>(&self, slots: usize, pin: Option<usize>) -> bool {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        if headers.len() + usize::from(pin.is_some()) != slots
            || neighbors.len() != coefficients.len()
        {
            return false;
        }
        let mut pivoted = vec![false; slots];
        let mut end = 0usize;
        for header in headers {
            let (v, start) = (header.vertex.index(), end);
            end = end.saturating_add(header.len.index());
            if Some(v) == pin
                || v >= slots
                || std::mem::replace(&mut pivoted[v], true)
                || end > neighbors.len()
            {
                return false;
            }
            let later = neighbors[start..end]
                .iter()
                .all(|&j| j.index() < slots && !pivoted[j.index()]);
            let shares = &coefficients[start..end];
            let pivot = header.inverse_degree.is_finite() && header.inverse_degree > T::zero();
            let admitted = shares.iter().all(|&c| S::admits(c));
            if !later || !pivot || !admitted || !within_one(shares.iter().copied()) {
                return false;
            }
        }
        end == neighbors.len()
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::factor::clique_tree::{Multi, Single};
    use crate::factor::config::CoverSurplus;
    use crate::factor::sddm::{Sddm, SignedWdd};
    use crate::factor::sign::Unsigned;
    use crate::factor::within_one;
    use crate::sgdd::{Balance, Grounding};
    use crate::storage::{Rows, UpperAdjacency};
    use crate::widen;

    fn stars(m: usize) -> [(&'static str, Vec<f32>); 3] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let log_uniform = |_| ((rng.next_u64() >> 40) as f32 / (1 << 21) as f32 - 4.0).exp2();
        [
            ("0.1", vec![0.1; m]),
            ("0.7", vec![0.7; m]),
            ("log-uniform", (0..m).map(log_uniform).collect()),
        ]
    }

    /// Worst relative error of each entry's summed fill against `w_i S_{>i} / d`, and `d`, both in `f64`.
    fn fill_error<C: Copies>(w: &[f32], fill: &Fill<f32, u32, C>) -> (f64, f64) {
        let mut order: Vec<u32> = (1..=w.len() as u32).collect();
        order.sort_by(|&a, &b| {
            w[a as usize - 1]
                .total_cmp(&w[b as usize - 1])
                .then(a.cmp(&b))
        });
        let mut position = vec![0; w.len() + 1];
        order
            .iter()
            .enumerate()
            .for_each(|(p, &j)| position[j as usize] = p);
        let mut sums = vec![0.0; w.len()];
        for &(lo, hi, _, weight) in fill {
            sums[position[lo as usize].min(position[hi as usize])] += f64::from(weight);
        }
        let sorted: Vec<f64> = order
            .iter()
            .map(|&j| f64::from(w[j as usize - 1]))
            .collect();
        let d: f64 = sorted.iter().sum();
        let mut suffix = d;
        let worst = (sorted.iter().zip(&sums).take(w.len() - 1))
            .map(|(&wi, &sum)| {
                suffix -= wi;
                (sum / (wi * suffix / d) - 1.0).abs()
            })
            .fold(0.0, f64::max);
        (worst, d)
    }

    /// Worst relative error of every stored share, the heaviest's included, against `w_j / d`; `w_j` is neighbor `j`'s weight, `j ≥ 1`.
    fn share_error<T: Real>(steps: &ApproximateCholesky<T, u32>, w: &[f64], d: f64) -> f64 {
        assert_eq!(steps.coefficients.len(), w.len(), "every neighbor's share");
        (steps.neighbors.iter().zip(&steps.coefficients))
            .map(|(&j, &c)| (crate::widen(c) / (w[j as usize - 1] / d) - 1.0).abs())
            .fold(0.0, f64::max)
    }

    fn pivot_defects<C: Copies>(copies: C, label: &str, w: &[f32]) -> Option<String> {
        let mut clique = CliqueTreeSampler::<f32, u32, C>::new(copies);
        let mut fill = Fill::new();
        let mut steps = ApproximateCholesky::<f32, u32>::with_capacity(w.len() + 1, w.len());
        let draws = &mut Xoshiro256PlusPlus::seed_from_u64(0);
        let star = (1..).zip(w).map(|(j, &w)| (j, copies, w));
        steps.push(
            0,
            clique.eliminate(draws, star, &mut fill, Unsigned),
            w.len() + 1,
        );
        let (fill_off, d) = fill_error(w, &fill);
        let off = (f64::from(steps.headers[0].inverse_degree) * d - 1.0).abs();
        let w: Vec<f64> = w.iter().copied().map(f64::from).collect();
        let shares_off = share_error(&steps, &w, d);
        let bounded = within_one(steps.coefficients.iter().copied());
        (off >= 1e-6 || shares_off >= 1e-6 || fill_off >= 1e-5 || !bounded).then(|| {
            format!("{label}: D⁻¹ off {off:e}, shares off {shares_off:e}, fill off {fill_off:e}, Σ|c| ≤ 1 {bounded}")
        })
    }

    fn unique_fill_defects(label: &str, w: &[f32]) -> Option<String> {
        let mut clique = CliqueTreeSampler::<f32, u32, Single>::new(Single);
        let mut fill = Fill::new();
        let star = (1..).zip(w.iter().copied());
        clique.eliminate_unique(
            Xoshiro256PlusPlus::seed_from_u64(0),
            star,
            |e| fill.push(e),
            Unsigned,
        );
        let (fill_off, _) = fill_error(w, &fill);
        (fill_off >= 1e-5).then(|| format!("{label}: fill off {fill_off:e}"))
    }

    #[test]
    fn high_degree_f32_star_keeps_its_inverse_degree_and_fill() {
        let ac2 = Multi::split(2).expect("k = 2 is AC2");
        let mut defects = Vec::new();
        for m in [6000, 20000, 50000] {
            for (kind, w) in stars(m) {
                let label = |arm| format!("{arm} m {m} w {kind}");
                defects.extend(pivot_defects(Single, &label("AC"), &w));
                defects.extend(pivot_defects(ac2, &label("AC2"), &w));
                defects.extend(unique_fill_defects(&label("unique"), &w));
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }

    /// At degree 10⁵ the heaviest share as `1 − Σ` erred 1.9e-7 relative in `f64` (#306); as `w/d` it rounds once.
    #[test]
    fn a_high_degree_f64_star_rounds_each_share_once() {
        let w = vec![0.1; 100_000];
        let mut clique = CliqueTreeSampler::<f64, u32, Single>::new(Single);
        let mut steps = ApproximateCholesky::<f64, u32>::with_capacity(w.len() + 1, w.len());
        let star = (1..).zip(&w).map(|(j, &w)| (j, Single, w));
        let draws = &mut Xoshiro256PlusPlus::seed_from_u64(0);
        steps.push(
            0,
            clique.eliminate(draws, star, &mut Fill::new(), Unsigned),
            w.len() + 1,
        );
        let d = w.iter().sum();
        let off = share_error(&steps, &w, d);
        assert!(off <= f64::EPSILON, "a share off by {off:e}");
    }

    /// A grounded Z-matrix on `n` vertices, a spanning path and random edges, log-uniform; its `D`-switched copy by random `σ`; and `σ`.
    fn switched<T: Real>(n: usize, seed: u64) -> (Sddm<T, u32>, SignedWdd<T, u32>, Vec<bool>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut pairs: Vec<(usize, usize)> = (1..n).map(|v| (v - 1, v)).collect();
        for _ in 0..2 * n {
            let (a, b) = (rng.next_u64() as usize % n, rng.next_u64() as usize % n);
            pairs.extend((a != b).then_some((a.min(b), a.max(b))));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let flip: Vec<bool> = (0..n).map(|_| rng.next_u64() % 2 == 1).collect();
        let (mut ptrs, mut indices, mut z, mut signed) = (vec![0], vec![], vec![], vec![]);
        for i in 0..n {
            for &(_, j) in pairs.iter().filter(|&&(a, _)| a == i) {
                let w: T = narrow(((rng.next_u64() % 1600) as f64 / 100.0 - 8.0).exp2());
                indices.push(j as u32);
                z.push(-w);
                signed.push(if flip[i] != flip[j] { w } else { -w });
            }
            ptrs.push(indices.len());
        }
        let surplus: Vec<T> = (0..n)
            .map(|v| narrow(if v % 5 == 0 { 0.5 } else { 0.0 }))
            .collect();
        let adjacency =
            |values| UpperAdjacency::new(Rows::new(ptrs.clone(), indices.clone(), values, n));
        let sddm = Sddm::of_block(
            adjacency(z),
            Balance::Balanced,
            Grounding::of(surplus.clone()),
            CoverSurplus::Auto,
        );
        (
            sddm,
            SignedWdd::new(adjacency(signed), Grounding::of(surplus)),
            flip,
        )
    }

    /// Magnitudes, draws and order agree, so only each share's sign moves, by `σ_v σ_j`.
    fn switched_defects<T: Real>(seed: u64, copies: Option<Multi>) -> Option<String> {
        let (sddm, signed, flip) = switched::<T>(80, seed);
        let draws = Xoshiro256PlusPlus::seed_from_u64(seed);
        let (z, z_pin) = CliqueTreeElimination::new(copies).eliminate(sddm, draws.clone(), None);
        let (s, s_pin) = CliqueTreeElimination::new(copies).eliminate(signed, draws, None);
        let bits = |c: T| widen(c).to_bits();
        let headers = |steps: &ApproximateCholesky<T, u32>| -> Vec<_> {
            (steps.headers.iter())
                .map(|h| (h.vertex, h.len, bits(h.inverse_degree)))
                .collect()
        };
        let mut end = 0;
        let mut flipped = Vec::new();
        for h in &z.headers {
            let v = h.vertex as usize;
            for k in end..end + h.len as usize {
                let c = z.coefficients[k];
                flipped.push(bits(if flip[v] != flip[z.neighbors[k] as usize] {
                    -c
                } else {
                    c
                }));
            }
            end += h.len as usize;
        }
        let same = z_pin.is_none()
            && s_pin.is_none()
            && headers(&z) == headers(&s)
            && z.neighbors == s.neighbors
            && flipped == s.coefficients.iter().map(|&c| bits(c)).collect::<Vec<_>>();
        (!same).then(|| format!("seed {seed}, {} copies", copies.map_or(1, |c| c.get())))
    }

    #[test]
    fn a_switched_z_matrix_eliminates_signed_as_its_seed_does_unsigned() {
        let ac2 = Multi::split(2).expect("AC2");
        let mut defects = Vec::new();
        for seed in 0..16 {
            for copies in [None, Some(ac2)] {
                defects.extend(switched_defects::<f64>(seed, copies));
                defects.extend(switched_defects::<f32>(seed, copies).map(|d| d + ", f32"));
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }
}
