//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (GKS 2023, Algorithm 8, stars by Algorithm 6).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{CliqueTreeSampler, Copies, Fill, Multi, Single};
use crate::factor::sddm::{first_copy, Sddm};
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
    pub(in crate::factor) fn new(split: Option<Multi>) -> Self {
        Self(match split {
            None => ByCopies::Ac(Elimination::new(Single)),
            Some(split) => ByCopies::Ac2(Elimination::new(split)),
        })
    }

    /// Every vertex of `M` in minimum-degree order, each star replaced by a sampled clique tree, but a floating `M`'s last: returned as the slot the solve zeroes.
    pub(super) fn eliminate(
        &mut self,
        sddm: Sddm<T, I>,
        draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        match &mut self.0 {
            ByCopies::Ac(elimination) => elimination.eliminate(sddm, draws, tail),
            ByCopies::Ac2(elimination) => elimination.eliminate(sddm, draws, tail),
        }
    }
}

/// One copy count's elimination: `C` reaches every edge, so it is fixed per task.
struct Elimination<T, I, C: Copies> {
    split: C,
    graph: EliminationGraph<T, I, C>,
    clique: CliqueTreeSampler<T, I, C>,
    fill: Fill<T, I, C>,
    order: MinDegreeOrder<I>,
}

impl<T: Real, I: Index, C: Copies> Elimination<T, I, C> {
    fn new(split: C) -> Self {
        Self {
            split,
            graph: EliminationGraph::new(),
            clique: CliqueTreeSampler::new(split),
            fill: Fill::new(),
            order: MinDegreeOrder::new(),
        }
    }

    fn eliminate(
        &mut self,
        sddm: Sddm<T, I>,
        mut draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        let slots = sddm.n();
        let pin = match sddm.grounded() {
            // The ground is never a pivot, so `Σs` is never formed: within unchanged, grids +0-8% PCG iterations at 1-10 grounds.
            true => Some(slots),
            // Floating, ending at a small |f| left lili's PCG unconverged; forcing always cost +3-5% PCG iterations.
            false => {
                (tail.filter(|f| f.spans_wide())).map(|f| first_copy(sddm.sheets(), f.largest()))
            }
        };
        self.graph.load(&sddm, self.split);
        drop(sddm);
        let (n, edge_ends) = (self.graph.n(), self.graph.edge_ends());
        self.order.load(&self.graph, self.split, pin, edge_ends);
        let mut steps = ApproximateCholesky::with_capacity(n, edge_ends);
        self.pivot_all(&mut draws, &mut steps, slots);
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        steps.neighbors.shrink_to_fit();
        steps.coefficients.shrink_to_fit();
        let last = pin.unwrap_or_else(|| self.order.next_vertex());
        let pinned = (last < slots).then_some(last);
        if n > graph::REUSED_UP_TO {
            *self = Self::new(self.split);
        }
        (steps, pinned)
    }

    #[inline(never)] // Inlined, its codegen moves with the buffer handling: grids +1-3%.
    fn pivot_all(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        steps: &mut ApproximateCholesky<T, I>,
        slots: usize,
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
            let column = clique.eliminate(draws, graph.neighbors(v), fill);
            steps.push(v, column, slots);
            graph.eliminate_vertex(v);
            for &(u, w, copies, weight) in fill.iter() {
                graph.push_pair(u, w, copies, weight);
            }
            order.update(clique.degree_losses(), fill);
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
                    let mut lanes = [V::zero(); LANES];
                    lanes[0] = y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(LANES);
                    let mut cs = coefficients[span].chunks_exact(LANES);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..LANES {
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
const LANES: usize = 4;

#[cfg(feature = "serde")]
impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Each slot but `pin` pivots once, giving only to slots not yet pivoted, shares of at most its whole.
    pub(super) fn fits(&self, slots: usize, pin: Option<usize>) -> bool {
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
            if !later || !pivot || !within_one(shares.iter().copied()) {
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
    use crate::factor::within_one;

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

    fn pivot_defects<C: Copies>(split: C, label: &str, w: &[f32]) -> Option<String> {
        let mut clique = CliqueTreeSampler::<f32, u32, C>::new(split);
        let mut fill = Fill::new();
        let mut steps = ApproximateCholesky::<f32, u32>::with_capacity(w.len() + 1, w.len());
        let draws = &mut Xoshiro256PlusPlus::seed_from_u64(0);
        let star = (1..).zip(w).map(|(j, &w)| (j, split, w));
        steps.push(0, clique.eliminate(draws, star, &mut fill), w.len() + 1);
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
        clique.eliminate_unique(Xoshiro256PlusPlus::seed_from_u64(0), star, |e| fill.push(e));
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
            clique.eliminate(draws, star, &mut Fill::new()),
            w.len() + 1,
        );
        let d = w.iter().sum();
        let off = share_error(&steps, &w, d);
        assert!(off <= f64::EPSILON, "a share off by {off:e}");
    }
}
