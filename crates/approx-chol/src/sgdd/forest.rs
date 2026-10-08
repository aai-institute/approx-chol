//! A Gram's components and their gauge, read from its columns, never from rounded totals.

use super::bipartite::Gram;
use super::cross::Block;
use super::Gauge;
use num_traits::PrimInt;

// `ρ ≥ λ_min`, so a well-conditioned block grounds; an exact alias reads `ρ ~ ε²`.
const FLOATING_RAYLEIGH_BOUND: f64 = 1e-13;

// 10u: a scaling that misses the kernel leaves surplus that floating would drop.
const SCALED_FLOATING_BOUND: f64 = 5.0 * f64::EPSILON;

/// A component's levels, each side ascending.
#[derive(Clone, Debug, Default)]
pub(crate) struct Members {
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
}

/// `Floating` iff a folded component's scaling `f` has `fᵀAf ≤ 10u·fᵀDf`, summed as squares per column.
pub(crate) fn scaled_gauges<I: PrimInt>(
    gram: &Gram<'_, I>,
    // Per level `[rows | cols]`: the tested component and its factor.
    levels: &[Option<(usize, f64)>],
    n_rows: usize,
    n_components: usize,
) -> Vec<Gauge> {
    let mut residual = vec![0.0; n_components];
    let mut diagonal = vec![0.0; n_components];
    for o in (0..gram.n_columns()).map(|k| gram.observation(k)) {
        if o.w == 0.0 {
            continue;
        }
        match (levels[o.row], levels[n_rows + o.col]) {
            (Some((k, fi)), Some((l, fj))) if k == l => {
                let (p, q) = (fi * o.x, fj * o.y);
                residual[k] += o.w * (p + q) * (p + q);
                diagonal[k] += o.w * (p * p + q * q);
            }
            (row, col) => {
                for (k, p) in [
                    row.map(|(k, f)| (k, f * o.x)),
                    col.map(|(k, f)| (k, f * o.y)),
                ]
                .into_iter()
                .flatten()
                {
                    residual[k] += o.w * p * p;
                    diagonal[k] += o.w * p * p;
                }
            }
        }
    }
    residual
        .iter()
        .zip(&diagonal)
        .map(|(&r, &d)| {
            if d.is_finite() && r <= SCALED_FLOATING_BOUND * d {
                Gauge::Floating
            } else {
                Gauge::Grounded
            }
        })
        .collect()
}

/// Union-find over the levels `[rows | cols]` of `C`, scaled so `|x|·u_i = |y|·u_j` on each join.
pub(crate) struct ScaledForest {
    n_rows: usize,
    parent: Vec<u32>,
    /// `u_node / u_parent`: the null vector along the joins, when one exists.
    scale: Vec<f64>,
    size: Vec<u32>,
    /// Per root: its set's `uᵀ(D − |C|)u` and `uᵀDu`, in the root's units.
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl ScaledForest {
    /// One pass over the columns, then one over the cells for their cancellation.
    pub(crate) fn grow<I: PrimInt>(gram: &Gram<'_, I>, c: &Block) -> Self {
        let n_local = c.nrows + c.ncols;
        let mut forest = Self {
            n_rows: c.nrows,
            parent: (0..n_local as u32).collect(),
            scale: vec![1.0; n_local],
            size: vec![1; n_local],
            numerator: vec![0.0; n_local],
            denominator: vec![0.0; n_local],
        };
        let mut magnitude = vec![0.0; c.nnz()];
        // uᵀ(D − |C|)u = Σ_o w(|x|u_i − |y|u_j)² + 2Σ_cells u_i·u_j(Σ|wxy| − |Σwxy|).
        for o in (0..gram.n_columns()).map(|k| gram.observation(k)) {
            if o.w == 0.0 {
                continue;
            }
            let (i, j) = (o.row, c.nrows + o.col);
            let stored = c.span(o.row);
            let cell = (o.x != 0.0 && o.y != 0.0)
                .then(|| {
                    c.indices[stored.clone()]
                        .binary_search(&(o.col as u32))
                        .ok()
                })
                .flatten()
                .map(|offset| stored.start + offset);
            let ((ri, ui), (rj, uj)) = match cell {
                Some(p) => {
                    magnitude[p] += o.cell().abs();
                    forest.join(i, j, o.x.abs() / o.y.abs())
                }
                None => (forest.find(i), forest.find(j)),
            };
            let (a, b) = (o.x.abs() * ui, o.y.abs() * uj);
            if cell.is_some() {
                forest.numerator[ri] += o.w * (a - b) * (a - b);
            } else {
                forest.numerator[ri] += o.w * a * a;
                forest.numerator[rj] += o.w * b * b;
            }
            forest.denominator[ri] += o.w * a * a;
            forest.denominator[rj] += o.w * b * b;
        }
        for i in 0..c.nrows {
            let cells = c.span(i);
            let (root, ui) = forest.find(i);
            for ((j, value), &magnitude) in c.row(i).zip(&magnitude[cells]) {
                let cancelled = magnitude - value.abs();
                if cancelled > 0.0 {
                    let (_, uj) = forest.find(c.nrows + j);
                    forest.numerator[root] += 2.0 * ui * uj * cancelled;
                }
            }
        }
        forest
    }

    /// The root of `node`'s set and `u_node` in its units, compressing the path.
    fn find(&mut self, node: usize) -> (usize, f64) {
        let parent = self.parent[node] as usize;
        if parent == node {
            return (node, 1.0);
        }
        // Union by size leaves most levels one link from their root; those need no rewrite.
        if self.parent[parent] as usize == parent {
            return (parent, self.scale[node]);
        }
        let (root, u_parent) = self.find(parent);
        let u = self.scale[node] * u_parent;
        (self.parent[node], self.scale[node]) = (root as u32, u);
        (root, u)
    }

    /// Joins the sets of `i` and `j` so that `u_j = ratio·u_i`, unless they already share one.
    fn join(&mut self, i: usize, j: usize, ratio: f64) -> ((usize, f64), (usize, f64)) {
        let ((ri, ui), (rj, uj)) = (self.find(i), self.find(j));
        if ri == rj {
            return ((ri, ui), (rj, uj));
        }
        // Union by size keeps the paths short; the absorbed root's sums change units by `s²`.
        let (root, absorbed, s, joined) = if self.size[ri] >= self.size[rj] {
            (ri, rj, ui * ratio / uj, ((ri, ui), (ri, ui * ratio)))
        } else {
            (rj, ri, uj / (ui * ratio), ((rj, uj / ratio), (rj, uj)))
        };
        (self.parent[absorbed], self.scale[absorbed]) = (root as u32, s);
        self.size[root] += self.size[absorbed];
        self.numerator[root] += self.numerator[absorbed] * s * s;
        self.denominator[root] += self.denominator[absorbed] * s * s;
        joined
    }

    /// Each set with its root, ordered by lowest level and with sorted members.
    pub(crate) fn components(&mut self) -> Vec<(usize, Members)> {
        let mut slot = vec![usize::MAX; self.parent.len()];
        let mut components: Vec<(usize, Members)> = Vec::new();
        for node in 0..self.parent.len() {
            let (root, _) = self.find(node);
            if slot[root] == usize::MAX {
                slot[root] = components.len();
                components.push((root, Members::default()));
            }
            let (_, members) = &mut components[slot[root]];
            if node < self.n_rows {
                members.rows.push(node);
            } else {
                members.cols.push(node - self.n_rows);
            }
        }
        components
    }

    /// `Floating` iff `ρ = uᵀ(D − |C|)u / uᵀDu ≤ λ*` over the set of `root`.
    pub(crate) fn gauge(&self, root: usize) -> Gauge {
        let (numerator, denominator) = (self.numerator[root], self.denominator[root]);
        // An overflowed `u` leaves the alias unproven, so it grounds.
        if denominator.is_finite() && numerator / denominator <= FLOATING_RAYLEIGH_BOUND {
            Gauge::Floating
        } else {
            Gauge::Grounded
        }
    }
}

/// Components of a graph's `C` by DFS from each unvisited level, members sorted.
pub(crate) fn graph_components(cross: &super::cross::Cross) -> Vec<Members> {
    let n_rows = cross.n_rows();
    let mut visited = vec![false; cross.n_local()];
    let mut components = Vec::new();
    let mut stack = Vec::new();
    for start in 0..cross.n_local() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        stack.push(start);
        let mut members = Members::default();
        while let Some(node) = stack.pop() {
            if node < n_rows {
                members.rows.push(node);
            } else {
                members.cols.push(node - n_rows);
            }
            for (j, _) in cross.neighbors(node) {
                if !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        members.rows.sort_unstable();
        members.cols.sort_unstable();
        components.push(members);
    }
    components
}
